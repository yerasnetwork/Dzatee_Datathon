# =============================================================================
# fc_pipeline.py — ForteBank Datathon 26: контроль кассовой дисциплины по видео
# Локальная VLM (Qwen3-VL, Apache-2.0) + LoRA. Этот же файл пойдёт в сервис.
# =============================================================================
import os, re, io, json, math, base64, hashlib, random, subprocess, time, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

DEFAULTS = dict(
    model_id="Qwen/Qwen3-VL-4B-Instruct", load_4bit=True,
    # окна и кадры
    fps=2.0, win_sec=8.0, long_side=576,
    pos_stride=1.0, neg_stride=2.0, infer_stride=2.0,
    margin=1.5, pre_gap=3.0, post_as_neg=False,
    ramp=True, ramp_stride=1.0,    # окна «нарастания» с мягкой целью 0->1 между start_sec и официальным таймкодом
    # обучение
    epochs=2, lr=1e-4, grad_accum=8, neg_ratio=1.5, verdict_w=3.0,
    lora_r=16, lora_alpha=32, lora_dropout=0.05, seed=42,
    # анализ
    max_windows=48, time_budget_sec=200, cluster_gap=5.0, max_events=3,
)


def make_cfg(**kw):
    d = dict(DEFAULTS)
    unknown = set(kw) - set(d)
    if unknown:
        raise ValueError(f"неизвестные параметры: {sorted(unknown)}")
    d.update(kw)
    return SimpleNamespace(**d)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ разметка
def vid_num(name):
    """'Х-22.mov' / 'X-1' / '#U0425-47.mov' -> номер видео (последняя группа цифр)."""
    s = re.sub(r"#U[0-9A-Fa-f]{4}", "", str(name))
    s = os.path.splitext(s)[0] if s.lower().endswith(tuple(VIDEO_EXT)) else s
    nums = re.findall(r"\d+", s)
    if not nums:
        raise ValueError(f"не нашёл номер видео в имени: {name!r}")
    return int(nums[-1])


def parse_time(s):
    """'22 секунда' -> 22; '1минута 8 секунда' -> 68; '1:08' -> 68; 12.5 -> 12.5."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return None if (isinstance(s, float) and math.isnan(s)) else float(s)
    t = str(s).strip().lower().replace(",", ".")
    if not t or t in ("nan", "none", "-"):
        return None
    m = re.fullmatch(r"(\d+):(\d{1,2})(?:\.(\d+))?", t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + (float("0." + m.group(3)) if m.group(3) else 0.0)
    m = re.search(r"(?:(\d+)\s*мин\w*)?\s*(\d+(?:\.\d+)?)\s*сек", t)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    try:
        return float(t)
    except ValueError:
        return None


def _clean_key(x):
    return re.sub(r"[^a-z0-9_]+", "_", str(x).strip().lower()).strip("_")


def read_official_labels(xlsx):
    """Официальный open_labels.xlsx -> {номер: {'violation': bool, 't': сек|None}}."""
    import pandas as pd
    df = pd.read_excel(xlsx, header=0)
    out = {}
    for _, r in df.iterrows():
        name, ts, lab = r.iloc[0], r.iloc[1], r.iloc[2]
        if pd.isna(name):
            continue
        lab = str(lab).lower()
        out[vid_num(name)] = dict(violation=("наруш" in lab and "без" not in lab), t=parse_time(None if pd.isna(ts) else ts))
    return out


DEFAULT_TYPES = {
    "violation": dict(name_ru="Нарушение кассовой дисциплины", name_en="Cash-desk discipline violation", rule_ref="", definition=""),
}
DEFAULT_DESC = dict(ru="Возможное нарушение кассовой дисциплины.", en="Possible violation of cash-desk discipline.")


def read_team_annotations(xlsx):
    """Файл команды (шаблон annotations_template.xlsx) -> (events_by_num, types)."""
    import pandas as pd
    sh = pd.read_excel(xlsx, sheet_name=None)
    low = {str(k).strip().lower(): k for k in sh}
    if "events" not in low:
        raise ValueError("в файле нет листа 'events'")
    ev = sh[low["events"]].copy()
    ev.columns = [str(c).strip().lower() for c in ev.columns]

    types = {}
    if "types" in low:
        ty = sh[low["types"]].copy()
        ty.columns = [str(c).strip().lower() for c in ty.columns]
        for _, r in ty.iterrows():
            k = r.get("type_key")
            if k is None or pd.isna(k) or not str(k).strip():
                continue
            k = _clean_key(k)
            g = lambda c: ("" if (c not in ty.columns or pd.isna(r.get(c))) else str(r.get(c)).strip())
            types[k] = dict(name_ru=g("name_ru") or k, name_en=g("name_en") or k, rule_ref=g("rule_ref"), definition=g("definition"))

    def num(x):
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else parse_time(x)

    by_num = {}
    for _, r in ev.iterrows():
        if pd.isna(r.get("video")):
            continue
        n = vid_num(r["video"])
        lab = "" if pd.isna(r.get("official_label")) else str(r.get("official_label")).lower()
        if "без" in lab:
            continue
        t = num(r.get("event_sec"))
        if t is None:
            t = num(r.get("official_time_sec"))
        if t is None:
            continue
        k = r.get("type_key")
        k = _clean_key(k) if (k is not None and not pd.isna(k) and str(k).strip()) else "other"
        ru = "" if pd.isna(r.get("desc_ru")) else str(r.get("desc_ru")).strip()
        en = "" if pd.isna(r.get("desc_en")) else str(r.get("desc_en")).strip()
        by_num.setdefault(n, []).append(dict(t=t, start=num(r.get("start_sec")), end=num(r.get("end_sec")), type=k,
                                             ru=ru or DEFAULT_DESC["ru"], en=en or DEFAULT_DESC["en"]))
        if k not in types:
            types[k] = dict(name_ru=k, name_en=k, rule_ref="", definition="")
    if by_num and "other" not in types:
        types["other"] = dict(name_ru="Прочее подозрительное действие", name_en="Other suspicious action", rule_ref="", definition="")
    return by_num, types


# -------------------------------------------------------------------- видео
class Video:
    def __init__(self, key, nums, path, duration, events):
        self.key, self.nums, self.path, self.duration, self.events = key, nums, str(path), float(duration), events
        self.files = []

    @property
    def violation(self):
        return bool(self.events)

    def __repr__(self):
        return f"Video({self.key}, {self.duration:.1f}s, events={[round(e['t'], 1) for e in self.events]})"


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def ffprobe_duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out.decode().strip())


def prepare_data(zip_path, data_dir):
    """Распаковать open.zip -> (папка, путь к official xlsx | None)."""
    data_dir = Path(data_dir)
    if not any(p.suffix.lower() in VIDEO_EXT for p in data_dir.rglob("*")):
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
    vids = [p for p in data_dir.rglob("*") if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("._")]
    xl = [p for p in data_dir.rglob("*.xlsx") if "label" in p.name.lower() and not p.name.startswith("~")]
    log(f"видео: {len(vids)}, файл разметки: {xl[0].name if xl else 'НЕ НАЙДЕН'}")
    return data_dir, (xl[0] if xl else None)


def merge_close_events(events, tol=1.5):
    out = []
    for e in sorted(events, key=lambda e: e["t"]):
        if out and e["t"] - out[-1]["t"] < tol:
            continue
        out.append(e)
    return out


def build_videos(data_dir, official, team_events=None, types=None):
    """Уникальные видео (дубликаты по md5 склеиваются, события объединяются) + словарь типов."""
    by_num = {}
    for p in sorted(Path(data_dir).rglob("*")):
        if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("._"):
            by_num[vid_num(p.name)] = p
    groups = {}
    for n, p in sorted(by_num.items()):
        groups.setdefault(md5_file(p), []).append(n)
    types = dict(types) if types else dict(DEFAULT_TYPES)
    default_type = "violation" if "violation" in types else ("other" if "other" in types else next(iter(types)))
    videos = []
    for nums in sorted(groups.values()):
        nums = sorted(nums)
        events, flags = [], []
        for n in nums:
            off = official.get(n)
            if off is None:
                log(f"  ! нет официальной метки для видео {n}")
                continue
            flags.append(off["violation"])
            if not off["violation"]:
                continue
            rows = (team_events or {}).get(n)
            if rows:
                events += [dict(e) for e in rows]
            elif off["t"] is not None:
                events.append(dict(t=off["t"], start=None, end=None, type=default_type, **DEFAULT_DESC))
            else:
                log(f"  ! видео {n}: нарушение без таймкода — пропускаю событие")
        if len(set(flags)) > 1:
            log(f"  ! дубликаты {nums} с разными метками {flags}")
        dur = ffprobe_duration(by_num[nums[0]])
        for e in events:
            e["t"] = min(max(float(e["t"]), 0.0), dur)
            e.setdefault("ru", DEFAULT_DESC["ru"]); e.setdefault("en", DEFAULT_DESC["en"])
        videos.append(Video("+".join(map(str, nums)), nums, by_num[nums[0]], dur, merge_close_events(events)))
    return videos, types


def describe_dataset(videos):
    pos = [v for v in videos if v.violation]
    log(f"уникальных видео: {len(videos)} | с нарушением: {len(pos)} | без: {len(videos) - len(pos)} | "
        f"событий: {sum(len(v.events) for v in videos)} | суммарно {sum(v.duration for v in videos):.0f} с")
    for v in videos:
        if len(v.nums) > 1:
            log(f"  дубликат: файлы {v.nums} — одно видео, события {[e['t'] for e in v.events]}")


def split_videos(videos, val_frac=0.25, seed=42, min_val_each=2):
    rnd = random.Random(seed)
    pos = [v for v in videos if v.violation]
    neg = [v for v in videos if not v.violation]
    rnd.shuffle(pos); rnd.shuffle(neg)
    kp = min(max(min_val_each, round(len(pos) * val_frac)), max(len(pos) - 1, 0))
    kn = min(max(min_val_each, round(len(neg) * val_frac)), max(len(neg) - 1, 0))
    return pos[kp:] + neg[kn:], pos[:kp] + neg[:kn]


# -------------------------------------------------------------------- кадры
def extract_frames(v, root, cfg, force=False):
    """ffmpeg: кадры с частотой cfg.fps, длинная сторона cfg.long_side -> v.files."""
    d = Path(root) / f"v{v.key.replace('+', '_')}_{cfg.fps:g}fps_{int(cfg.long_side)}"
    d.mkdir(parents=True, exist_ok=True)
    files = sorted(d.glob("*.jpg"))
    if force or not files:
        for f in files:
            f.unlink()
        L = int(cfg.long_side)
        vf = f"fps={cfg.fps},scale=w='if(gt(iw,ih),{L},-2)':h='if(gt(iw,ih),-2,{L})'"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v.path, "-vf", vf, "-q:v", "3", str(d / "%05d.jpg")], check=True)
        files = sorted(d.glob("*.jpg"))
    v.files = files
    return v


def n_win_frames(cfg):
    return int(round(cfg.win_sec * cfg.fps))


def window_starts(duration, cfg, stride):
    W = cfg.win_sec
    if duration <= W:
        return [0.0]
    s = [float(x) for x in np.arange(0.0, duration - W + 1e-6, stride)]
    if duration - W - s[-1] > 0.5:
        s.append(duration - W)
    return [math.floor(x * 1000) / 1000 for x in s]


def load_window_frames(v, t0, cfg):
    """-> (список PIL-кадров окна, фактическое начало окна в секундах)."""
    K, n = n_win_frames(cfg), len(v.files)
    i0 = max(0, min(int(round(t0 * cfg.fps)), n - K))
    idx = [min(max(i0 + k, 0), n - 1) for k in range(K)]
    return [Image.open(v.files[i]).convert("RGB") for i in idx], i0 / cfg.fps


def fit32(frames):
    """Размер кадров -> кратный 32 (патч 16 x слияние 2 у Qwen3-VL)."""
    w, h = frames[0].size
    W, H = max(64, int(round(w / 32)) * 32), max(64, int(round(h / 32)) * 32)
    return frames if (W, H) == (w, h) else [f.resize((W, H), Image.BICUBIC) for f in frames]


# ------------------------------------------------------ окна для обучения
def build_samples(videos, cfg):
    """-> (позитивы, негативы). Позитив: нарушение внутри окна не ближе margin к краям.
    Негатив: видео без нарушений целиком; в видео с нарушением — только окна ДО нарушения."""
    W = cfg.win_sec
    pos, neg = [], []
    for v in videos:
        last_t0 = max(0.0, v.duration - W)
        if not v.events:
            neg += [dict(vid=v.key, t0=t0, ev=None, kind="clean") for t0 in window_starts(v.duration, cfg, cfg.neg_stride)]
            continue
        for e in v.events:
            lo = max(0.0, e["t"] - W + cfg.margin)
            hi = min(e["t"] - cfg.margin, last_t0)
            if hi < lo:
                lo = hi = min(max(0.0, e["t"] - W / 2), last_t0)
            for t0 in np.arange(lo, hi + 1e-6, cfg.pos_stride):
                pos.append(dict(vid=v.key, t0=float(t0), ev=e, kind="pos"))
        limit = min(((e["start"] - 1.0) if e.get("start") is not None else (e["t"] - cfg.pre_gap)) for e in v.events)
        t0 = 0.0
        while t0 + W <= limit + 1e-6:
            neg.append(dict(vid=v.key, t0=t0, ev=None, kind="hard")); t0 += cfg.neg_stride
        if cfg.post_as_neg and all(e.get("end") is not None for e in v.events):
            t0 = max(e["end"] for e in v.events) + 1.0
            while t0 + W <= v.duration + 1e-6:
                neg.append(dict(vid=v.key, t0=t0, ev=None, kind="post")); t0 += cfg.neg_stride
    return pos, neg


def build_ramp_samples(videos, cfg):
    """Окна «нарастания». Последний кадр окна уже после начала действия (start_sec - 1 с), но раньше, чем нарушение
    стало очевидным (официальный таймкод + margin). Цель мягкая: p растёт линейно от 0 до 1. Так оценка 0-100
    растёт по мере развития действия (стоит 0% -> встаёт и идёт к двери ~70% -> ушла 100%), а не прыгает 0 -> 100."""
    W = cfg.win_sec
    out = []
    for v in videos:
        if not v.events:
            continue
        last_t0 = max(0.0, v.duration - W)
        spans = []
        for e in v.events:
            st = e["start"] if e.get("start") is not None else e["t"] - cfg.pre_gap + 1.0
            en = e["end"] if e.get("end") is not None else e["t"] + 2.0
            spans.append((min(st, e["t"]) - 1.0, max(en, e["t"])))
        for i, e in enumerate(v.events):
            lo_te, hi_te = spans[i][0], e["t"] + cfg.margin       # p=0 до lo_te; с hi_te окно считается позитивом
            span = hi_te - lo_te
            k = 1
            while lo_te + k * cfg.ramp_stride < hi_te - 1e-6:
                te = lo_te + k * cfg.ramp_stride
                t0 = te - W
                k += 1
                if t0 < 0.0 or t0 > last_t0 + 1e-6:
                    continue
                if any(j != i and t0 < spans[j][1] and t0 + W > spans[j][0] for j in range(len(v.events))):
                    continue                                       # окно задевает другое нарушение этого видео
                out.append(dict(vid=v.key, t0=float(t0), ev=e, kind="ramp", target=float((te - lo_te) / span)))
    return out


def epoch_order(pos, neg, ratio, rng, extra=()):
    k = min(len(neg), int(round(ratio * len(pos))))
    order = list(pos) + list(extra) + rng.sample(neg, k)
    rng.shuffle(order)
    return order


# -------------------------------------------------------------- аугментации
def _persp_coeffs(src, dst):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):          # выход (x,y) -> вход (u,v)
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def augment_clip(frames, rng):
    """Одни и те же случайные параметры для всех кадров окна: имитация другого ракурса."""
    w, h = frames[0].size
    out = frames
    if rng.random() < 0.5:                                         # перспектива
        j = 0.07
        dst = [(0, 0), (w, 0), (w, h), (0, h)]
        src = [(x + rng.uniform(-j, j) * w, y + rng.uniform(-j, j) * h) for x, y in dst]
        c = _persp_coeffs(src, dst)
        out = [f.transform((w, h), Image.PERSPECTIVE, c, Image.BICUBIC) for f in out]
    if rng.random() < 0.5:                                         # поворот
        a = rng.uniform(-10, 10)
        out = [f.rotate(a, resample=Image.BICUBIC) for f in out]
    if rng.random() < 0.7:                                         # кроп + масштаб
        s = rng.uniform(0.75, 1.0)
        cw, ch = int(w * s), int(h * s)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        out = [f.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC) for f in out]
    if rng.random() < 0.5:                                         # зеркало
        out = [ImageOps.mirror(f) for f in out]
    if rng.random() < 0.7:                                         # яркость/контраст
        b, c = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
        out = [ImageEnhance.Contrast(ImageEnhance.Brightness(f).enhance(b)).enhance(c) for f in out]
    return out


def strip_image(frames, n=6, height=360):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = []
    for i in idx:
        f = frames[i]
        tiles.append(f.resize((int(f.width * height / f.height), height)))
    sheet = Image.new("RGB", (sum(t.width for t in tiles), height))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0)); x += t.width
    return sheet


# ----------------------------------------------------------- промпт и ответ
SYSTEM_PROMPT = "You are a careful video analyst for bank cash-desk compliance. Describe only what is visible."
NEG_ANSWER = "NO\nTYPE: none\nTIME: -\nRU: Нарушений не видно.\nEN: No violation is visible."


def make_instruction(types):
    lines = "\n".join(f"- {k}: {t.get('name_en') or k}" for k, t in types.items())
    return ("The images are consecutive frames, with timestamps, from a fixed camera above a bank cashier's desk. "
            "Decide whether a violation of cashier discipline is visible in this clip. "
            "Report only what is visible; if you are not sure, answer NO.\n"
            "Reply in exactly this format:\n"
            "YES or NO\nTYPE: <type or none>\nTIME: <seconds from clip start, or ->\n"
            "RU: <what is visible and why it is a violation, in Russian>\nEN: <the same in English>\n"
            "Allowed types:\n" + lines)


def build_prompt(times, types):
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"]
    for t in times:
        parts.append(f"t={t:.1f}s <|vision_start|><|image_pad|><|vision_end|>\n")
    parts.append(make_instruction(types) + "<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


def build_answer(sample, t0_actual):
    if sample["ev"] is None:
        return NEG_ANSWER
    if sample.get("kind") == "ramp":                  # учим только вердикт (мягкая цель), без текста
        return "YES"
    e = sample["ev"]
    return f"YES\nTYPE: {e['type']}\nTIME: {e['t'] - t0_actual:.1f}\nRU: {e['ru']}\nEN: {e['en']}"


def parse_answer(text):
    t = text.strip()
    d = dict(verdict=None, type=None, time=None, ru="", en="")
    head = t.split("\n", 1)[0].strip().upper()
    if head.startswith("YES"):
        d["verdict"] = True
    elif head.startswith("NO"):
        d["verdict"] = False
    m = re.search(r"TYPE:\s*([A-Za-z0-9_\-]+)", t)
    if m:
        d["type"] = m.group(1).strip().lower()
    m = re.search(r"TIME:\s*(-?\d+(?:[.,]\d+)?)", t)
    if m:
        d["time"] = float(m.group(1).replace(",", "."))
    m = re.search(r"RU:\s*(.*?)(?=\n\s*EN:|\Z)", t, re.S)
    if m:
        d["ru"] = m.group(1).strip()
    m = re.search(r"EN:\s*(.*)\Z", t, re.S)
    if m:
        d["en"] = m.group(1).strip()
    return d


# ------------------------------------------------------------------- модель
def compute_dtype():
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    # bf16 нативно только с архитектуры Ampere (sm80+: A100/L4/A10/4090); на T4 is_bf16_supported() врёт (эмуляция)
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def auto_plan():
    """Подбор модели/разрешения под видеопамять."""
    import torch
    if not torch.cuda.is_available():
        log("GPU не найден: Runtime -> Change runtime type -> GPU")
        return {}
    gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    log(f"GPU: {torch.cuda.get_device_name(0)}, {gb:.0f} ГБ, dtype={compute_dtype()}")
    if gb < 20:
        plan = dict(model_id="Qwen/Qwen3-VL-4B-Instruct", load_4bit=True, long_side=576)
    elif gb < 32:
        plan = dict(model_id="Qwen/Qwen3-VL-8B-Instruct", load_4bit=True, long_side=640)
    else:
        plan = dict(model_id="Qwen/Qwen3-VL-8B-Instruct", load_4bit=False, long_side=704)
    log("рекомендуемые настройки:", plan)
    return plan


def load_base(model_id, load_4bit=True):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen3VLForConditionalGeneration as VLM
    except ImportError:
        from transformers import AutoModelForImageTextToText as VLM
    dtype = compute_dtype()
    kw = dict(dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"])
    model = VLM.from_pretrained(model_id, **kw)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def add_lora(model, cfg):
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def encode_example(processor, v, s, cfg, types, rng, augment=True):
    """Окно -> тензоры + число токенов ответа (n_ans)."""
    tok = processor.tokenizer
    frames, t0a = load_window_frames(v, s["t0"], cfg)
    if augment:
        frames = augment_clip(frames, rng)
    frames = fit32(frames)
    times = [i / cfg.fps for i in range(len(frames))]
    prompt = build_prompt(times, types)
    answer = build_answer(s, t0a)
    enc = processor(text=[prompt + answer + "<|im_end|>"], images=frames, return_tensors="pt")
    ans_ids = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = enc["input_ids"][0].tolist()
    n_ans = len(ans_ids)
    if ids[-n_ans:] != ans_ids:                       # граница токенизации не совпала — считаем длину промпта отдельно
        n_ans = len(ids) - len(processor(text=[prompt], images=frames, return_tensors="pt")["input_ids"][0])
    return enc, n_ans


def train_lora(model, processor, train_videos, cfg, types, log_every=20, ckpt_dir=None):
    import torch
    import torch.nn.functional as F
    dev = next(model.parameters()).device
    dtype = compute_dtype()
    pos, neg = build_samples(train_videos, cfg)
    ramp = build_ramp_samples(train_videos, cfg) if cfg.ramp else []
    log(f"окон: позитивных {len(pos)}, нарастания (ramp) {len(ramp)}, негативных {len(neg)} "
        f"(hard: {sum(1 for x in neg if x['kind'] == 'hard')})")
    if not pos:
        raise RuntimeError("нет позитивных окон — проверьте таймкоды нарушений")
    tok = processor.tokenizer
    yes_id = tok.encode("YES", add_special_tokens=False)[0]
    no_id = tok.encode("NO", add_special_tokens=False)[0]
    vids = {v.key: v for v in train_videos}
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    per_epoch = len(pos) + len(ramp) + min(len(neg), int(round(cfg.neg_ratio * len(pos))))
    total_upd = max(1, math.ceil(per_epoch * cfg.epochs / cfg.grad_accum))
    warm = max(1, int(0.06 * total_upd))

    def lr_at(u):
        if u < warm:
            return cfg.lr * (u + 1) / warm
        c = 0.5 * (1 + math.cos(math.pi * (u - warm) / max(1, total_upd - warm)))
        return cfg.lr * (0.1 + 0.9 * c)

    on_cuda = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(on_cuda and dtype == torch.float16))
    model.train()
    step = upd = bad = 0
    run, t_start = [], time.time()
    for ep in range(cfg.epochs):
        for s in epoch_order(pos, neg, cfg.neg_ratio, rng, extra=ramp):
            enc, A = encode_example(processor, vids[s["vid"]], s, cfg, types, rng, augment=True)
            batch = {k: (x.to(dev, dtype=dtype) if x.is_floating_point() else x.to(dev)) for k, x in enc.items()}
            with torch.autocast(dev.type, dtype=dtype, enabled=on_cuda):
                out = model(**batch, use_cache=False, logits_to_keep=A + 1)
            logits = out.logits[0, :-1].float()
            tgt = batch["input_ids"][0, -A:]
            if s.get("kind") == "ramp":                   # мягкая цель: P(YES) = target, только по паре YES/NO
                lp = F.log_softmax(logits[0, [yes_id, no_id]], dim=-1)
                g = float(s["target"])
                loss = cfg.verdict_w * -(g * lp[0] + (1.0 - g) * lp[1])
            else:
                ce = F.cross_entropy(logits, tgt, reduction="none")
                loss = cfg.verdict_w * ce[0] + ce[1:].mean()
            step += 1
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad in (1, 5, 20):
                    log(f"  ! loss={loss.item()} на шаге {step} (пропускаю). Если часто — используйте GPU с bf16 (L4/A100).")
                continue
            scaler.scale(loss / cfg.grad_accum).backward()
            run.append(loss.item())
            if step % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(upd)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                upd += 1
            if step % log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if on_cuda else 0.0
                log(f"эпоха {ep + 1}/{cfg.epochs} шаг {step} loss {np.mean(run[-log_every:]):.3f} "
                    f"lr {lr_at(max(upd - 1, 0)):.2e} токенов {batch['input_ids'].shape[1]} VRAM {mem:.1f} ГБ "
                    f"{(time.time() - t_start) / 60:.1f} мин")
        if ckpt_dir:                                   # чекпойнт после каждой эпохи (на Drive переживёт обрыв сессии)
            try:
                model.save_pretrained(str(ckpt_dir))
                log(f"  чекпойнт адаптера после эпохи {ep + 1} -> {ckpt_dir}")
            except Exception as e:
                log(f"  ! не удалось сохранить чекпойнт: {e}")
    model.eval()
    return model


# ------------------------------------------------------------------- анализ
def grab_frame_b64(video_path, t, duration=None):
    if duration:
        t = min(t, max(duration - 0.15, 0))
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.2f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "3",
           "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    return base64.b64encode(data).decode() if data else ""


def sigmoid(z):
    """Устойчивая сигмоида: без OverflowError при очень больших по модулю z."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z) if z > -700 else 0.0
    return e / (1.0 + e)


def find_events(scores, cfg, tau, min_pos):
    """scores [(t0, p)] -> кластеры соседних окон с p >= tau (не менее min_pos окон)."""
    W = cfg.win_sec
    clusters = []
    for t0, p in sorted(x for x in scores if x[1] >= tau):
        c = t0 + W / 2
        if clusters and c - clusters[-1]["last_c"] <= cfg.cluster_gap:
            clusters[-1]["w"].append((t0, p)); clusters[-1]["last_c"] = c
        else:
            clusters.append(dict(w=[(t0, p)], last_c=c))
    clusters = [c for c in clusters if len(c["w"]) >= min_pos]
    for c in clusters:
        c["best"] = max(c["w"], key=lambda x: x[1])
        c["p"] = c["best"][1]
    return clusters


# Чек-лист: узкие вопросы «да/нет» к базовой модели на каждое окно (без обучения самой модели).
# Каждый ответ -> одно число (логит ДА минус логит НЕТ). Решение потом принимает маленькая модель на ~20 числах.
CHECK_SYSTEM = ("You are a careful video analyst. You look at frames from a fixed camera in a bank cash office and answer "
                "questions about what is visible. Answer only what you can see.")
CHECKLIST = [
    ("seated",       "Is a person sitting at the desk, working at the computer or handling money?"),
    ("desk_empty",   "Is the desk area completely empty, with nobody sitting or standing at it?"),
    ("standing",     "Is the employee standing up from the chair or walking away from the desk?"),
    ("leaving",      "Is a person walking out of the room through the door?"),
    ("visitor",      "Is a visitor or client sitting or standing at the desk opposite the employee?"),
    ("two_visitors", "Are two or more visitors at the desk at the same time, besides the employee?"),
    ("entering",     "Is someone entering the room or standing in the doorway?"),
    ("crowd",        "Are three or more people visible in the room?"),
    ("cash_desk",    "Are banknotes clearly visible lying on the desk?"),
    ("handover",     "Is money or a document being handed from one person to another?"),
    ("cash_mixed",   "Are banknotes of two different people lying together on the desk?"),
    ("safe_open",    "Is the door of the safe or metal cabinet open?"),
    ("key",          "Is a key lying on the desk or left in the lock of the safe?"),
    ("belongings",   "Are personal belongings such as clothes, a coat, a bag or a cup lying on the desk?"),
    ("door_open",    "Is the door of the room standing open?"),
    ("id_shown",     "Is a visitor showing an identity card or document to the employee?"),
    ("unattended",   "Is money or an open safe left in the room with nobody watching it?"),
    ("violation",    "Does anything in this clip look like a violation of cashier safety rules?"),
    # --- вторая очередь вопросов (уточняющие)
    ("alone",        "Is the employee the only person in the room?"),
    ("behind_desk",  "Is someone other than the employee standing or sitting on the employee's side of the desk, near the safe or the computer?"),
    ("peeking",      "Is someone looking into the work area through the glass wall or the doorway?"),
    ("cash_hand",    "Is the employee holding banknotes in her hands?"),
    ("cash_count",   "Is money being counted or put into a money counting machine?"),
    ("safe_closed",  "Is the door of the safe or metal cabinet closed?"),
    ("items_desk",   "Are there any objects on the desk other than the laptop, the telephone and the printer?"),
    ("cup",          "Is there a cup or mug on the desk, or is someone drinking?"),
    ("phone_use",    "Is the employee using a mobile phone?"),
    ("laptop_on",    "Is the laptop screen turned on?"),
    ("laptop_closed","Is the laptop closed?"),
    ("door_closed",  "Is the glass door of the room closed?"),
    ("id_visible",   "Is an identity card or passport visible in someone's hand or lying on the desk?"),
    ("normal",       "Does everything in this clip look like normal, correct cashier work?"),
]
CHECK_KEYS = [k for k, _ in CHECKLIST]

# Второй вид: крупный план стола и сейфа (кадры вырезаются из исходного видео по рамке, найденной самой моделью).
CHECK_SYSTEM_DESK = ("You are a careful video analyst. The frames are a close-up of the desk and the safe in a bank cash office, "
                     "taken one second apart by a fixed camera. Answer only what you can see.")
DESK_CHECKLIST = [
    ("d_person",  "Is a person sitting or standing at the desk?"),
    ("d_items",   "Are there any objects on the desk other than the laptop, the telephone and the printer?"),
    ("d_clear",   "Is the desk tidy, with nothing on it except the laptop, the telephone and the printer?"),
    ("d_cash",    "Are there banknotes or money lying on the desk?"),
    ("d_key",     "Is there a key lying on the desk or hanging in the lock of the safe?"),
    ("d_cup",     "Is there a cup, a mug or a bottle on the desk?"),
    ("d_tablet",  "Is there a tablet or a second mobile device lying on the desk?"),
    ("d_paper",   "Are there papers, cards or documents lying on the desk?"),
    ("d_safe_open", "Is the door of the safe open?"),
    ("d_laptop_open", "Is the laptop open?"),
    ("d_lock",    "Does the laptop screen show a blue lock screen or a login screen?"),
    ("d_hand",    "Is someone's hand touching or picking up something on the desk?"),
]
DESK_KEYS = [k for k, _ in DESK_CHECKLIST]


def check_prefix(times, system=None):
    parts = [f"<|im_start|>system\n{system or CHECK_SYSTEM}<|im_end|>\n<|im_start|>user\n"]
    for t in times:
        parts.append(f"t={t:.1f}s <|vision_start|><|image_pad|><|vision_end|>\n")
    return "".join(parts)


def check_suffix(question):
    return f"{question} Answer with YES or NO only.<|im_end|>\n<|im_start|>assistant\n"


# Признаки для «пробы» (линейного классификатора поверх замороженной модели)
FEAT_LAYER_FRACS = (0.45, 0.6, 0.75, 1.0)           # скрытое состояние последнего токена на этих долях глубины
IMG_LAYER_FRACS = (0.5, 0.7)                        # среднее по токенам кадров (3 части окна по времени) на этих слоях
QUAD_LAYER_FRAC = 0.6                               # средние по четвертям кадра (усреднено по времени) на этом слое
N_FEAT_VECS = len(FEAT_LAYER_FRACS) + 3 * len(IMG_LAYER_FRACS) + 4


class Analyzer:
    def __init__(self, model, processor, cfg, types, tau=0.5, min_pos=1):
        import torch
        self.model, self.processor, self.cfg, self.types = model, processor, cfg, types
        self.tau, self.min_pos = tau, min_pos
        self.tok = processor.tokenizer
        self.dev = next(model.parameters()).device
        self.dtype = compute_dtype() if self.dev.type == "cuda" else torch.float32
        self.yes_id = self.tok.encode("YES", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("NO", add_special_tokens=False)[0]
        assert self.yes_id != self.no_id, "токены YES/NO совпали"
        self.img_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.times = [i / cfg.fps for i in range(n_win_frames(cfg))]
        first = lambda w: self.tok.encode(w, add_special_tokens=False)[0]
        self.yes_ids = sorted({first(w) for w in ("YES", "Yes", "yes")})
        self.no_ids = sorted({first(w) for w in ("NO", "No", "no")})

    def _inputs(self, text, frames):
        enc = self.processor(text=[text], images=frames, return_tensors="pt")
        return {k: (v.to(self.dev, dtype=self.dtype) if v.is_floating_point() else v.to(self.dev)) for k, v in enc.items()}

    def _ctx(self):
        import torch
        return torch.autocast(self.dev.type, dtype=self.dtype, enabled=(self.dev.type == "cuda"))

    def window_score(self, frames):
        """P(нарушение) по логитам первого токена ответа: YES против NO."""
        import torch
        with torch.no_grad(), self._ctx():
            inp = self._inputs(build_prompt(self.times, self.types), frames)
            lg = self.model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        z = (lg[self.yes_id] - lg[self.no_id]).item()
        if not math.isfinite(z):
            self.nan_count = getattr(self, "nan_count", 0) + 1
            if self.nan_count in (1, 10):
                log("  ! NaN в логитах (fp16 на T4?) — окно считаю как «нет нарушения». Лучше GPU с bf16 (L4/A100).")
            return 0.0
        return sigmoid(z)

    def window_features(self, frames):
        """Один прямой проход -> (z, признаки float16 [n_vec, D]).
        Векторы: скрытое состояние последнего токена на слоях FEAT_LAYER_FRACS; среднее по токенам кадров (три части окна
        по времени) на слоях IMG_LAYER_FRACS; четыре четверти кадра на слое QUAD_LAYER_FRAC. z = логит YES - логит NO."""
        import torch
        with torch.no_grad(), self._ctx():
            inp = self._inputs(build_prompt(self.times, self.types), frames)
            out = self.model(**inp, use_cache=False, logits_to_keep=1, output_hidden_states=True)
        hs = out.hidden_states
        n_layers = len(hs) - 1
        lg = out.logits[0, -1].float()
        z = (lg[self.yes_id] - lg[self.no_id]).item()
        vec = [hs[max(1, int(round(f * n_layers)))][0, -1].float() for f in FEAT_LAYER_FRACS]
        pos = (inp["input_ids"][0] == self.img_id).nonzero().squeeze(1)
        parts = torch.tensor_split(pos, 3)
        for f in IMG_LAYER_FRACS:
            h = hs[max(1, int(round(f * n_layers)))][0]
            for part in parts:
                vec.append(h[part].float().mean(0))
        h = hs[max(1, int(round(QUAD_LAYER_FRAC * n_layers)))][0][pos].float()
        quads, off = torch.zeros(4, h.shape[-1], device=h.device), 0
        grid = inp["image_grid_thw"].tolist()
        for t, gh, gw in grid:                           # токены кадра идут по строкам: (gh/2) x (gw/2) после слияния 2x2
            gh2, gw2 = gh // 2, gw // 2
            n = t * gh2 * gw2
            c = h[off:off + n].reshape(gh2, gw2, -1)
            off += n
            hh, ww = max(gh2 // 2, 1), max(gw2 // 2, 1)
            quads[0] += c[:hh, :ww].mean((0, 1)); quads[1] += c[:hh, ww:].mean((0, 1))
            quads[2] += c[hh:, :ww].mean((0, 1)); quads[3] += c[hh:, ww:].mean((0, 1))
        vec += list(quads / max(len(grid), 1))
        feats = torch.stack(vec).clamp(-6e4, 6e4).cpu().numpy().astype(np.float16)
        return (z if math.isfinite(z) else 0.0), feats

    def check_selftest(self, frames, tol=2.0, batch=4):
        """Сверка быстрых путей с полным на одном окне (на реальной модели): с кэшем и с кэшем + пакетом вопросов."""
        t = time.time(); full = self.window_checklist(frames, cached=False); t_full = time.time() - t
        t = time.time(); fast = self.window_checklist(frames, cached=True); t_fast = time.time() - t
        t = time.time(); bat = self.window_checklist(frames, cached=True, batch=batch); t_bat = time.time() - t
        diff = float(np.abs(full - fast).max()); diff_b = float(np.abs(full - bat).max())
        log(f"чек-лист: без кэша {t_full:.1f} с, с кэшем {t_fast:.1f} с, с кэшем и пакетом по {batch} {t_bat:.1f} с на окно; "
            f"макс. расхождение логитов: кэш {diff:.2f}, пакет {diff_b:.2f}"
            + ("" if max(diff, diff_b) <= tol else "  ! расхождение больше обычного — покажите этот вывод"))
        return max(diff, diff_b)

    def window_checklist(self, frames, questions=None, cached=True, batch=1, times=None, system=None):
        """Ответы на вопросы чек-листа для одного окна -> np.array [n_questions] (логит ДА минус логит НЕТ).
        Кадры кодируются один раз, дальше к кэшу дописывается только короткий вопрос. cached=False — проверочный путь без кэша.
        batch > 1: несколько вопросов за один проход (кэш размножается на batch строк; быстрее при том же результате)."""
        import torch
        qs = list(CHECKLIST if questions is None else questions)
        prefix = check_prefix(self.times if times is None else times, system)
        out = np.zeros(len(qs), dtype=np.float32)

        def zval(lg):
            lg = lg.float()
            return (torch.logsumexp(lg[self.yes_ids], 0) - torch.logsumexp(lg[self.no_ids], 0)).item()

        with torch.no_grad(), self._ctx():
            if not cached:
                for i, (_, q) in enumerate(qs):
                    inp = self._inputs(prefix + check_suffix(q), frames)
                    out[i] = zval(self.model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1])
            else:
                inp = self._inputs(prefix, frames)
                o = self.model(**inp, use_cache=True, logits_to_keep=1)
                cache, L = o.past_key_values, inp["input_ids"].shape[1]
                B = max(1, min(int(batch), len(qs)))
                if B == 1:
                    for i, (_, q) in enumerate(qs):
                        ids = self.tok(check_suffix(q), add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.dev)
                        cp = torch.arange(L, L + ids.shape[1], device=self.dev)
                        lg = self.model(input_ids=ids, past_key_values=cache, cache_position=cp, use_cache=True,
                                        logits_to_keep=1).logits[0, -1]
                        out[i] = zval(lg)
                        cache.crop(L)
                else:
                    cache.batch_repeat_interleave(B)
                    pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                    for c0 in range(0, len(qs), B):
                        chunk = qs[c0:c0 + B]
                        n = len(chunk)
                        chunk = chunk + [chunk[-1]] * (B - n)              # последняя группа добивается до полного размера
                        toks = [self.tok(check_suffix(q), add_special_tokens=False)["input_ids"] for _, q in chunk]
                        S = max(len(t) for t in toks)
                        ids = torch.full((B, S), pad_id, dtype=torch.long)
                        mask = torch.zeros((B, L + S), dtype=torch.long)
                        mask[:, :L] = 1
                        for r, t in enumerate(toks):
                            ids[r, :len(t)] = torch.tensor(t)
                            mask[r, L:L + len(t)] = 1
                        cp = torch.arange(L, L + S, device=self.dev)
                        lg = self.model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev), past_key_values=cache,
                                        cache_position=cp, use_cache=True).logits
                        for r in range(n):
                            out[c0 + r] = zval(lg[r, len(toks[r]) - 1])
                        cache.crop(L)
        return np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0)

    def caption(self, frames, max_new_tokens=80, lang="en"):
        """Короткое описание окна обычным текстом (что делают сотрудница и посетители, что лежит на столе)."""
        import torch
        if lang == "ru":
            q = ("Сотрудница — человек, который работает за столом у ноутбука и сейфа; все остальные люди — посетители. "
                 "Опиши по-русски одним–двумя короткими предложениями, что делают сотрудница и посетители и что лежит на столе. "
                 "Описывай только то, что видно, без предположений.")
        else:
            q = ("Describe in two short sentences what the employee and any visitors are doing and what lies on the desk. "
                 "Mention the door, the safe and personal belongings if they matter.")
        with torch.no_grad(), self._ctx():
            inp = self._inputs(check_prefix(self.times) + f"{q}<|im_end|>\n<|im_start|>assistant\n", frames)
            gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        return self.tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def describe(self, frames):
        import torch
        with torch.no_grad(), self._ctx():
            inp = self._inputs(build_prompt(self.times, self.types) + "YES\n", frames)
            gen = self.model.generate(**inp, max_new_tokens=220, do_sample=False, use_cache=True)
        txt = self.tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_answer("YES\n" + txt)

    def score_video(self, v, verbose=False):
        """Оценка всех окон видео -> [(t0, p)] (окна идут «через одно», чтобы при нехватке времени потерять равномерно)."""
        cfg = self.cfg
        stride = max(cfg.infer_stride, (v.duration - cfg.win_sec) / max(cfg.max_windows - 1, 1))
        starts = window_starts(v.duration, cfg, stride)
        order = list(range(0, len(starts), 2)) + list(range(1, len(starts), 2))
        out, t_start = [], time.time()
        for i in order:
            if time.time() - t_start > cfg.time_budget_sec:
                log(f"  ! лимит времени: оценено {len(out)} из {len(starts)} окон")
                break
            frames, t0a = load_window_frames(v, starts[i], cfg)
            out.append((t0a, self.window_score(fit32(frames))))
        out.sort()
        if verbose:
            log(f"{v.key}: окон {len(out)}, max p = {max(p for _, p in out):.2f}, {time.time() - t_start:.0f} с")
        return out

    def finalize(self, v, scores, tau=None, min_pos=None, describe=True):
        """Окна -> результат в формате API (verdict, violations[...])."""
        cfg = self.cfg
        tau = self.tau if tau is None else tau
        min_pos = self.min_pos if min_pos is None else min_pos
        cl = sorted(find_events(scores, cfg, tau, min_pos), key=lambda c: -c["p"])[:cfg.max_events]
        events = []
        for c in cl:
            t0 = c["best"][0]
            d = {}
            if describe:
                frames, t0 = load_window_frames(v, t0, cfg)
                d = self.describe(fit32(frames))
            rel = d.get("time")
            te = t0 + (rel if (rel is not None and 0 <= rel <= cfg.win_sec) else cfg.win_sec / 2)
            te = min(max(te, 0.0), v.duration)
            key = d.get("type")
            if key not in self.types:
                key = "other" if "other" in self.types else (next(iter(self.types)) if len(self.types) == 1 else "other")
            info = self.types.get(key, dict(name_ru="Прочее подозрительное действие", rule_ref=""))
            ts = sorted({round(min(max(te + dt, 0), v.duration), 2) for dt in (-1.0, 0.0, 1.0)})
            events.append(dict(
                type_key=key, type=info.get("name_ru") or key, timecode_sec=round(te, 1), score=round(100 * c["p"]),
                description=(d.get("ru") or DEFAULT_DESC["ru"]), description_en=(d.get("en") or DEFAULT_DESC["en"]),
                rule_ref=info.get("rule_ref", ""),
                evidence_frames=[dict(timestamp_sec=t, image_base64=grab_frame_b64(v.path, t, v.duration)) for t in ts]))
        merged = []
        for e in sorted(events, key=lambda e: e["timecode_sec"]):
            if merged and e["timecode_sec"] - merged[-1]["timecode_sec"] < 3.0:
                if e["score"] > merged[-1]["score"]:
                    merged[-1] = e
                continue
            merged.append(e)
        return dict(verdict="violation" if merged else "no_violation", violations=merged)

    def analyze(self, v, describe=True):
        return self.finalize(v, self.score_video(v), describe=describe)


def to_api(result, with_rule=True, lang="ru"):
    """Результат -> JSON строго по схеме кейса (+ необязательные поля score/rule)."""
    out = dict(verdict=result["verdict"], violations=[])
    for e in result["violations"]:
        desc = e["description"] if lang == "ru" else e.get("description_en") or e["description"]
        if with_rule and e.get("rule_ref"):
            desc += f" Правило: {e['rule_ref']}"
        out["violations"].append(dict(type=e["type"], timecode_sec=e["timecode_sec"], description=desc,
                                      evidence_frames=e["evidence_frames"]))
    return out


# ------------------------------------------------------------------- метрики
def video_metrics(videos, all_scores, cfg, tau, min_pos):
    tp = fp = fn = tn = 0
    for v in videos:
        pred = bool(find_events(all_scores[v.key], cfg, tau, min_pos))
        if v.violation and pred: tp += 1
        elif v.violation: fn += 1
        elif pred: fp += 1
        else: tn += 1
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, specificity=spec, precision=prec, f2=f2)


def tune_threshold(videos, all_scores, cfg, taus=None, min_pos_grid=(1, 2, 3), verbose=True):
    """Подбор tau и min_pos: максимум F2 (пропуск хуже ложной тревоги), при равенстве — меньше FP."""
    taus = taus if taus is not None else [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    best, rows = None, []
    for mp in min_pos_grid:
        for tau in taus:
            m = video_metrics(videos, all_scores, cfg, tau, mp)
            rows.append((tau, mp, m))
            key = (m["f2"], m["specificity"], tau)
            if best is None or key > best[0]:
                best = (key, tau, mp, m)
    if verbose:
        log("tau  min_pos  TP FP FN TN  recall  spec   F2")
        for tau, mp, m in rows:
            if m["f2"] >= best[3]["f2"] - 0.1:
                log(f"{tau:.2f}   {mp}      {m['tp']:>2} {m['fp']:>2} {m['fn']:>2} {m['tn']:>2}  {m['recall']:.2f}   {m['specificity']:.2f}  {m['f2']:.2f}")
        log(f"-> выбрано tau={best[1]}, min_pos={best[2]}: {best[3]}")
    return best[1], best[2], best[3]


def eval_localization(videos, results, tols=(2.0, 3.0, 5.0)):
    """results[key] = результат analyze(). Для каждого GT-события ближайшее найденное."""
    errs, ok_t, n_t, missed = [], 0, 0, 0
    for v in videos:
        for e in v.events:
            pred = results.get(v.key, {}).get("violations", [])
            if not pred:
                missed += 1
                continue
            b = min(pred, key=lambda p: abs(p["timecode_sec"] - e["t"]))
            err = abs(b["timecode_sec"] - e["t"])
            errs.append(err)
            if err <= max(tols):
                n_t += 1
                ok_t += int(b["type_key"] == e["type"])
    if errs:
        log(f"локализация: событий найдено {len(errs)}, пропущено {missed}; медиана ошибки {np.median(errs):.1f} с, средняя {np.mean(errs):.1f} с")
        log("  доля в допуске: " + ", ".join(f"±{t:g} с: {np.mean([x <= t for x in errs]):.0%}" for t in tols))
        log(f"  тип совпал: {ok_t}/{n_t}")
    else:
        log("локализация: нет найденных событий")
    return errs


def score_videos(analyzer, videos):
    out = {}
    for i, v in enumerate(videos, 1):
        out[v.key] = analyzer.score_video(v, verbose=True)
    return out


def extract_features(an, videos, cfg, out_path, stride=1.0, aug_stride=2.0, seed=0, log_every=25):
    """Признаки окон всех видео -> npz (обучение пробы потом идёт без GPU).
    Исходные окна с шагом stride + аугментированные (другой ракурс) с шагом aug_stride."""
    rng = random.Random(seed)
    plan = []
    for v in videos:
        for aug, st in ((0, stride), (1, aug_stride)):
            if st and st > 0:
                plan += [(v, aug, t0) for t0 in window_starts(v.duration, cfg, st)]
    log(f"окон к обработке: {len(plan)} (из них аугментированных {sum(1 for p in plan if p[1])})")
    keys, t0s, augs, zs, feats = [], [], [], [], []
    t_start = time.time()
    for i, (v, aug, t0) in enumerate(plan, 1):
        frames, t0a = load_window_frames(v, t0, cfg)
        if aug:
            frames = augment_clip(frames, rng)
        z, f = an.window_features(fit32(frames))
        keys.append(v.key); t0s.append(t0a); augs.append(aug); zs.append(z); feats.append(f)
        if i % log_every == 0 or i == len(plan):
            el = time.time() - t_start
            log(f"  {i}/{len(plan)} окон, {el / 60:.1f} мин, осталось ~{el / i * (len(plan) - i) / 60:.1f} мин")
    meta = {v.key: dict(duration=v.duration, events=[dict(t=e["t"], start=e.get("start"), end=e.get("end"), type=e.get("type"))
                                                     for e in v.events]) for v in videos}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, key=np.array(keys), t0=np.array(t0s, dtype=np.float32), aug=np.array(augs, dtype=np.int8),
                        z=np.array(zs, dtype=np.float32), F=np.stack(feats),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)), cfg=np.array(json.dumps(vars(cfg), ensure_ascii=False)))
    log(f"сохранено: {out_path} ({Path(out_path).stat().st_size / 2 ** 20:.0f} МБ), окон {len(keys)}, признаков на окно {feats[0].shape}")
    return out_path


def extract_checklist(an, videos, cfg, out_path, stride=2.0, aug_stride=4.0, seed=0, log_every=20):
    """Ответы чек-листа по окнам всех видео -> npz (дальше обучение маленькой модели идёт без GPU)."""
    rng = random.Random(seed)
    plan = []
    for v in videos:
        for aug, st in ((0, stride), (1, aug_stride)):
            if st and st > 0:
                plan += [(v, aug, t0) for t0 in window_starts(v.duration, cfg, st)]
    log(f"окон к обработке: {len(plan)} (из них аугментированных {sum(1 for p in plan if p[1])}), вопросов {len(CHECKLIST)}")
    keys, t0s, augs, Q = [], [], [], []
    t_start = time.time()
    for i, (v, aug, t0) in enumerate(plan, 1):
        frames, t0a = load_window_frames(v, t0, cfg)
        if aug:
            frames = augment_clip(frames, rng)
        Q.append(an.window_checklist(fit32(frames)))
        keys.append(v.key); t0s.append(t0a); augs.append(aug)
        if i % log_every == 0 or i == len(plan):
            el = time.time() - t_start
            log(f"  {i}/{len(plan)} окон, {el / 60:.1f} мин, осталось ~{el / i * (len(plan) - i) / 60:.1f} мин")
    meta = {v.key: dict(duration=v.duration, events=[dict(t=e["t"], start=e.get("start"), end=e.get("end"), type=e.get("type"))
                                                     for e in v.events]) for v in videos}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, key=np.array(keys), t0=np.array(t0s, dtype=np.float32), aug=np.array(augs, dtype=np.int8),
                        Q=np.stack(Q).astype(np.float32), qkeys=np.array(CHECK_KEYS),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)), cfg=np.array(json.dumps(vars(cfg), ensure_ascii=False)))
    log(f"сохранено: {out_path} ({Path(out_path).stat().st_size / 2 ** 20:.1f} МБ), окон {len(keys)}")
    return out_path


def extract_captions(an, videos, cfg, out_path, stride=10.0, log_every=10):
    """Текстовые описания окон (для просмотра глазами и для текста в отчёте) -> json {видео: [{t0, text}]}."""
    out, plan = {}, []
    for v in videos:
        plan += [(v, t0) for t0 in window_starts(v.duration, cfg, stride)]
    log(f"описаний к обработке: {len(plan)}")
    t_start = time.time()
    for i, (v, t0) in enumerate(plan, 1):
        frames, t0a = load_window_frames(v, t0, cfg)
        out.setdefault(v.key, []).append(dict(t0=round(t0a, 1), text=an.caption(fit32(frames))))
        if i % log_every == 0 or i == len(plan):
            el = time.time() - t_start
            log(f"  {i}/{len(plan)}, {el / 60:.1f} мин, осталось ~{el / i * (len(plan) - i) / 60:.1f} мин")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"сохранено: {out_path}")
    return out


# ------------------------------------------------------------ вид «стол крупно»
GROUND_PROMPT = ("Locate the office desk and the safe (a grey metal cabinet) standing next to it in this image. "
                 "Output the bounding box of each in JSON format.")


def parse_boxes(text, w, h):
    """Ответ модели с рамками -> [(x1, y1, x2, y2)] в долях кадра. Qwen3-VL даёт относительные координаты 0..1000;
    доли (0..1) и пиксели (больше 1000) тоже понимаем."""
    num = r"(-?\d+(?:\.\d+)?)"
    out = []
    for m in re.finditer(r"\[\s*" + r"\s*,\s*".join([num] * 4) + r"\s*\]", text):
        x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
        mx = max(x1, y1, x2, y2)
        if mx > 1000.0:
            x1, x2, y1, y2 = x1 / w, x2 / w, y1 / h, y2 / h
        elif mx > 1.5:
            x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
        if x2 - x1 > 0.02 and y2 - y1 > 0.02:
            out.append((x1, y1, x2, y2))
    return out


def pad_box(box, pad=0.12):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (max(0.0, x1 - pad * w), max(0.0, y1 - pad * h), min(1.0, x2 + pad * w), min(1.0, y2 + pad * h))


def _ground(self, frame, max_new_tokens=160):
    """Одна рамка (стол + сейф) по одному кадру: модель сама показывает, где они. -> (текст ответа, [рамки в долях])."""
    import torch
    text = ("<|im_start|>system\nYou are a careful visual assistant.<|im_end|>\n<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>{GROUND_PROMPT}<|im_end|>\n<|im_start|>assistant\n")
    with torch.no_grad(), self._ctx():
        inp = self._inputs(text, [frame])
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    out = self.tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return out, parse_boxes(out, *frame.size)


def _find_desk_box(self, frames, verbose=False):
    """Рамка стола и сейфа по нескольким кадрам видео (медиана по кадрам) -> (x1, y1, x2, y2) в долях или None."""
    got = []
    for fr in frames:
        txt, boxes = self.ground(fr)
        boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < 0.9]
        if verbose:
            log("   рамки:", [tuple(round(x, 2) for x in b) for b in boxes], "|", txt[:90].replace("\n", " "))
        if boxes:
            got.append((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)))
    if not got:
        return None
    box = tuple(float(np.median([g[i] for g in got])) for i in range(4))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return pad_box(box) if 0.04 <= area <= 0.85 else None


Analyzer.ground = _ground
Analyzer.find_desk_box = _find_desk_box


def cut_desk_frames(video_path, box, out_dir, fps=1.0):
    """ffmpeg: вырезать рамку из исходного видео (без потери разрешения) и взять кадры fps в секунду -> список файлов."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.jpg"):
        f.unlink()
    x1, y1, x2, y2 = box if box else (0.0, 0.0, 1.0, 1.0)
    vf = f"fps={fps},crop=iw*{x2 - x1:.4f}:ih*{y2 - y1:.4f}:iw*{x1:.4f}:ih*{y1:.4f}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vf", vf, "-q:v", "2", str(d / "%05d.jpg")], check=True)
    return sorted(d.glob("*.jpg"))


def desk_window_frames(files, t0, nview=8, fps=1.0, target_px=448, max_up=2.5, size=None):
    """Кадры вида «стол крупно» для окна, начинающегося в t0: nview кадров подряд, размер кратен 32. -> (кадры, размер)."""
    n = len(files)
    i0 = int(round(t0 * fps))
    idx = [min(max(i0 + k, 0), n - 1) for k in range(nview)]
    imgs = [Image.open(files[i]).convert("RGB") for i in idx]
    if size is None:
        w, h = imgs[0].size
        sc = min(max_up, math.sqrt(target_px * target_px / (w * h)))
        size = (max(64, int(round(w * sc / 32)) * 32), max(64, int(round(h * sc / 32)) * 32))
    return [im.resize(size, Image.BICUBIC) for im in imgs], size


def extract_desk(an, videos, cfg, out_path, work_dir, stride=2.0, nview=8, log_every=40, montage_path=None):
    """Второй вид (крупный план стола и сейфа) для тех же окон, что в extract_checklist (только исходные, без аугментации).
    -> npz: key, t0, aug, Q [n, len(DESK_KEYS)], qkeys, meta (рамки), cfg."""
    t_start, boxes, plan = time.time(), {}, []
    for v in videos:
        pick = [v.files[min(int(round(f * v.duration * cfg.fps)), len(v.files) - 1)] for f in (0.15, 0.5, 0.85)]
        box = an.find_desk_box([fit32([Image.open(p).convert("RGB")])[0] for p in pick])
        boxes[v.key] = box
        log(f"  {v.key}: рамка {tuple(round(x, 2) for x in box) if box else 'не найдена, беру весь кадр'}")
    log(f"рамки найдены за {time.time() - t_start:.0f} с")
    tiles = []
    times = [float(k) for k in range(nview)]
    K, rows, keys, t0s = n_win_frames(cfg), [], [], []
    for v in videos:
        files = cut_desk_frames(v.path, boxes[v.key], Path(work_dir) / f"desk_{v.key.replace('+', '_')}")
        size = None
        if montage_path and files:
            tiles.append(Image.open(files[min(len(files) - 1, int(v.duration / 2))]).convert("RGB"))
        n = len(v.files)
        for t0 in window_starts(v.duration, cfg, stride):
            i0 = max(0, min(int(round(t0 * cfg.fps)), n - K))
            frames, size = desk_window_frames(files, i0 / cfg.fps, nview, size=size)
            rows.append(an.window_checklist(frames, questions=DESK_CHECKLIST, batch=4, times=times, system=CHECK_SYSTEM_DESK))
            keys.append(v.key); t0s.append(i0 / cfg.fps)
            if len(rows) % log_every == 0:
                el = time.time() - t_start
                log(f"  {len(rows)} окон, {el / 60:.1f} мин")
    if montage_path and tiles:
        h = 200
        tiles = [t.resize((max(1, int(t.width * h / t.height)), h)) for t in tiles]
        per = 8
        lines = [tiles[i:i + per] for i in range(0, len(tiles), per)]
        sheet = Image.new("RGB", (max(sum(t.width for t in l) for l in lines), h * len(lines)))
        for r, l in enumerate(lines):
            x = 0
            for t in l:
                sheet.paste(t, (x, r * h)); x += t.width
        sheet.save(montage_path, quality=85)
    meta = {v.key: dict(box=boxes[v.key], duration=v.duration,
                        events=[dict(t=e["t"], start=e.get("start"), end=e.get("end"), type=e.get("type")) for e in v.events]) for v in videos}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, key=np.array(keys), t0=np.array(t0s, dtype=np.float32), aug=np.zeros(len(keys), dtype=np.int8),
                        Q=np.stack(rows).astype(np.float32), qkeys=np.array(DESK_KEYS),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)), cfg=np.array(json.dumps(vars(cfg), ensure_ascii=False)))
    log(f"сохранено: {out_path}, окон {len(keys)}, {(time.time() - t_start) / 60:.1f} мин")
    return out_path


# ----------------------------------------------- решающий модуль на ответах чек-листа
# Обучается один раз вне сервиса (sklearn) и хранится в bundle/decision.json; здесь только numpy.
QTEXT = dict(CHECKLIST + DESK_CHECKLIST)

TYPE_TEXT = {   # шаблоны: только то, что можно утверждать по видео, формулировки осторожные
    "leaving_workplace": dict(
        ru="Возможное нарушение: сотрудница покинула рабочее место, оставив его без присмотра. Модель может принять за сотрудницу посетителя, поэтому стоит проверить по кадрам, что вышла именно сотрудница.",
        en="Possible violation: the employee left the workplace unattended. The model may mistake a visitor for the employee, so check in the frames that it is the employee who left."),
    "unsafe_money": dict(
        ru="Возможное нарушение: деньги, сейф или ключи на рабочем месте остались без должной защиты.",
        en="Possible violation: cash, the safe or keys at the workplace were left without proper protection."),
    "stranger_coming": dict(
        ru="Возможное нарушение порядка допуска к рабочему месту: рядом находится посетитель или посторонний, чья личность по видео не подтверждена документом, либо вход в рабочую зону остаётся открытым. Модель могла не заметить, как клиент показывал удостоверение, поэтому стоит проверить кадры.",
        en="Possible violation of the access procedure at the workplace: a visitor or an outsider is nearby and the video does not show their identity being confirmed, or the entrance to the work area is left open. The model may have missed an identity card being shown, so check the frames."),
    "client_handling": dict(
        ru="Возможное нарушение порядка работы с клиентами и их деньгами: операция выполняется не по установленному порядку.",
        en="Possible breach of the procedure for serving clients and handling their money."),
    "other": dict(ru="Возможное нарушение кассовой дисциплины.", en="Possible violation of cash-desk discipline."),
}
# какие признаки (вопрос, порог логита, фраза) упоминать в описании каждого типа
FACTS = {
    "leaving_workplace": [("desk_empty", 1.0, "рабочее место пустое", "the workplace is empty"),
                          ("leaving", 1.0, "человек выходит из помещения", "a person is leaving the room"),
                          ("cash_desk", 2.0, "на столе видны деньги", "banknotes are visible on the desk"),
                          ("safe_open", 2.0, "сейф открыт", "the safe is open")],
    "unsafe_money": [("safe_open", 1.0, "сейф открыт", "the safe is open"), ("cash_desk", 1.0, "на столе видны деньги", "banknotes are visible on the desk"),
                     ("key", 1.0, "ключ лежит на столе или у сейфа", "a key lies on the desk or near the safe"),
                     ("desk_empty", 1.0, "за столом никого нет", "nobody is at the desk")],
    "stranger_coming": [("entering", 1.0, "кто-то входит в помещение", "someone enters the room"), ("visitor", 1.0, "у стола находится посетитель", "a visitor is at the desk"),
                        ("behind_desk", 1.0, "посторонний находится на стороне сотрудницы", "an outsider is on the employee's side of the desk"),
                        ("peeking", 1.0, "кто-то заглядывает в рабочую зону", "someone looks into the work area"),
                        ("door_open", 2.0, "дверь открыта", "the door is open")],
    "client_handling": [("two_visitors", 1.0, "у стола одновременно несколько посетителей", "several visitors are at the desk at the same time"),
                        ("cash_mixed", 1.0, "на столе рядом лежат деньги разных людей", "banknotes of different people lie together on the desk"),
                        ("handover", 1.0, "передают деньги или документы", "money or documents are handed over")],
}


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _smooth(p, kern):
    """Свёртка кривой p с ядром нечётной длины (края дополняются крайними значениями); длина результата = len(p)."""
    if not kern or len(p) < 2:
        return p
    k = np.asarray(kern, dtype=float)
    if len(k) % 2 == 0:
        k = np.append(k, 0.0)
    k = k / k.sum()
    h = len(k) // 2
    pad = np.concatenate([np.full(h, p[0]), p, np.full(h, p[-1])])
    return np.convolve(pad, k, mode="valid")


def derive_features(Qb, base_keys, derived):
    """Ответы чек-листа [n, len(base_keys)] -> признаки [n, len(base_keys) + len(derived)].
    derived: список ["min"|"max", вопрос_a, вопрос_b]; min по логитам — это «и» (например, стол пуст И на нём вещи)."""
    Qb = np.asarray(Qb, dtype=float)
    if not derived:
        return Qb
    ix = {k: i for i, k in enumerate(base_keys)}
    cols = [np.minimum(Qb[:, ix[a]], Qb[:, ix[b]]) if op == "min" else np.maximum(Qb[:, ix[a]], Qb[:, ix[b]]) for op, a, b in derived]
    return np.concatenate([Qb, np.stack(cols, 1)], 1)


class DecisionModel:
    """Окно -> вероятность нарушения (логистическая регрессия на ответах чек-листа), события, тип. Только numpy."""

    def __init__(self, spec):
        self.spec = spec
        self.qkeys = list(spec["qkeys"])                       # вопросы, которые задаются модели
        self.derived = [list(d) for d in spec.get("derived", [])]
        self.questions = [(k, QTEXT[k]) for k in self.qkeys if k in CHECK_KEYS]         # общий вид (16 кадров)
        self.desk_questions = [(k, QTEXT[k]) for k in self.qkeys if k in DESK_KEYS]     # крупный план стола и сейфа
        if self.qkeys != [k for k, _ in self.questions] + [k for k, _ in self.desk_questions]:
            raise ValueError("в decision.json вопросы общего вида должны идти раньше вопросов про стол крупно")
        g = lambda k: np.asarray(spec[k], dtype=float)
        self.mu, self.sd, self.w, self.b = g("mu"), g("sd"), g("w"), float(spec["b"])
        self.type_keys = list(spec.get("type_keys", []))
        if self.type_keys:
            self.tmu, self.tsd, self.tW, self.tb = g("tmu"), g("tsd"), g("tW"), g("tb")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def features(self, Q):
        return derive_features(Q, self.qkeys, self.derived)

    def window_probs(self, Q):
        z = ((self.features(Q) - self.mu) / self.sd) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def type_probs(self, Q):
        if not self.type_keys:
            return None
        z = ((self.features(Q) - self.tmu) / self.tsd) @ self.tW.T + self.tb
        return _softmax(z).mean(0)

    def _after(self, tt, Qo, t_ref, lo, hi, key):
        """Среднее ответа `key` по окнам, начавшимся в [t_ref+lo, t_ref+hi]; None, если таких окон нет (событие у самого конца ролика)."""
        j = self.qkeys.index(key)
        m = (tt >= t_ref + lo - 1e-6) & (tt <= t_ref + hi + 1e-6)
        return float(Qo[m, j].mean()) if m.any() else None

    def is_client_exit(self, e, tt, Qo):
        """«Уход с рабочего места» неверен, если вышел клиент, а сотрудница осталась. Два признака:
        1) после события (через 4–10 с) за столом снова кто-то сидит, а посетителя уже нет: сотрудница на месте;
        2) если после события в ролике кадров нет (событие у самого конца): до события у стола был посетитель, и в окне события
           за столом сидящий есть. При настоящем уходе стол после события пуст, а посетителя у стола до него не было."""
        g = self.spec.get("client_exit")
        if not g or e["type_key"] != "leaving_workplace" or "seated" not in self.qkeys or e["type_p"] < g.get("type_p_min", 0.0):
            return False
        lo, hi, t0 = g.get("lo", 4.0), g.get("hi", 10.0), e["win_t0"]
        has_v = "visitor" in self.qkeys
        s_after = self._after(tt, Qo, t0, lo, hi, "seated")
        if s_after is not None:
            v_after = self._after(tt, Qo, t0, lo, hi, "visitor") if has_v else -99.0
            e["seated_after"], e["visitor_after"] = s_after, v_after
            return s_after > g.get("seated_above", -4.0) and v_after < g.get("visitor_below", 1.0)
        if not has_v:
            return False
        v_before = self._after(tt, Qo, t0, -10.0, -2.0, "visitor")
        s_peak = float(Qo[int(np.argmin(np.abs(tt - t0))), self.qkeys.index("seated")])
        e["seated_after"], e["visitor_after"] = s_peak, (v_before if v_before is not None else -99.0)
        return v_before is not None and v_before > g.get("visitor_before_above", -3.0) and s_peak > g.get("seated_peak_above", -3.0)

    def _events(self, ps, tt, Qo, duration, win_sec):
        """Кривая p -> события (кластеры окон с p не ниже доли от максимума), каждое с типом, фактами и окнами кластера (_idx)."""
        sp = self.spec
        smax = float(ps.max())
        thr = min(max(sp.get("tau_event", 0.0), sp.get("rel_event", 0.8) * smax), smax)
        step = float(np.median(np.diff(tt))) if len(tt) > 1 else 2.0
        gap = max(sp.get("cluster_gap", 5.0), 1.6 * step)
        cl, cur, last = [], [], None
        for i in np.where(ps >= thr)[0]:
            c = tt[i] + win_sec / 2
            if cur and c - last > gap:
                cl.append(cur); cur = []
            cur.append(int(i)); last = c
        if cur:
            cl.append(cur)
        evs = []
        for idx in cl:
            idx = np.array(idx)
            pk = float(ps[idx].max())
            m = idx[ps[idx] >= sp.get("rel_centroid", 0.8) * pk]
            t = float(np.sum((tt[m] + win_sec / 2) * ps[m]) / np.sum(ps[m])) + sp.get("time_offset", 0.0)
            t = min(max(t, 0.3), max(duration - 0.3, 0.3))
            tp = self.type_probs(Qo[m])
            tkey = self.type_keys[int(np.argmax(tp))] if tp is not None else "other"
            if tp is not None and float(tp.max()) < sp.get("other_below", 0.0):
                tkey = "other"            # классификатор типов почти не уверен -> не навязываем конкретный тип и норму
            facts = []
            qm = Qo[m].mean(0)
            for qk, th, ru, en in FACTS.get(tkey, []):
                if qk in self.qkeys and qm[self.qkeys.index(qk)] > th:
                    facts.append((ru, en))
            evs.append(dict(t=t, p=pk, type_key=tkey, type_p=(float(tp.max()) if tp is not None else 0.0), facts=facts,
                            win_t0=float(tt[m][int(np.argmax(ps[m]))]), _idx=idx))
        evs.sort(key=lambda e: -e["p"])
        return evs

    def find(self, starts, Q, duration, win_sec):
        """starts [n], Q [n, len(qkeys)] -> dict(score, events=[dict(t, p, type_key, type_p, facts)])."""
        sp = self.spec
        o = np.argsort(starts)
        tt, Qo = np.asarray(starts, dtype=float)[o], np.asarray(Q, dtype=float)[o]
        ps = _smooth(self.window_probs(Qo), sp.get("kern"))
        smax = float(ps.max())
        if smax < sp.get("tau_video", 0.0):
            return dict(score=smax, events=[], p=ps, t=tt)
        # события «уход с рабочего места», после которых за столом сидит сотрудница (вышел клиент), не считаются нарушением:
        # их окна гасятся, и берётся следующий по силе кластер
        masked, n_masked, dropped = np.zeros(len(ps), bool), 0, []
        pe, evs = ps, self._events(ps, tt, Qo, duration, win_sec)
        for _ in range(4):
            bad = [e for e in evs if self.is_client_exit(e, tt, Qo)]
            if not bad:
                break
            for e in bad:
                lo_t, hi_t = tt[e["_idx"]].min() - win_sec / 2, tt[e["_idx"]].max() + win_sec / 2
                masked |= (tt >= lo_t) & (tt <= hi_t)
                dropped.append(dict(t=e["t"], p=e["p"], type_key=e["type_key"], type_p=e["type_p"],
                                    seated_after=e.get("seated_after", float("nan")), visitor_after=e.get("visitor_after", float("nan"))))
            n_masked += len(bad)
            pe = np.where(masked, 0.0, ps)
            if float(pe.max()) <= 1e-9:
                evs = []
                break
            evs = self._events(pe, tt, Qo, duration, win_sec)
        smax_f = float(pe.max())
        if n_masked and (not evs or smax_f < sp.get("client_exit", {}).get("tau", 0.0)):
            return dict(score=smax_f, events=[], p=pe, t=tt, n_masked=n_masked, dropped=dropped)
        if sp.get("drop_other_if_typed", True) and any(e["type_key"] != "other" for e in evs):
            evs = [e for e in evs if e["type_key"] != "other"]      # «не определено» рядом с событием с типом — лишний шум для проверяющего
        keep = []
        for e in evs:
            if len(keep) >= sp.get("max_events", 2):
                break
            if keep and (e["p"] < sp.get("rel_second", 0.92) * keep[0]["p"] or min(abs(e["t"] - k["t"]) for k in keep) < sp.get("min_sep", 8.0)):
                continue
            keep.append(e)
        for e in keep:
            e.pop("_idx", None)
        return dict(score=smax_f, events=sorted(keep, key=lambda e: e["t"]), p=pe, t=tt, n_masked=n_masked, dropped=dropped)


class ChecklistAnalyzer(Analyzer):
    """Анализатор без дообучения: чек-лист вопросов к замороженной Qwen3-VL + DecisionModel. Тот же интерфейс, что у Analyzer."""

    def __init__(self, model, processor, cfg, types, decision, tau=0.5, min_pos=1, caption_ru=True, batch=None):
        super().__init__(model, processor, cfg, types, tau=tau, min_pos=min_pos)
        self.decision, self.caption_ru = decision, caption_ru
        self.batch = int(batch if batch is not None else os.environ.get("FC_CHECK_BATCH", 4))   # вопросов за один проход

    def warmup(self):
        """Один пробный проход на сером кадре: прогревает CUDA и сразу показывает, что модель работает (вызывается при загрузке)."""
        L = int(self.cfg.long_side)
        w = max(64, int(round(0.55 * L / 32)) * 32)
        frames = [Image.new("RGB", (w, max(64, int(round(L / 32)) * 32)), (128, 128, 128)) for _ in range(n_win_frames(self.cfg))]
        t = time.time()
        z = self.window_checklist(frames, questions=self.decision.questions[:max(2, self.batch)], batch=self.batch)
        log(f"прогрев модели: {time.time() - t:.1f} с")
        return z

    def scan(self, v, verbose=False):
        """Окна по всему видео -> (начала [n], ответы [n, len(qkeys)]). Окна идут «через одно», чтобы при нехватке времени терять равномерно.
        Если в решающем модуле есть вопросы про стол крупно, рамка находится один раз на видео, а кадры вырезаются во временную папку."""
        import tempfile
        cfg, dm = self.cfg, self.decision
        stride = max(cfg.infer_stride, (v.duration - cfg.win_sec) / max(int(cfg.max_windows) - 1, 1))
        starts = window_starts(v.duration, cfg, stride)
        order = list(range(0, len(starts), 2)) + list(range(1, len(starts), 2))
        got, t_start = {}, time.time()
        n_base = len(dm.questions)
        with tempfile.TemporaryDirectory() as tmp:
            files, dsize = [], None
            if dm.desk_questions:
                try:
                    n = len(v.files)
                    pick = [v.files[min(int(round(f * v.duration * cfg.fps)), n - 1)] for f in (0.15, 0.5, 0.85)]
                    box = self.find_desk_box([fit32([Image.open(p).convert("RGB")])[0] for p in pick])
                    files = cut_desk_frames(v.path, box, Path(tmp) / "desk")
                except Exception as ex:
                    log(f"  ! вид «стол крупно» недоступен ({ex}); беру средние значения этих признаков")
                    files = []
            for i in order:
                if len(got) >= min(4, len(order)) and time.time() - t_start > cfg.time_budget_sec:
                    log(f"  ! лимит времени: оценено {len(got)} из {len(starts)} окон")
                    break
                frames, t0a = load_window_frames(v, starts[i], cfg)
                zb = self.window_checklist(fit32(frames), questions=dm.questions, batch=self.batch)
                if dm.desk_questions:
                    if files:
                        dframes, dsize = desk_window_frames(files, t0a, 8, size=dsize)
                        zd = self.window_checklist(dframes, questions=dm.desk_questions, batch=self.batch,
                                                   times=[float(k) for k in range(8)], system=CHECK_SYSTEM_DESK)
                    else:
                        zd = dm.mu[n_base:n_base + len(dm.desk_questions)].astype(np.float32)
                    zb = np.concatenate([zb, zd])
                got[i] = (t0a, zb)
        ks = sorted(got)
        if verbose:
            log(f"{v.key}: окон {len(ks)}, {time.time() - t_start:.0f} с")
        return np.array([got[i][0] for i in ks]), np.stack([got[i][1] for i in ks])

    def analyze(self, v, describe=True):
        cfg = self.cfg
        if not v.files:
            return dict(verdict="no_violation", violations=[])
        starts, Q = self.scan(v)
        res = self.decision.find(starts, Q, v.duration, cfg.win_sec)
        for d in res.get("dropped") or []:
            log(f"{v.key}: событие «уход с места» на {d['t']:.1f} с снято — вышел клиент, сотрудница осталась за столом "
                f"(сидящий после: {d['seated_after']:+.1f}, посетитель после: {d['visitor_after']:+.1f})")
        events = []
        for e in res["events"]:
            te = e["t"]
            key = e["type_key"] if e["type_key"] in self.types else ("other" if "other" in self.types else next(iter(self.types)))
            info = self.types.get(key, dict(name_ru="Прочее подозрительное действие", rule_ref=""))
            tt = TYPE_TEXT.get(key, TYPE_TEXT["other"])
            ru, en = tt["ru"], tt["en"]
            if e["facts"]:
                ru += " По ответам модели на уточняющие вопросы видно: " + "; ".join(f[0] for f in e["facts"]) + "."
                en += " According to the model's answers to follow-up questions: " + "; ".join(f[1] for f in e["facts"]) + "."
            if describe and self.caption_ru:
                try:
                    frames, _ = load_window_frames(v, e["win_t0"], cfg)
                    cap = self.caption(fit32(frames), max_new_tokens=90, lang="ru")
                    if len(cap) >= 20 and re.search("[А-Яа-я]", cap):
                        ru += " Автоматическое описание сцены: " + cap
                except Exception as ex:
                    log(f"  ! описание сцены не получено: {ex}")
            ts = sorted({round(min(max(te + dt, 0), v.duration), 2) for dt in (-1.0, 0.0, 1.0)})
            events.append(dict(
                type_key=key, type=info.get("name_ru") or key, timecode_sec=round(te, 1), score=round(100 * e["p"]),
                description=ru, description_en=en, rule_ref=info.get("rule_ref", ""),
                evidence_frames=[dict(timestamp_sec=t, image_base64=grab_frame_b64(v.path, t, v.duration)) for t in ts]))
        return dict(verdict="violation" if events else "no_violation", violations=events)


# -------------------------------------------------------------- сохранение
def save_bundle(model, cfg, types, tau, min_pos, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "adapter"))
    (out / "pipeline_config.json").write_text(json.dumps(
        dict(cfg=vars(cfg), types=types, tau=tau, min_pos=min_pos), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"сохранено в {out}")


def load_bundle(bundle_dir, load_4bit=None, merge=True):
    """Для сервиса: bundle -> анализатор. Есть decision.json -> ChecklistAnalyzer (без LoRA), иначе базовая модель + LoRA."""
    bd = Path(bundle_dir)
    meta = json.loads((bd / "pipeline_config.json").read_text(encoding="utf-8"))
    cfg = make_cfg(**meta["cfg"])
    if load_4bit is None:
        load_4bit = cfg.load_4bit
    model_id = os.environ.get("BASE_MODEL_PATH") or cfg.model_id
    model, processor = load_base(model_id, load_4bit)
    if (bd / "decision.json").exists():
        model.eval()
        an = ChecklistAnalyzer(model, processor, cfg, meta["types"], DecisionModel.load(bd / "decision.json"),
                               tau=meta.get("tau", 0.5), min_pos=meta.get("min_pos", 1))
        try:
            an.warmup()
        except Exception as ex:
            log(f"! прогрев модели не удался: {ex}")
        return an
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(bd / "adapter"))
    if merge and not load_4bit:
        model = model.merge_and_unload()
    model.eval()
    return Analyzer(model, processor, cfg, meta["types"], tau=meta["tau"], min_pos=meta["min_pos"])
