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


def epoch_order(pos, neg, ratio, rng):
    k = min(len(neg), int(round(ratio * len(pos))))
    order = list(pos) + rng.sample(neg, k)
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
    log(f"окон: позитивных {len(pos)}, негативных {len(neg)} (hard: {sum(1 for x in neg if x['kind'] == 'hard')})")
    if not pos:
        raise RuntimeError("нет позитивных окон — проверьте таймкоды нарушений")
    vids = {v.key: v for v in train_videos}
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    per_epoch = len(pos) + min(len(neg), int(round(cfg.neg_ratio * len(pos))))
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
        for s in epoch_order(pos, neg, cfg.neg_ratio, rng):
            enc, A = encode_example(processor, vids[s["vid"]], s, cfg, types, rng, augment=True)
            batch = {k: (x.to(dev, dtype=dtype) if x.is_floating_point() else x.to(dev)) for k, x in enc.items()}
            with torch.autocast(dev.type, dtype=dtype, enabled=on_cuda):
                out = model(**batch, use_cache=False, logits_to_keep=A + 1)
            logits = out.logits[0, :-1].float()
            tgt = batch["input_ids"][0, -A:]
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
        self.times = [i / cfg.fps for i in range(n_win_frames(cfg))]

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


# -------------------------------------------------------------- сохранение
def save_bundle(model, cfg, types, tau, min_pos, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "adapter"))
    (out / "pipeline_config.json").write_text(json.dumps(
        dict(cfg=vars(cfg), types=types, tau=tau, min_pos=min_pos), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"сохранено в {out}")


def load_bundle(bundle_dir, load_4bit=None, merge=True):
    """Для сервиса: базовая модель + LoRA из bundle -> Analyzer."""
    bd = Path(bundle_dir)
    meta = json.loads((bd / "pipeline_config.json").read_text(encoding="utf-8"))
    cfg = make_cfg(**meta["cfg"])
    if load_4bit is None:
        load_4bit = cfg.load_4bit
    model, processor = load_base(cfg.model_id, load_4bit)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(bd / "adapter"))
    if merge and not load_4bit:
        model = model.merge_and_unload()
    model.eval()
    return Analyzer(model, processor, cfg, meta["types"], tau=meta["tau"], min_pos=meta["min_pos"])
