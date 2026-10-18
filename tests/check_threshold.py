"""Что делает порог вывода — без GPU и без видео.

Берёт готовые ряды ответов чек-листа по 40 роликам обучающего набора (fixtures/client_exit_series.json), прогоняет их через
DecisionModel из decision.json и печатает: какие ролики остались «с нарушением», какие стали «нарушение не обнаружено»,
какие настоящие нарушения порог убрал и какие ложные срабатывания остались. Оценки — те же числа, что в интерфейсе
(«Оценка модели: N из 100»), но по сохранённым рядам (расхождение с экраном обычно 1–3 пункта).

Запуск:  python check_threshold.py --repo <каталог с fc_pipeline.py> --decision <decision.json> [--fixture <json>]
Код возврата 1, если порог нарушен: в ответе есть событие с оценкой ниже порога.
"""
import sys, json, copy, argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--decision", required=True)
ap.add_argument("--fixture", default=str(Path(__file__).resolve().parent / "fixtures" / "client_exit_series.json"))
a = ap.parse_args()
sys.path.insert(0, a.repo)
import fc_pipeline as fc

spec = json.loads(Path(a.decision).read_text(encoding="utf-8"))
fx = json.loads(Path(a.fixture).read_text(encoding="utf-8"))
assert fx["qkeys"] == spec["qkeys"], "порядок вопросов в фикстуре и в decision.json разный"
W = float(fx["win_sec"])
tau = max(float(spec.get("tau_video", 0.0)), float(spec.get("tau_event", 0.0)))
spec0 = copy.deepcopy(spec); spec0["tau_video"] = spec0["tau_event"] = 0.0
dm, dm0 = fc.DecisionModel(spec), fc.DecisionModel(spec0)


def run(m, v):
    f = fx["videos"][v]
    return m.find(np.array(f["starts"], float), np.array(f["Q"], float), float(f["duration"]), W)


def num(v):
    return int(v.split("+")[0])


bad, lost, gained, still_fp, hit, already = [], [], [], [], [], []
for v in sorted(fx["videos"], key=num):
    f = fx["videos"][v]
    r, r0 = run(dm, v), run(dm0, v)
    for e in r["events"]:
        if e["p"] < tau - 1e-9:
            bad.append((v, round(e["t"], 1), round(100 * e["p"])))
    peak = max([e["p"] for e in r["events"]], default=0.0)
    peak0 = max([e["p"] for e in r0["events"]], default=0.0)
    if f["violation"]:
        (hit if r["events"] else lost).append((v, round(100 * peak0)))
    elif r["events"]:
        still_fp.append((v, round(100 * peak), [e["type_key"] for e in r["events"]]))
    elif r0["events"]:
        gained.append((v, round(100 * peak0), []))          # событие было, порог его убрал
    else:
        already.append(v)                                    # чистым признан и без порога (правило «вышел клиент»)

nv = sum(f["violation"] for f in fx["videos"].values()); nc = len(fx["videos"]) - nv
print(f"Порог вывода: оценка не ниже {round(100 * tau + 0.5) if tau > 0 else 0} из 100 (tau_video = tau_event = {tau:g}); 0 = порога нет")
print(f"  нарушения найдены в {len(hit)} роликах из {nv}; потеряно: " + (", ".join(f"Х-{v} (оценка {s})" for v, s in lost) if lost else "ни одного"))
print(f"  чистые ролики: признано чистыми {len(gained) + len(already)} из {nc}; ложных срабатываний {len(still_fp)}")
if already:
    print("    без порога уже чистые (вышел клиент, а не сотрудница): " + ", ".join(f"Х-{v}" for v in already))
if gained:
    print("    порог убрал событие: " + ", ".join(f"Х-{v} (было {s})" for v, s, _ in gained))
print("  ложные срабатывания (чистые ролики, у которых модель всё ещё показывает событие), по убыванию оценки:")
for v, s, ty in sorted(still_fp, key=lambda x: -x[1]):
    print(f"    Х-{v:<6s} оценка {s:3d}   {', '.join(ty)}")
if bad:
    print("! события ниже порога в ответе:", bad)
    print("ПРОВЕРКА ПОРОГА: НЕ ПРОЙДЕНА"); sys.exit(1)
print("ПРОВЕРКА ПОРОГА: ПРОЙДЕНА (ни одного события ниже порога)")
