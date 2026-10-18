"""Порог вывода: события с оценкой ниже N (из 100) не показываются; если выше порога ничего нет — «Нарушение не обнаружено».

    python scripts/set_threshold.py 50     # порог 50 (так стоит в поставке)
    python scripts/set_threshold.py 25     # мягче
    python scripts/set_threshold.py 0      # порога нет: показывается самое сильное событие в любом ролике

Меняются только tau_video и tau_event в bundle/decision.json (50 записывается как 0.495: в интерфейсе оценка округляется).
Затем на готовых рядах ответов 40 роликов печатается, что порог делает (tests/check_threshold.py); если проверка не прошла,
файл возвращается. Сервис после смены порога нужно перезапустить (он читает decision.json при старте).
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "bundle" / "decision.json"


def main(argv):
    if len(argv) != 2 or not argv[1].isdigit() or not 0 <= int(argv[1]) <= 99:
        print(__doc__); return 2
    n = int(argv[1])
    tau = 0.0 if n == 0 else round((n - 0.5) / 100.0, 4)
    old_text = DEC.read_text(encoding="utf-8")
    d = json.loads(old_text)
    old = (d.get("tau_video"), d.get("tau_event"))
    d["tau_video"] = d["tau_event"] = tau
    DEC.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{DEC}: tau_video, tau_event {old} -> ({tau}, {tau})")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "check_threshold.py"), "--repo", str(ROOT), "--decision", str(DEC)])
    if r.returncode != 0:
        DEC.write_text(old_text, encoding="utf-8")
        print("Проверка не прошла: decision.json возвращён."); return 1
    print("Готово. Перезапустите сервис, чтобы он прочитал новый порог.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
