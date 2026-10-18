"""Правило «вышел клиент, а не сотрудница» — проверка без GPU и без видео.

Берёт готовые ряды ответов чек-листа по роликам обучающего набора (tests/fixtures/client_exit_series.json: окно 8 с, шаг 2 с,
18 чисел logit(YES)-logit(NO) на окно) и прогоняет их через DecisionModel из bundle/decision.json. Это тот же код и те же числа,
что работают в сервисе после модели, поэтому тест ловит поломку правила, не запуская Qwen3-VL.

Запуск (из корня проекта):   python tests/test_client_exit.py      или      pytest tests/test_client_exit.py
Другие пути:                 --repo <каталог с fc_pipeline.py и bundle/>   --decision <decision.json>   --fixture <json>

Числа проверяются на модели, обученной на всех 40 роликах (так она и стоит в сервисе). Если решающий модуль переобучить,
ожидаемые списки роликов ниже (EXPECT_CLEAN) надо пересмотреть, а не подгонять правило под них.
"""
import sys, json, copy, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXPECT_CLEAN = {"5+48", "39"}                           # ролики, где у стола осталась сотрудница, а вышел клиент -> «нарушения нет»
LEAVING_GT = ("13", "15", "22", "24", "26", "28")       # настоящие уходы сотрудницы (разметка команды) — должны остаться
_CTX = {}


def _paths(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo"); ap.add_argument("--decision"); ap.add_argument("--fixture")
    a, _ = ap.parse_known_args(argv if argv is not None else [])
    repo = Path(a.repo) if a.repo else (HERE.parent if (HERE.parent / "fc_pipeline.py").exists() else HERE)
    decision = Path(a.decision) if a.decision else repo / "bundle" / "decision.json"
    fixture = Path(a.fixture) if a.fixture else next((p for p in (HERE / "fixtures" / "client_exit_series.json", HERE / "client_exit_series.json") if p.exists()))
    return repo, decision, fixture


def ctx():
    if not _CTX:
        repo, decision, fixture = _paths(sys.argv[1:] if __name__ == "__main__" else None)
        sys.path.insert(0, str(repo))
        import fc_pipeline as fc
        spec = json.loads(decision.read_text(encoding="utf-8"))
        fx = json.loads(fixture.read_text(encoding="utf-8"))
        assert fx["qkeys"] == spec["qkeys"], "порядок вопросов в фикстуре и в decision.json разный"
        spec["tau_video"] = spec["tau_event"] = 0.0          # тесты проверяют правило «вышел клиент»; порог вывода (set_threshold.sh) на них не влияет
        _CTX.update(fc=fc, spec=spec, fx=fx, W=float(fx["win_sec"]))
    return _CTX


def series(v):
    f = ctx()["fx"]["videos"][v]
    return np.array(f["starts"], float), np.array(f["Q"], float), float(f["duration"]), f


def run_all(spec):
    c = ctx(); dm = c["fc"].DecisionModel(spec)
    out = {}
    for v in c["fx"]["videos"]:
        t, Q, dur, _ = series(v)
        out[v] = dm.find(t, Q, dur, c["W"])
    return out


def truncated(v, t_event_window):
    """Ряд ролика, обрезанный сразу после события: окон после него нет (событие «у самого конца» записи)."""
    t, Q, _, _ = series(v)
    keep = t <= t_event_window + 2.0 + 1e-6
    return t[keep], Q[keep], float(t[keep][-1] + ctx()["W"])


# ----------------------------------------------------------------------------- тесты
def test_bundle_has_rule():
    g = ctx()["spec"].get("client_exit")
    assert g and g["lo"] == 4.0 and g["hi"] == 10.0 and g["seated_above"] == -4.0 and g["visitor_below"] == 1.0 and g["type_p_min"] == 0.5 and g["tau"] == 0.5, g
    assert ctx()["spec"].get("drop_other_if_typed") is True


def test_client_exit_gives_no_violation():
    r = run_all(ctx()["spec"])
    for v in EXPECT_CLEAN:
        assert r[v]["events"] == [], (v, [(e["t"], e["type_key"]) for e in r[v]["events"]])
        assert r[v]["dropped"] and all(d["type_key"] == "leaving_workplace" for d in r[v]["dropped"]), (v, r[v]["dropped"])


def test_employee_leaving_is_kept():
    r = run_all(ctx()["spec"])
    for v in LEAVING_GT:
        gt = [t for t, _ in ctx()["fx"]["videos"][v]["gt"]]
        ev = [e for e in r[v]["events"] if e["type_key"] == "leaving_workplace"]
        assert ev and not r[v]["dropped"], (v, r[v]["events"], r[v]["dropped"])
        assert min(abs(e["t"] - g) for e in ev for g in gt) <= 6.0, (v, [e["t"] for e in ev], gt)


def test_no_violation_video_is_lost():
    r = run_all(ctx()["spec"])
    lost = [v for v, f in ctx()["fx"]["videos"].items() if f["violation"] and not r[v]["events"]]
    assert not lost, f"потеряны ролики с нарушением: {lost}"


def test_only_client_exits_become_clean():
    r = run_all(ctx()["spec"])
    clean = {v for v in r if not r[v]["events"]}
    assert clean == EXPECT_CLEAN, f"без событий остались: {sorted(clean)}, ожидалось {sorted(EXPECT_CLEAN)}"


def test_without_rule_behaviour_is_old():
    spec = copy.deepcopy(ctx()["spec"]); spec.pop("client_exit")
    r = run_all(spec)
    for v in EXPECT_CLEAN:
        assert r[v]["events"], v                      # без правила события возвращаются
        assert r[v]["dropped"] == [] and r[v]["n_masked"] == 0
    assert all(r[v]["events"] for v in r)             # и ни один ролик не остаётся без событий (политика «пропуск недопустим»)


def test_event_next_to_other_event_survives_masking():
    # ролик 38: событие «уход» в конце сняли, но другое событие (тип не определён) осталось -> вердикт «нарушение» сохраняется
    r = run_all(ctx()["spec"])["38"]
    assert r["n_masked"] == 1 and len(r["events"]) == 1 and r["events"][0]["type_key"] == "other", r


def test_rule_works_when_event_is_at_the_end_of_video():
    # события у самого конца записи: после него нет окон, правило смотрит на посетителя до события и на сидящего в окне события
    dm = ctx()["fc"].DecisionModel(ctx()["spec"])
    for v, t_ev in (("5+48", 50.0), ("39", 68.0)):                       # t_ev — начало окна пика события
        t, Q, dur = truncated(v, t_ev)
        r = dm.find(t, Q, dur, ctx()["W"])
        assert r["events"] == [] and r["dropped"], (v, r["events"], r["dropped"])
    for v, t_ev in (("13", 44.0), ("15", 13.0), ("22", 13.0), ("24", 6.0), ("26", 12.0), ("28", 8.0)):
        t, Q, dur = truncated(v, t_ev)
        r = dm.find(t, Q, dur, ctx()["W"])
        assert [e for e in r["events"] if e["type_key"] == "leaving_workplace"] and not r["dropped"], (v, r["events"], r["dropped"])


def test_gate_applies_only_to_confident_leaving_events():
    dm = ctx()["fc"].DecisionModel(ctx()["spec"])
    t, Q, dur, _ = series("5+48")
    base = dict(t=56.0, p=0.9, win_t0=50.0, facts=[])
    assert dm.is_client_exit(dict(base, type_key="leaving_workplace", type_p=0.84), t, Q) is True
    assert dm.is_client_exit(dict(base, type_key="leaving_workplace", type_p=0.49), t, Q) is False     # тип не уверенный -> не трогаем
    assert dm.is_client_exit(dict(base, type_key="stranger_coming", type_p=0.9), t, Q) is False         # другой тип -> не трогаем
    assert dm.is_client_exit(dict(base, type_key="other", type_p=0.9), t, Q) is False
    spec = copy.deepcopy(ctx()["spec"]); spec.pop("client_exit")
    assert ctx()["fc"].DecisionModel(spec).is_client_exit(dict(base, type_key="leaving_workplace", type_p=0.84), t, Q) is False   # правило выключено


def test_edge_cases_do_not_crash():
    fc = ctx()["fc"]; dm = fc.DecisionModel(ctx()["spec"]); W = ctx()["W"]
    t, Q, dur, _ = series("22")
    for n in (1, 2, 3):                                                   # очень короткие записи
        r = dm.find(t[:n], Q[:n], float(t[n - 1] + W), W)
        assert isinstance(r["events"], list) and 0.0 <= r["score"] <= 1.0
    r = dm.find(t[::-1], Q[::-1], dur, W)                                 # окна приходят не по порядку
    assert r["events"] and r["events"][0]["type_key"] == "leaving_workplace"
    r = dm.find(t, np.zeros_like(Q), dur, W)                              # «пустые» ответы модели
    assert isinstance(r["events"], list)
    saved = dm.is_client_exit
    dm.is_client_exit = lambda e, tt, Qo: True                            # снять всё подряд: ответ должен быть «событий нет», без падения
    try:
        r = dm.find(t, Q, dur, W)
    finally:
        dm.is_client_exit = saved
    assert r["events"] == [] and r["score"] < ctx()["spec"]["client_exit"]["tau"] and r["n_masked"] >= 1


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    ctx()
    bad = 0
    for fn in TESTS:
        try:
            fn(); print("  ok  ", fn.__name__)
        except Exception as ex:
            bad += 1; print("  FAIL", fn.__name__, "->", type(ex).__name__, str(ex)[:300])
    r = run_all(ctx()["spec"])
    print("\nроликов без событий:", sorted(v for v in r if not r[v]["events"]))
    for v in sorted(r):
        if r[v]["dropped"]:
            print(f"  {v}: снято {[(round(d['t'], 1), 'сидящий после %+.1f' % d['seated_after'], 'посетитель после %+.1f' % d['visitor_after']) for d in r[v]['dropped']]}")
    print("CLIENT-EXIT TESTS " + ("FAILED" if bad else "PASSED"))
    sys.exit(1 if bad else 0)
