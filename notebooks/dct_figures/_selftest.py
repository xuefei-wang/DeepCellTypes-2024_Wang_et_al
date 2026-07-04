"""Self-test: reproduce the paper's headline cell-type numbers.

Scores ``deepcelltypes_test_prediction.csv`` with the hierarchical collapse
(raw argmax, no CT abstention; classes below the msup50 floor,
``min_support=50``, are excluded from the macro/weighted mean — the
DeepCell Types paper-headline row) and checks the result against the
published values within ±0.1.

Run: ``python -m notebooks.dct_figures._selftest``
"""

from __future__ import annotations

from . import paths
from .scoring import CT2IDX, score_csv

_EXPECTED = {
    "macro_acc": 85.17,
    "macro_f1": 84.44,
    "weighted_acc": 91.57,
    "weighted_f1": 91.52,
}
_TOL = 0.1


def main() -> int:
    csv_path = paths.need(paths.OUTPUT / "deepcelltypes_test_prediction.csv")
    s = score_csv(csv_path, CT2IDX)

    print(f"n_cells={s['n_cells']}  n_kept={s['n_kept']}  "
          f"coverage={s['coverage'] * 100:.2f}%")
    ok = True
    for key in ("macro_acc", "macro_f1", "weighted_acc", "weighted_f1"):
        got = s[key]
        exp = _EXPECTED[key]
        delta = got - exp
        flag = "OK " if abs(delta) <= _TOL else "FAIL"
        if abs(delta) > _TOL:
            ok = False
        print(f"  [{flag}] {key:13s} = {got:6.2f}%  "
              f"(expected {exp:6.2f}%, delta {delta:+.3f})")

    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
