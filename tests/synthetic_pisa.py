"""Small FAKE PISA files with the real file names and column names, for testing Phase 0 onwards.

    from tests.synthetic_pisa import make_fake_pisa
    make_fake_pisa("data/synthetic/raw")      # writes intl/2018/CY07_MSU_STU_QQQ.sav, ...

The generating process is simple and known (scores rise with ESCS, fall with
repetition and bullying, drop between 2018 and 2022), so later phases can check
that they recover sensible signs. Some undeclared PISA missing codes (9999 in
indices, 9 in a categorical item) are planted to test prep.clean.drop_missing_codes.
NOTHING produced from these files is a finding about real students.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = {"AAA": 0.6, "BBB": 0.2, "CCC": -0.1, "DDD": -0.5, "EEE": -0.9, "PHL": -1.1}
FILES = {2018: ("CY07_MSU_STU_QQQ.sav", "CY07_MSU_SCH_QQQ.sav"),
         2022: ("CY08MSP_STU_QQQ.SAV", "CY08MSP_SCH_QQQ.SAV")}
RENAMED = {2018: {"BULLIED": "BEINGBULLIED", "DISCLIM": "DISCLIMA"}, 2022: {}}


def _cycle(year: int, rng: np.random.Generator, n_students: int, n_schools: int):
    stu_rows, sch_rows = [], []
    shift = -0.15 if year == 2022 else 0.0                     # the 2022 drop
    for c_i, (cnt, mu) in enumerate(COUNTRIES.items()):
        cnt_num = 100 + c_i
        sch_id = cnt_num * 100000 + np.arange(1, n_schools + 1)
        sch_escs = rng.normal(mu, 0.6, n_schools)
        max_comm = 6 if year == 2022 else 5
        comm = np.clip(np.round(3 + 1.5 * sch_escs + rng.normal(0, 1, n_schools)), 1, max_comm)
        stype = np.where(rng.random(n_schools) < (0.85 if cnt == "PHL" else 0.7), 1, 2)
        sch = pd.DataFrame({
            "CNT": cnt, "CNTSCHID": sch_id, "SC001Q01TA": comm, "SC013Q01TA": stype,
            "STRATIO": np.clip(rng.normal(25 - 5 * sch_escs, 6), 3, 90),
            "EDUSHORT": rng.normal(-0.4 * sch_escs, 1), "STAFFSHORT": rng.normal(-0.3 * sch_escs, 1)})
        sch.loc[rng.random(n_schools) < 0.05, "SC001Q01TA"] = 9   # undeclared missing code
        sch_rows.append(sch)

        k = rng.integers(0, n_schools, n_students)
        escs = sch_escs[k] + rng.normal(0, 0.8, n_students)
        repeat = (rng.random(n_students) < 1 / (1 + np.exp(1.5 + 1.2 * escs))).astype(float)
        lang = np.where(rng.random(n_students) < (0.85 if cnt == "PHL" else 0.15), 2, 1)
        bullied = rng.normal(0.3 - 0.3 * escs, 1, n_students)
        belong = rng.normal(0.2 * escs, 1, n_students)
        ability = (0.9 * escs - 0.5 * repeat - 0.15 * bullied - 0.2 * (lang == 2) + 0.4 * mu
                   + shift + rng.normal(0, 0.7, n_students))
        stu = pd.DataFrame({
            "CNT": cnt, "CNTSCHID": sch_id[k], "CNTSTUID": cnt_num * 100000 + np.arange(n_students),
            "STRATUM": [f"{cnt}{1 + s % 4:02d}" for s in k],
            "W_FSTUWT": rng.lognormal(3 + 0.3 * c_i, 0.4, n_students),
            "ESCS": escs, "HOMEPOS": escs * 0.8 + rng.normal(0, 0.5, n_students),
            "HISEI": np.clip(50 + 15 * escs + rng.normal(0, 10, n_students), 11, 89),
            "PAREDINT": np.clip(np.round(12 + 2.5 * escs + rng.normal(0, 2, n_students)), 3, 18),
            "ST004D01T": rng.integers(1, 3, n_students), "GRADE": np.where(repeat == 1, -1, 0),
            "REPEAT": repeat, "DURECEC": np.clip(np.round(2 + escs + rng.normal(0, 1, n_students)), 0, 5),
            "IMMIG": np.where(rng.random(n_students) < (0.01 if cnt == "PHL" else 0.1), 3, 1),
            "ST022Q01TA": lang, "BELONG": belong, "TEACHSUP": rng.normal(0.1 * escs, 1, n_students),
            "BULLIED": bullied, "DISCLIM": rng.normal(0.2 * escs, 1, n_students)})
        for dom, off in (("MATH", 0.0), ("READ", 0.1), ("SCIE", 0.05)):
            for i in range(1, 11):
                stu[f"PV{i}{dom}"] = 470 + 75 * (ability + off) + rng.normal(0, 25, n_students)
        stu.loc[rng.random(n_students) < 0.02, "BELONG"] = 9999      # undeclared missing code
        stu.loc[rng.random(n_students) < 0.05, "DURECEC"] = np.nan
        stu_rows.append(stu.rename(columns=RENAMED[year]))
    return pd.concat(stu_rows, ignore_index=True), pd.concat(sch_rows, ignore_index=True)


def make_fake_pisa(root: str | Path, seed: int = 138, n_students: int = 900, n_schools: int = 40) -> Path:
    """Write fake international files for 2018 and 2022 under root/intl/<year>/ (PHL included)."""
    import pyreadstat

    rng = np.random.default_rng(seed)
    root = Path(root)
    for year, (stu_name, sch_name) in FILES.items():
        stu, sch = _cycle(year, rng, n_students, n_schools)
        d = root / "intl" / str(year)
        d.mkdir(parents=True, exist_ok=True)
        pyreadstat.write_sav(stu, str(d / stu_name))
        pyreadstat.write_sav(sch, str(d / sch_name))
    return root


if __name__ == "__main__":
    print(make_fake_pisa("data/synthetic/raw"))
