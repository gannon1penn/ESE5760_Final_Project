import pandas as pd
from pathlib import Path

from src.io_utils import load_two_column_csv, load_multi_pair_csv
from src.metrics import edge_metrics, estimate_delay, current_metrics, seu_metrics, snm_proxy


def run_analysis(data_dir):
    data_dir = Path(data_dir)

    d = load_two_column_csv(data_dir / "10T_Test_D.csv")
    d2 = load_two_column_csv(data_dir / "10T_Test_D2.csv")
    q1b = load_two_column_csv(data_dir / "QB1_TEST_CSV.csv")
    current = load_two_column_csv(data_dir / "10T_Test_I.csv")
    seu = load_multi_pair_csv(data_dir / "N0_Test.csv")
    snm = load_two_column_csv(data_dir / "Quatro_SNM_test.csv")

    transient_summary = pd.DataFrame([
        {"signal": "D", **edge_metrics(d)},
        {"signal": "D2", **edge_metrics(d2)},
        {"signal": "Q1B", **edge_metrics(q1b)},
    ])

    delay_summary = pd.DataFrame([
        {"path": "D -> Q1B", **estimate_delay(d, q1b)},
    ])

    power_summary = pd.DataFrame([
        {"signal": "I(VDD)", **current_metrics(current, vdd=1.1)},
    ])

    seu_summary = []
    for name, df in seu.items():
        seu_summary.append({"scenario": name, **seu_metrics(df)})
    seu_summary = pd.DataFrame(seu_summary)

    snm_summary = pd.DataFrame([
        {"curve": "Quatro_SNM_test", **snm_proxy(snm)}
    ])

    return {
        "d": d,
        "d2": d2,
        "q1b": q1b,
        "current": current,
        "seu": seu,
        "snm": snm,
        "transient_summary": transient_summary,
        "delay_summary": delay_summary,
        "power_summary": power_summary,
        "seu_summary": seu_summary,
        "snm_summary": snm_summary,
    }