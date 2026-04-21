from pathlib import Path
import pandas as pd


def load_two_column_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "time", df.columns[1]: "value"})
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def load_multi_pair_csv(path):
    raw = pd.read_csv(path)
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    results = {}
    cols = list(raw.columns)

    for i in range(0, len(cols), 2):
        x_col = cols[i]
        y_col = cols[i + 1]
        name = x_col.replace(" X", "").strip()

        df = raw[[x_col, y_col]].copy()
        df.columns = ["time", "value"]
        df = df.dropna().reset_index(drop=True)

        results[name] = df

    return results