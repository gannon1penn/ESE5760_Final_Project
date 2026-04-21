import numpy as np


def crossing_time(df, threshold, direction="falling"):
    t = df["time"].to_numpy()
    y = df["value"].to_numpy()

    if direction == "falling":
        indices = np.where(y <= threshold)[0]
    else:
        indices = np.where(y >= threshold)[0]

    if len(indices) == 0:
        return np.nan

    return t[indices[0]]


def edge_metrics(df):
    initial = df["value"].iloc[0]
    final = df["value"].iloc[-1]
    v_max = df["value"].max()
    v_min = df["value"].min()

    if initial > final:
        direction = "falling"
        high = initial
        low = final
    else:
        direction = "rising"
        high = final
        low = initial

    amp = high - low

    t90 = crossing_time(df, low + 0.9 * amp, direction=direction)
    t50 = crossing_time(df, low + 0.5 * amp, direction=direction)
    t10 = crossing_time(df, low + 0.1 * amp, direction=direction)

    return {
        "initial_value": initial,
        "final_value": final,
        "min_value": v_min,
        "max_value": v_max,
        "direction": direction,
        "t90": t90,
        "t50": t50,
        "t10": t10,
        "transition_time": abs(t10 - t90) if not np.isnan(t10) and not np.isnan(t90) else np.nan
    }


def estimate_delay(input_df, output_df):
    in_initial = input_df["value"].iloc[0]
    in_final = input_df["value"].iloc[-1]
    out_initial = output_df["value"].iloc[0]
    out_final = output_df["value"].iloc[-1]

    in_threshold = in_final + 0.5 * (in_initial - in_final)
    out_threshold = out_final + 0.5 * (out_initial - out_final)

    t_in = crossing_time(input_df, in_threshold, direction="falling")
    t_out = crossing_time(output_df, out_threshold, direction="falling")

    return {
        "input_t50": t_in,
        "output_t50": t_out,
        "delay_s": t_out - t_in if not np.isnan(t_in) and not np.isnan(t_out) else np.nan
    }


def current_metrics(df, vdd=1.1):
    t = df["time"].to_numpy()
    i = df["value"].to_numpy()

    current_draw = -i
    power = vdd * current_draw

    return {
        "avg_current_A": np.mean(current_draw),
        "peak_current_A": np.max(current_draw),
        "avg_power_W": np.mean(power),
        "peak_power_W": np.max(power),
        "energy_J": np.trapz(power, t),
        "charge_C": np.trapz(current_draw, t),
    }


def seu_metrics(df):
    t = df["time"].to_numpy()
    y = df["value"].to_numpy()

    baseline = y[0]
    min_idx = np.argmin(y)
    min_voltage = y[min_idx]
    min_time = t[min_idx]

    dip = baseline - min_voltage
    area = np.trapz(np.maximum(baseline - y, 0), t)

    def recovery_time(target_ratio):
        target = target_ratio * baseline
        idx = np.where((t > min_time) & (y >= target))[0]
        if len(idx) == 0:
            return np.nan
        return t[idx[0]] - min_time

    return {
        "baseline_V": baseline,
        "min_voltage_V": min_voltage,
        "dip_V": dip,
        "min_time_s": min_time,
        "recovery_90_s": recovery_time(0.90),
        "recovery_95_s": recovery_time(0.95),
        "recovery_99_s": recovery_time(0.99),
        "area_below_baseline": area,
    }


def snm_proxy(df):
    y = df["value"].to_numpy()
    x = df["time"].to_numpy()

    peak_idx = np.argmax(y)

    return {
        "peak_metric": y[peak_idx],
        "peak_x": x[peak_idx],
        "curve_area": np.trapz(y, x),
    }