#!/usr/bin/env python3
"""Extract LTP/LTD nonlinearity (Ap/Ad) and device metrics from Excel.

Preferred input workbook format
------------------------------
Sheet "LTP": Pulse | Cycle_1 | Cycle_2 | ...   (conductance low -> high)
Sheet "LTD": Pulse | Cycle_1 | Cycle_2 | ...   (conductance high -> low)

One curve is sufficient for Ap/Ad, G ratio, and pulse number. C2C is a
user-entered experimental value and is not inferred from curves. A 4-column raw sheet of
Pulse, LTP, Pulse, LTD (with or without headers) is also accepted.

Run
---
python3 som_ltp_ltd_analyzer.py raw_data.xlsx --outdir som_analysis
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def numeric_columns(frame):
    """Return numeric columns after discarding blank/header rows."""
    cols = []
    for col in frame.columns:
        v = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=float)
        if v.size >= 2:
            cols.append(v)
    return cols


def looks_like_pulse(v):
    if v.size < 2:
        return False
    d = np.diff(v)
    return (np.allclose(v, np.round(v)) and np.all(d > 0) and
            np.allclose(d, d[0]) and d[0] in (1, 1.0))


def aligned(curves):
    """Linearly align curves of differing point counts to the longest curve."""
    points = max(len(c) for c in curves)
    xnew = np.linspace(0, 1, points)
    return np.asarray([np.interp(xnew, np.linspace(0, 1, len(c)), c) for c in curves])


def extract_sheet_curves(frame):
    cols = numeric_columns(frame)
    if cols and looks_like_pulse(cols[0]):
        cols = cols[1:]
    if not cols:
        raise ValueError("No conductance column was found.")
    return aligned(cols)


def load_curves(path):
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    by_name = {str(k).strip().lower(): v for k, v in sheets.items()}
    if "ltp" in by_name and "ltd" in by_name:
        return extract_sheet_curves(by_name["ltp"]), extract_sheet_curves(by_name["ltd"])

    # Fallback: first sheet, 2 columns (LTP,LTD) or 4 columns
    # (pulse,LTP,pulse,LTD), optionally with a header row.
    raw = next(iter(sheets.values()))
    cols = numeric_columns(raw)
    if len(cols) >= 4 and looks_like_pulse(cols[0]) and looks_like_pulse(cols[2]):
        return aligned([cols[1]]), aligned([cols[3]])
    if len(cols) >= 2:
        return aligned([cols[0]]), aligned([cols[1]])
    raise ValueError("Use LTP/LTD sheets or a raw sheet with LTP and LTD columns.")


def endpoint_normalize(curve, increasing, gmin, gmax):
    start, end = curve[0], curve[-1]
    if increasing and end <= start:
        raise ValueError("LTP must be ordered low conductance -> high conductance.")
    if not increasing and end >= start:
        raise ValueError("LTD must be ordered high conductance -> low conductance.")
    target_start, target_end = (gmin, gmax) if increasing else (gmax, gmin)
    return target_start + (curve - start) * (target_end - target_start) / (end - start)


def model_ltp(p, a, pmax):
    b = 1.0 / (1.0 - np.exp(-pmax / a))
    return b * (1.0 - np.exp(-p / a))


def model_ltd(p, a, pmax):
    b = 1.0 / (1.0 - np.exp(-pmax / a))
    return 1.0 - b * (1.0 - np.exp((p - pmax) / a))


def fit_a(normalized_curve, kind):
    """Fit endpoint-constrained exponential A using a robust log-grid search."""
    pmax = len(normalized_curve) - 1
    p = np.arange(pmax + 1, dtype=float)
    # Covers strong saturation through practically linear behaviour.
    candidates = pmax * np.logspace(-4, 5, 30000)
    if kind == "ltp":
        predicted = np.array([model_ltp(p, a, pmax) for a in candidates])
    else:
        predicted = np.array([model_ltd(p, a, pmax) for a in candidates])
    mse = np.mean((predicted - normalized_curve[None, :]) ** 2, axis=1)
    idx = int(np.argmin(mse))
    return float(candidates[idx]), predicted[idx], float(mse[idx])


def analyze(ltp_cycles, ltd_cycles, c2c_percent_value=0.0):
    points = max(ltp_cycles.shape[1], ltd_cycles.shape[1])
    ltp_cycles = aligned(ltp_cycles); ltd_cycles = aligned(ltd_cycles)
    if ltp_cycles.shape[1] != points:
        ltp_cycles = aligned([np.interp(np.linspace(0,1,points), np.linspace(0,1,len(c)), c) for c in ltp_cycles])
    if ltd_cycles.shape[1] != points:
        ltd_cycles = aligned([np.interp(np.linspace(0,1,points), np.linspace(0,1,len(c)), c) for c in ltd_cycles])
    raw_ltp, raw_ltd = ltp_cycles.mean(axis=0), ltd_cycles.mean(axis=0)
    gmin = 0.5 * (raw_ltp[0] + raw_ltd[-1])
    gmax = 0.5 * (raw_ltp[-1] + raw_ltd[0])
    ltp = endpoint_normalize(raw_ltp, True, gmin, gmax)
    ltd = endpoint_normalize(raw_ltd, False, gmin, gmax)
    span = gmax - gmin
    ltp_norm = (ltp - gmin) / span
    ltd_norm = (ltd - gmin) / span
    ap, fit_ltp, mse_p = fit_a(ltp_norm, "ltp")
    # The published LTD expression is defined from the low-conductance end
    # (P=0) toward the high-conductance end (P=Pmax).  Experimental LTD is
    # normally recorded high->low, hence the reversal only for fitting.
    ad, fit_ltd_forward, mse_d = fit_a(ltd_norm[::-1], "ltd")
    c2c = float(c2c_percent_value)
    if c2c < 0:
        raise ValueError("C2C must be zero or greater.")
    metrics = {
        "Gmin_S": gmin, "Gmax_S": gmax, "G_ratio": gmax / gmin,
        "points": points, "pulses_Pmax": points - 1,
        "Ap_pulses": ap, "Ap_over_Pmax": ap / (points - 1),
        "Ad_pulses": ad, "Ad_over_Pmax": ad / (points - 1),
        "LTP_fit_MSE_normalized": mse_p, "LTD_fit_MSE_normalized": mse_d,
        "C2C_percent_input": c2c,
    }
    return metrics, ltp, ltd, fit_ltp * span + gmin, fit_ltd_forward[::-1] * span + gmin


def save_outputs(outdir, metrics, ltp, ltd, fit_ltp, fit_ltd):
    outdir.mkdir(parents=True, exist_ok=True)
    p = np.arange(len(ltp))
    pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}).to_excel(outdir / "som_extracted_metrics.xlsx", index=False)
    # Add fitted curves as a second worksheet.
    with pd.ExcelWriter(outdir / "som_curve_fit.xlsx", engine="openpyxl") as writer:
        pd.DataFrame({"Pulse": p, "LTP_matched_S": ltp, "LTP_fit_S": fit_ltp,
                      "LTD_matched_S": ltd, "LTD_fit_S": fit_ltd}).to_excel(writer, sheet_name="Curve_fit", index=False)
        pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}).to_excel(writer, sheet_name="Metrics", index=False)
    plt.figure(figsize=(7, 4.7))
    plt.plot(p, ltp, "o", ms=4, label="LTP data"); plt.plot(p, fit_ltp, "-", lw=2, label="LTP exponential fit")
    plt.plot(p, ltd, "s", ms=4, label="LTD data"); plt.plot(p, fit_ltd, "-", lw=2, label="LTD exponential fit")
    plt.xlabel("Pulse number, P"); plt.ylabel("Conductance (S)"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "som_ltp_ltd_fit.png", dpi=300); plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extract Ap, Ad, pulse number, and G ratio from LTP/LTD Excel raw data.")
    parser.add_argument("excel", help="Input .xlsx/.xls file")
    parser.add_argument("--outdir", default="som_extraction_results")
    parser.add_argument("--c2c", type=float, default=0.0, help="Measured C2C (%) entered by user")
    args = parser.parse_args()
    ltp_cycles, ltd_cycles = load_curves(args.excel)
    metrics, ltp, ltd, fit_ltp, fit_ltd = analyze(ltp_cycles, ltd_cycles, args.c2c)
    save_outputs(Path(args.outdir), metrics, ltp, ltd, fit_ltp, fit_ltd)
    print("\nSOM LTP/LTD extraction")
    for key, value in metrics.items():
        print(f"{key}: {'not measurable (need >=2 cycles)' if value is None else value}")
    print(f"\nSaved: {args.outdir}/som_extracted_metrics.xlsx")
    print(f"Saved: {args.outdir}/som_curve_fit.xlsx")
    print(f"Saved: {args.outdir}/som_ltp_ltd_fit.png")


if __name__ == "__main__":
    main()
