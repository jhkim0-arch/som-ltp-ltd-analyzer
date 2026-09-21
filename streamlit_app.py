"""Browser-based SOM LTP/LTD data analyzer.

Run locally: streamlit run streamlit_app.py
"""
from io import BytesIO
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from som_ltp_ltd_analyzer import analyze, load_curves


st.set_page_config(page_title="SOM LTP/LTD Analyzer", page_icon="⚡", layout="wide")
st.title("SOM LTP/LTD Analyzer")
st.caption("Upload raw conductance data to extract G ratio, pulse number, Ap, and Ad. Enter the measured C2C value yourself.")

with st.sidebar:
    st.header("Input data")
    uploaded = st.file_uploader("Raw Excel file", type=["xlsx", "xls"])
    st.info("Preferred format: separate `LTP` and `LTD` sheets. Each sheet uses `Pulse | Cycle_1 | Cycle_2 | ...`.")
    st.header("Analysis")
    c2c_input = st.number_input("Measured C2C (%)", min_value=0.0, value=0.0, step=0.1,
                                help="C2C is not automatically extracted. Enter your separately measured value.")
    run = st.button("Analyze LTP/LTD", type="primary", use_container_width=True)


def excel_bytes(metrics, ltp, ltd, fit_ltp, fit_ltd):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame({"Pulse": range(len(ltp)), "LTP_matched_S": ltp, "LTP_fit_S": fit_ltp,
                      "LTD_matched_S": ltd, "LTD_fit_S": fit_ltd}).to_excel(writer, sheet_name="Curve_fit", index=False)
    return output.getvalue()


if run:
    if uploaded is None:
        st.error("Upload an Excel file first.")
        st.stop()
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(uploaded.getvalue())
        temp_path = Path(temp.name)
    try:
        with st.spinner("Extracting SOM device metrics..."):
            ltp_cycles, ltd_cycles = load_curves(temp_path)
            metrics, ltp, ltd, fit_ltp, fit_ltd = analyze(ltp_cycles, ltd_cycles, c2c_input)
        st.success("Analysis complete")
        a, b, c, d = st.columns(4)
        a.metric("G ratio", f"{metrics['G_ratio']:.3f}")
        b.metric("Pulses", int(metrics["pulses_Pmax"]))
        c.metric("Ap / Pmax", f"{metrics['Ap_over_Pmax']:.4f}")
        d.metric("Ad / Pmax", f"{metrics['Ad_over_Pmax']:.4f}")
        st.subheader("Extracted device parameters")
        view = pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())})
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption(f"C2C = {metrics['C2C_percent_input']:.3f}% (user-entered, not auto-extracted)")
        st.subheader("LTP/LTD exponential fitting")
        fig, ax = plt.subplots(figsize=(9, 5))
        p = range(len(ltp))
        ax.plot(p, ltp, "o", ms=4, label="LTP measured / endpoint matched")
        ax.plot(p, fit_ltp, "-", lw=2, label="LTP Ap fit")
        ax.plot(p, ltd, "s", ms=4, label="LTD measured / endpoint matched")
        ax.plot(p, fit_ltd, "-", lw=2, label="LTD Ad fit")
        ax.set(xlabel="Pulse number", ylabel="Conductance (S)")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        png = BytesIO(); fig.savefig(png, format="png", dpi=300, bbox_inches="tight"); plt.close(fig)
        xlsx = excel_bytes(metrics, ltp, ltd, fit_ltp, fit_ltd)
        col1, col2 = st.columns(2)
        col1.download_button("Download results Excel", xlsx, "som_ltp_ltd_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        col2.download_button("Download fitting plot", png.getvalue(), "som_ltp_ltd_fit.png", "image/png")
    except Exception as exc:
        st.exception(exc)
    finally:
        temp_path.unlink(missing_ok=True)
else:
    st.subheader("Excel input format")
    st.code("""LTP sheet                         LTD sheet
Pulse | Cycle_1 | Cycle_2          Pulse | Cycle_1 | Cycle_2
0     | 3.64E-5 | ...              0     | 4.18E-5 | ...
1     | 3.75E-5 | ...              1     | 3.63E-5 | ...
...   | ...                         ...   | ...
""", language="text")
    st.write("LTP is entered low → high. LTD is entered high → low.")
