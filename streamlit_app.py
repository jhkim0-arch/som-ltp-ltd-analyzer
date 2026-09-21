"""Browser app for SOM parameter extraction and NeuroSim-style MNIST simulation."""
from io import BytesIO
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from som_ltp_ltd_analyzer import analyze, load_curves
from som_mnist_simulator import run_mnist_simulation

st.set_page_config(page_title="SOM Analyzer and MNIST", page_icon="⚡", layout="wide")
st.title("SOM LTP/LTD Analyzer and MNIST Simulator")
st.caption("Raw LTP/LTD Excel → device parameters → NeuroSim 3.0 MLP-style MNIST simulation")

with st.sidebar:
    st.header("Input data")
    uploaded = st.file_uploader("Raw Excel file", type=["xlsx", "xls"])
    st.info("Use separate `LTP` and `LTD` sheets: `Pulse | Cycle_1 | Cycle_2 | ...`.")
    st.header("Device input")
    c2c_input = st.number_input("Measured C2C (%)", min_value=0.0, value=0.0, step=0.1,
                                help="Enter separately measured C2C. It is not automatically extracted.")


def read_upload(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Upload an Excel file first.")
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = Path(temp.name)
    try:
        return load_curves(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def results_excel(metrics, ltp, ltd, fit_ltp, fit_ltd):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame({"Pulse": range(len(ltp)), "LTP_matched_S": ltp, "LTP_fit_S": fit_ltp,
                      "LTD_matched_S": ltd, "LTD_fit_S": fit_ltd}).to_excel(writer, sheet_name="Curve_fit", index=False)
    return output.getvalue()


analysis_tab, mnist_tab = st.tabs(["LTP/LTD Analysis", "MNIST Simulation"])

with analysis_tab:
    st.subheader("Device parameter extraction")
    st.write("LTP must be low → high conductance. LTD must be high → low conductance.")
    if st.button("Analyze LTP/LTD", type="primary"):
        try:
            with st.spinner("Extracting device parameters..."):
                ltp_cycles, ltd_cycles = read_upload(uploaded)
                metrics, ltp, ltd, fit_ltp, fit_ltd = analyze(ltp_cycles, ltd_cycles, c2c_input)
            a, b, c, d = st.columns(4)
            a.metric("G ratio", f"{metrics['G_ratio']:.3f}")
            b.metric("Pulses", int(metrics["pulses_Pmax"]))
            c.metric("Ap / Pmax", f"{metrics['Ap_over_Pmax']:.4f}")
            d.metric("Ad / Pmax", f"{metrics['Ad_over_Pmax']:.4f}")
            st.caption(f"C2C = {metrics['C2C_percent_input']:.3f}% (user-entered)")
            st.dataframe(pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}), use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(9, 5))
            p = range(len(ltp))
            ax.plot(p, ltp, "o", ms=4, label="LTP measured / endpoint matched")
            ax.plot(p, fit_ltp, "-", lw=2, label="LTP Ap fit")
            ax.plot(p, ltd, "s", ms=4, label="LTD measured / endpoint matched")
            ax.plot(p, fit_ltd, "-", lw=2, label="LTD Ad fit")
            ax.set(xlabel="Pulse number", ylabel="Conductance (S)"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            png = BytesIO(); fig.savefig(png, format="png", dpi=300, bbox_inches="tight"); plt.close(fig)
            left, right = st.columns(2)
            left.download_button("Download results Excel", results_excel(metrics, ltp, ltd, fit_ltp, fit_ltd), "som_ltp_ltd_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            right.download_button("Download fitting plot", png.getvalue(), "som_ltp_ltd_fit.png", "image/png")
        except Exception as exc:
            st.exception(exc)
    else:
        st.code("""LTP sheet                 LTD sheet
Pulse | Cycle_1          Pulse | Cycle_1
0     | 2.00E-6          0     | 6.00E-5
1     | 1.27E-5          1     | 5.13E-5
...                       ...
""", language="text")

with mnist_tab:
    st.subheader("MNIST simulation")
    st.caption("NeuroSim 3.0 MLP-style 20×20–100–10 network with differential-pair conductance updates.")
    x1, x2, x3 = st.columns(3)
    with x1:
        mode = st.selectbox("Device mode", ["Measured SOM", "Ideal"])
        epochs = st.slider("Epochs", 1, 20, 5)
    with x2:
        learning_rate = st.number_input("Learning rate", 0.0001, 0.2, 0.01, 0.005, format="%.4f")
        batch_size = st.selectbox("Batch size", [64, 128, 256], index=1)
    with x3:
        train_limit = st.selectbox("Training samples", [5000, 10000, 20000, 60000], index=1)
        test_limit = st.selectbox("Test samples", [1000, 2000, 5000, 10000], index=1)
    st.info("Ideal: G ratio 50, 64 states. Measured SOM: uploaded raw LTP/LTD after endpoint matching; pulses = points − 1.")
    if st.button("Run MNIST simulation", type="primary"):
        try:
            ltp_cycles, ltd_cycles = read_upload(uploaded)
            status, log = st.empty(), st.empty()
            bar = st.progress(0); epoch_log = []

            def progress(epoch, accuracy):
                epoch_log.append({"Epoch": epoch, "Test accuracy (%)": accuracy})
                bar.progress(int(100 * epoch / epochs))
                status.write(f"Epoch {epoch}/{epochs}: test accuracy = {accuracy:.3f}%")
                log.dataframe(pd.DataFrame(epoch_log), use_container_width=True, hide_index=True)

            with st.spinner("Downloading MNIST if needed, then training the device-aware MLP..."):
                history, matrix, metric = run_mnist_simulation(
                    ltp_cycles.mean(axis=0), ltd_cycles.mean(axis=0), c2c_input, mode, epochs,
                    batch_size, learning_rate, train_limit, test_limit, progress=progress,
                )
            bar.progress(100); status.success(f"Completed: final test accuracy = {history[-1]:.3f}%")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Device", metric.mode); m2.metric("G ratio", f"{metric.g_ratio:.3f}")
            m3.metric("Pulses", metric.pulses); m4.metric("C2C", f"{metric.c2c_percent:.3f}%")
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(range(len(history)), history, "o-", color="#1f77b4")
            ax1.set(xlabel="Epoch", ylabel="Test accuracy (%)", ylim=(0, 100), title="MNIST accuracy vs epoch")
            ax1.grid(alpha=.25); fig1.tight_layout()
            fig2, ax2 = plt.subplots(figsize=(6, 5)); image = ax2.imshow(matrix, cmap="Blues")
            for row in range(10):
                for col in range(10):
                    ax2.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8,
                             color="white" if matrix[row, col] > matrix.max() * .55 else "black")
            ax2.set(xticks=range(10), yticks=range(10), xlabel="Predicted label", ylabel="True label",
                    title=f"Confusion matrix (accuracy {history[-1]:.2f}%)")
            fig2.colorbar(image, ax=ax2, label="Samples"); fig2.tight_layout()
            left, right = st.columns(2)
            with left: st.pyplot(fig1, use_container_width=True)
            with right: st.pyplot(fig2, use_container_width=True)
            history_csv = pd.DataFrame({"Epoch": range(len(history)), "Test accuracy (%)": history}).to_csv(index=False).encode()
            matrix_csv = pd.DataFrame(matrix, index=[f"True_{i}" for i in range(10)], columns=[f"Pred_{i}" for i in range(10)]).to_csv().encode()
            d1, d2 = st.columns(2)
            d1.download_button("Download accuracy CSV", history_csv, "accuracy_vs_epoch.csv", "text/csv")
            d2.download_button("Download confusion matrix CSV", matrix_csv, "confusion_matrix.csv", "text/csv")
            plt.close(fig1); plt.close(fig2)
        except Exception as exc:
            st.exception(exc)
