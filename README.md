# SOM LTP/LTD Analyzer

Browser application for extracting device parameters from SOM LTP/LTD conductance data.

## What it does

- Upload an Excel workbook containing LTP and LTD conductance curves.
- Automatically extract endpoint-matched `Gmin`, `Gmax`, G ratio, pulse count, and the exponential-fit parameters `Ap` and `Ad`.
- Enter measured `C2C (%)` manually. C2C is deliberately **not** automatically inferred from the input curves.
- View the fitted curves in the browser and download a results Excel workbook and PNG plot.

## Excel layout

Use two sheets named `LTP` and `LTD`.

| Pulse | Cycle_1 | Cycle_2 |
| --- | --- | --- |
| 0 | conductance | conductance |
| 1 | conductance | conductance |

LTP must be ordered from low to high conductance, and LTD from high to low conductance. A single LTP and LTD column is sufficient.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy with Streamlit Community Cloud

1. Push these files to a public GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, choose the repository and branch, then set the main file to `streamlit_app.py`.
4. Click **Deploy**.

Uploaded Excel data is processed temporarily and is not committed to the repository.
