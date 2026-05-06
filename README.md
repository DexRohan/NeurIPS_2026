# Anonymous NeurIPS code supplement

This archive contains the code and notebook runners for the CWS/OWS temperature quality-control experiments. The dataset files are not included because the underlying data cannot be redistributed.

## Data attachment

The dataset files are not included because the underlying data cannot be redistributed with this anonymous review archive. See `DATA_ACCESS.md` for the exact layout and setup options.

Minimal environment-variable setup:

```bash
export QC_DATA_ROOT=/path/to/external/data_root
export QC_PROJECT_ID=project_id
export QC_LOCAL_TIMEZONE=UTC
```

Alternatively, copy `path_to_data.txt.example` to `path_to_data.txt` and edit it locally. The edited file should not be submitted if it contains a private path.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Parquet input/output requires `pyarrow` or another pandas-compatible Parquet backend.

## Running the workflow

The notebook runners are in `notebooks/`. The main stages are:

1. `11_DataStructure_DateTime.ipynb`
2. `20_DataPre-processing_qc.ipynb`
3. `20_ML_OWS_Reference_Model.ipynb`
4. `22_CWS_Residual_Risk_Model.ipynb`
5. `23_CWS_QC_Risk_Fusion.ipynb`
6. `24_CWS_Reviewer_Robustness_Tests_1.ipynb` and `24_CWS_Reviewer_Robustness_Tests_2.ipynb`
7. `25_CWS_Spatial_Support_Audit.ipynb`

The scripts can also be imported directly from `scripts/preprocesing/` and `scripts/timestructure/`. The code defaults to relative paths or environment variables; no local machine paths are required.