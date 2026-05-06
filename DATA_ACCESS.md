# Data attachment instructions

The data files are not included in this anonymous review archive because they cannot be redistributed with the submission. Raw CWS data can be obtained from Netatmo and Wunderground. OWS for Zurich was obtained from Anet et. al (2024).

To run the code, place the data on the review machine using the layout below and point the code to it with either `QC_DATA_ROOT` or `path_to_data.txt`.

```text
<DATA_ROOT>/
  project_id/
    config_project.py
    data/
      0_raw/
      0_metadata/
      1_structured/
      2_filtered/
    results/
      11_data_structure/
      20_quality_control/
```

Recommended setup:

```bash
export QC_DATA_ROOT=/path/to/external/data_root
export QC_PROJECT_ID=project_id
export QC_LOCAL_TIMEZONE=UTC
```

Alternative setup: copy `path_to_data.txt.example` to `path_to_data.txt` and replace the placeholder with the same external data-root path. Do not commit or submit the edited `path_to_data.txt` if it contains a private local path.

Copy `config_project_template.py` to `<DATA_ROOT>/project_id/config_project.py` and update the dates, project coordinates, and file names for the provided data. For the legacy preprocessing script, copy `scripts/preprocesing/projectdata_template.json`, edit it, and set `QC_PROJECTDATA_JSON` to that file.
