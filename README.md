## Dependencies
```
pip install c8y-api==1.8.2
pip install pyod==1.1.0
pip install pandas==1.3.5 
pip install statsmodels==0.13.5
pip install plotly==5.15.0
pip install urllib3==1.26.6
```
## Run
```
python main.py
```
## Configuration
Modify the args dictionary for configuration options

- `source_auth` - credentials for the Cumulocity environment where data was - read from
- `target_auth` - credentials for the Cumulocity environment where data is uploaded to
- `devices_processed_in_parallel` - number of devices that will be processed simultaneously (performance configuration)
- `data_loading`
  - `start_date` - processing start date
  - `end_date` - processing end date
  - `directory` - directory of the time series' CSV files
  - `data_type` - `MEASUREMENT` if time series is regular, `EVENT` if time series is irregular
  - `measurement_frequency` -  relevant for `data_type` - `MEASUREMENT`
    -  only frequencies divisible by hour allowed: 1H, 30T, 20T, 15T, 12T etc
  - `devices`
    - `source` - id of the device where data was read from
    - `target` - id of the device where data is uploaded to
- `detect_anomalies` 
  - `use_existing_model` - whether an existing model should be used for anomaly detection
  - `directory` - directory of the model used for anomaly detection
- `impute_data_gaps` - whether data gaps should be imputed
- `plot`
  - `create_plot` - whether the results should be plotted
  - `directory` - directory of plots
- `upload_to_c8y`
  - `upload_anomaly_data` - whether anomaly detection results should be uploaded
  - `upload_imputation` - whether imputation results should be uploaded
  - `batch_upload_size` - number of batches the results will be divided to before upload (performance configuration)
  - `c8y_measurement_type` - Cumulocity measurement type