import numpy as np

from c8y_api import CumulocityRestApi
from c8y_api.model import Measurement, Measurements


def export_results(sensor, ts_df, args):
    # data is standardized in cumulocity as UTC
    # revert index back to UTC
    ts_df.index = ts_df.index.tz_convert("UTC")

    target_auth = args["target_auth"]
    target_connection = CumulocityRestApi(
        username=target_auth["username"],
        password=target_auth["password"],
        tenant_id=target_auth["tenant_id"],
        base_url=target_auth["base_url"],
    )

    target_sensor = sensor["target"]
    # Multiple measurement
    measurement_array = []
    print(f'{sensor["name"]}: Preparing time series upload')
    for index, row in ts_df.iterrows():
        if args["upload_to_c8y"]["upload_anomaly_data"]:
            c8y_Original = {
                "Amount": {
                    "unit": "n",
                    "value": int(np.round(row["data"]))
                    if not np.isnan(row["data"])
                    else 0,
                },
                "Anomaly Label": {
                    "unit": "n",
                    "value": int(row["anomaly_label"])
                    if not np.isnan(row["anomaly_label"])
                    else 0,
                },
                "Anomaly Score": {
                    "unit": "n",
                    "value": float(row["anomaly_score"])
                    if not np.isnan(row["anomaly_score"])
                    else 0,
                },
                "Sensor error or no observations": {
                    "unit": "n",
                    "value": int(row["sensor_error"]),
                },
            }

            measurement_array.append(
                Measurement(
                    c8y=target_connection,
                    type=args["upload_to_c8y"]["c8y_measurement_type"],
                    source=target_sensor,
                    time=index.isoformat(),
                    c8y_Original=c8y_Original,
                )
            )
        if args["upload_to_c8y"]["upload_imputation"]:
            measurement_array.append(
                Measurement(
                    c8y=target_connection,
                    type=args["upload_to_c8y"]["c8y_measurement_type"],
                    source=target_sensor,
                    time=index.isoformat(),
                    c8y_Imputed={
                        "Amount": {
                            "unit": "n",
                            "value": int(np.round(row["imputed_data"])),
                        }
                    },
                    c8y_ImputedData={},
                )
            )

    print(f'{sensor["name"]}: Uploading time series')
    # NB! Splat operator compulsory!
    Measurements(c8y=target_connection).create(*measurement_array)

    print(f'{sensor["name"]}: Resources sent')
