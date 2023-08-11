from service import service
import configparser

config = configparser.RawConfigParser()
config.read("config/config.properties")

source_auth = dict(config.items("TARTU_AUTH"))
target_auth = dict(config.items("AUTH"))

args = {
    "source_auth": {
        "username": source_auth["username"],
        "password": source_auth["password"],
        "tenant_id": source_auth["tenant_id"],
        "base_url": source_auth["base_url"],
    },
    "target_auth": {
        "username": target_auth["username"],
        "password": target_auth["password"],
        "tenant_id": target_auth["tenant_id"],
        "base_url": target_auth["base_url"],
    },
    "devices_processed_in_parallel": 4,
    "data_loading": {
        "start_date": "2019-06-01",
        "end_date": "2023-06-01",
        "directory": "data/files",
        "data_type": "EVENT",
        # "data_type": "MEASUREMENT",
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        # only frequencies divisible by hour allowed: 1H, 30T, 20T, 15T, 12T etc
        # relevant for "data_type": "MEASUREMENT"
        # "measurement_frequency": "1H",
        "devices": [
            {"source": "123v1", "target": "123v2"},
            {"source": "456v1", "target": "456v2"},
        ],
    },
    "detect_anomalies": {
        "use_existing_model": False,
        "directory": "models",
    },
    "impute_data_gaps": True,
    "plot": {
        "create_plot": True,
        "directory": "plots",
    },
    "upload_to_c8y": {
        "upload_anomaly_data": True,
        "upload_imputation": True,
        "batch_upload_size": 10000,
        "c8y_measurement_type": "c8y_VehiclesMeasurement",
    },
}

if __name__ == "__main__":
    service(args)
