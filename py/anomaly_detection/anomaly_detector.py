from sklearn.preprocessing import MinMaxScaler
from ..preparer import is_last_sunday_of_march, is_last_sunday_of_october
from .lof import lof
import numpy as np
import pandas as pd

friday = 4
saturday = 5
sunday = 6
march = 3
october = 10


def mark_missing_data(
    ts_df, period_data, tumbled_days_df, tumbled_days_with_small_gaps_fixed_df
):
    idx_of_nans = np.any(np.isnan(tumbled_days_df), axis=1)
    tumbled_days_df["sensor_error"] = 0
    tumbled_days_df.loc[idx_of_nans, "sensor_error"] = 1

    idx_of_nans = np.any(np.isnan(tumbled_days_with_small_gaps_fixed_df), axis=1)
    tumbled_days_df["big_sensor_error"] = 0
    tumbled_days_df.loc[idx_of_nans, "big_sensor_error"] = 1

    periods_in_1_hour = period_data["periods_in_1_hour"]
    periods_in_1_day = period_data["periods_in_1_day"]
    periods_in_1_short_day = periods_in_1_day - periods_in_1_hour
    periods_in_1_long_day = periods_in_1_day + periods_in_1_hour

    # expand labels
    expanded_sensor_errors = np.array([])
    expanded_big_sensor_errors = np.array([])
    for index, day in tumbled_days_df.iterrows():
        if is_last_sunday_of_march(index):
            expanded_sensor_errors = np.append(
                expanded_sensor_errors,
                np.repeat(day["sensor_error"], periods_in_1_short_day),
            )
            expanded_big_sensor_errors = np.append(
                expanded_big_sensor_errors,
                np.repeat(day["big_sensor_error"], periods_in_1_short_day),
            )
        elif is_last_sunday_of_october(index):
            expanded_sensor_errors = np.append(
                expanded_sensor_errors,
                np.repeat(day["sensor_error"], periods_in_1_long_day),
            )
            expanded_big_sensor_errors = np.append(
                expanded_big_sensor_errors,
                np.repeat(day["big_sensor_error"], periods_in_1_long_day),
            )
        else:
            expanded_sensor_errors = np.append(
                expanded_sensor_errors, np.repeat(day["sensor_error"], periods_in_1_day)
            )
            expanded_big_sensor_errors = np.append(
                expanded_big_sensor_errors,
                np.repeat(day["big_sensor_error"], periods_in_1_day),
            )

    ts_df["sensor_error"] = expanded_sensor_errors
    ts_df["big_sensor_error"] = expanded_big_sensor_errors
    return ts_df


def detect_anomalies(
    ts_df, period_data, tumbled_workdays_df, tumbled_weekends_df, sensor, args
):
    tumbled_workdays_df = detect_anomalies_in_days(
        tumbled_workdays_df, sensor, args, "workdays"
    )

    tumbled_weekends_df = detect_anomalies_in_days(
        tumbled_weekends_df, sensor, args, "weekends"
    )

    # combine lof scores
    tumbled_days_df = tumbled_workdays_df.append(tumbled_weekends_df).sort_index()

    if "anomaly_score" not in tumbled_days_df.columns:
        print(
            "None of the provided days had a sufficient data quality for Anomaly Detection"
        )

        ts_df["anomaly_score"] = np.nan
        ts_df["anomaly_label"] = np.nan

        return ts_df

    # expand scores and labels
    expanded_scores = np.array([])
    expanded_labels = np.array([])

    periods_in_1_hour = period_data["periods_in_1_hour"]
    periods_in_1_day = period_data["periods_in_1_day"]
    periods_in_1_short_day = periods_in_1_day - periods_in_1_hour
    periods_in_1_long_day = periods_in_1_day + periods_in_1_hour
    for index, day in tumbled_days_df.iterrows():
        if is_last_sunday_of_march(index):
            expanded_scores = np.append(
                expanded_scores, np.repeat(day["anomaly_score"], periods_in_1_short_day)
            )
            expanded_labels = np.append(
                expanded_labels, np.repeat(day["anomaly_label"], periods_in_1_short_day)
            )
        elif is_last_sunday_of_october(index):
            expanded_scores = np.append(
                expanded_scores, np.repeat(day["anomaly_score"], periods_in_1_long_day)
            )
            expanded_labels = np.append(
                expanded_labels, np.repeat(day["anomaly_label"], periods_in_1_long_day)
            )
        else:
            expanded_scores = np.append(
                expanded_scores, np.repeat(day["anomaly_score"], periods_in_1_day)
            )
            expanded_labels = np.append(
                expanded_labels, np.repeat(day["anomaly_label"], periods_in_1_day)
            )

    ts_df["anomaly_score"] = expanded_scores
    ts_df["anomaly_label"] = expanded_labels

    return ts_df


def detect_anomalies_in_days(tumbled_days_df, sensor, args, data_type):
    use_existing_model = args["detect_anomalies"]["use_existing_model"]

    idx_of_no_nans = np.all(~np.isnan(tumbled_days_df), axis=1)
    non_nans = tumbled_days_df[idx_of_no_nans]

    if len(non_nans) > 1 or (use_existing_model and len(non_nans) != 0):
        tumbled_days_df = pd.DataFrame(
            MinMaxScaler().fit_transform(tumbled_days_df),
            columns=tumbled_days_df.columns,
            index=tumbled_days_df.index,
        )
        scores, labels = lof(non_nans, sensor, args, data_type)

        tumbled_days_df["anomaly_score"] = np.nan
        tumbled_days_df["anomaly_label"] = np.nan

        tumbled_days_df["anomaly_score"][np.where(idx_of_no_nans)[0]] = scores
        tumbled_days_df["anomaly_label"][np.where(idx_of_no_nans)[0]] = labels

    return tumbled_days_df
