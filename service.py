from itertools import repeat
import multiprocessing
import time
import numpy as np

from py.anomaly_detection.anomaly_detector import detect_anomalies, mark_missing_data
from py.plot import plot_result
from py.sensor_finder import get_sensor_list
from py.export_events import export_results
from py.imputer import (
    impute_small_gaps,
    impute_big_gaps,
)
from py.preparer import (
    get_period_data,
    to_period_df,
    define_nights,
    separate_workdays_and_weekends,
)


def process(sensor, args):
    begin = time.time()

    print(f'{sensor["name"]}: Loading data')
    ts_df = to_period_df(sensor["source"], args)

    period_data = get_period_data(ts_df)

    print(f'{sensor["name"]}: Defining nights')
    ts_df = define_nights(ts_df)

    print(f'{sensor["name"]}: Separating data to workdays and weekends')
    (
        workdays_df,
        tumbled_workdays_df,
        weekends_df,
        tumbled_weekends_df,
    ) = separate_workdays_and_weekends(ts_df[["data"]], period_data)
    tumbled_days_df = tumbled_workdays_df.append(tumbled_weekends_df).sort_index()

    print(f'{sensor["name"]}: Starting to impute small gaps')
    ts_df = impute_small_gaps(
        ts_df,
        period_data,
        workdays_df,
        tumbled_workdays_df,
        weekends_df,
        tumbled_weekends_df,
    )

    # Why call separate func again? to have tumbled dataframes with imputed small gaps
    (
        workdays_df,
        tumbled_workdays_df,
        weekends_df,
        tumbled_weekends_df,
    ) = separate_workdays_and_weekends(ts_df[["imputed_data"]], period_data)
    tumbled_days_with_small_gaps_fixed_df = tumbled_workdays_df.append(
        tumbled_weekends_df
    ).sort_index()

    print(f'{sensor["name"]}: Marking missing data')
    ts_df = mark_missing_data(
        ts_df, period_data, tumbled_days_df, tumbled_days_with_small_gaps_fixed_df
    )

    print(f'{sensor["name"]}: Starting Anomaly detection')
    ts_df = detect_anomalies(
        ts_df, period_data, tumbled_workdays_df, tumbled_weekends_df, sensor, args
    )

    if args["impute_data_gaps"]:
        print(f'{sensor["name"]}: Starting to impute big gaps')
        ts_df = impute_big_gaps(ts_df)

    if args["plot"]["create_plot"]:
        plot_result(ts_df, sensor, args)

    if (
        args["upload_to_c8y"]["upload_anomaly_data"]
        or args["upload_to_c8y"]["upload_imputation"]
    ):
        batch_size = args["upload_to_c8y"]["batch_upload_size"] or 10000
        split_ts_df = np.array_split(ts_df, range(batch_size, len(ts_df), batch_size))
        for i, split in enumerate(split_ts_df):
            print(
                f'{sensor["name"]}: Starting export for batch {i+1}/{len(split_ts_df)}'
            )
            export_results(sensor, split, args)

    end = time.time()
    print(f'{sensor["name"]}: Total runtime for sensor is {end - begin}')

    return sensor


def service(args):
    anyDataPersistenceSelected = (
        args["plot"]["create_plot"]
        or args["upload_to_c8y"]["upload_anomaly_data"]
        or args["upload_to_c8y"]["upload_imputation"]
    )
    if not (anyDataPersistenceSelected):
        raise AttributeError("No data persistence was requested. Exiting.")

    if args["upload_to_c8y"]["upload_imputation"] and not args["impute_data_gaps"]:
        raise AttributeError(
            "Imputed data can not be upload when 'impute_data_gaps' is false. Exiting."
        )

    begin = time.time()

    sensor_list = get_sensor_list(args)

    parallel_batch_size = args["devices_processed_in_parallel"] or 4
    sensor_list = np.array_split(
        sensor_list, range(parallel_batch_size, len(sensor_list), parallel_batch_size)
    )

    for sensors in sensor_list:
        with multiprocessing.Pool(processes=len(sensors)) as pool:
            for sensor in pool.starmap(process, zip(sensors, repeat(args))):
                print("Finished Imputing", sensor)

    end = time.time()
    print(f"Total runtime of the AD service is {end - begin}")
