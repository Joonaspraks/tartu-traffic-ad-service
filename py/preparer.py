import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import glob
import datetime

friday = 4
saturday = 5
sunday = 6
march = 3
october = 10


nanos_in_1_hour = 3600 * 1_000_000_000
nanos_in_1_day = 24 * 3600 * 1_000_000_000


def to_period_df(sensor_source, args):
    data_loading_args = args["data_loading"]
    start_date = data_loading_args["start_date"]
    end_date = data_loading_args["end_date"]
    input_directory = data_loading_args["directory"]
    data_type = data_loading_args["data_type"]

    print("Reading csvs")
    filenames = glob.glob(f"{input_directory}/{sensor_source}/*.csv")

    if len(filenames) == 0:
        raise ValueError(
            f"No .csv files were provided in folder {input_directory}. Exiting"
        )

    if data_type == "EVENT":
        event_series = pd.Series()
        for filename in filenames:
            print(filename)
            event_series = event_series.append(pd.read_csv(filename)["time"])

        if event_series.empty:
            raise ValueError("Provided files contained no rows. Exiting.")

        events = pd.unique(event_series)

        print("Converting data to time series")
        dti = pd.to_datetime(events, utc=True).tz_convert("Europe/Helsinki")
        time_series = pd.DataFrame(0, index=dti, columns=["data"]).sort_index()

    elif data_type == "MEASUREMENT":
        measurement_df = pd.DataFrame()
        for filename in filenames:
            print(filename)
            measurement_df = measurement_df.append(pd.read_csv(filename))

        if measurement_df.empty:
            raise ValueError("Provided files contained no rows. Exiting.")

        measurement_df = measurement_df.drop_duplicates(subset=["time"])

        print("Converting data to time series")
        dti = pd.to_datetime(measurement_df["time"].values, utc=True).tz_convert(
            "Europe/Helsinki"
        )
        time_series = pd.DataFrame(
            measurement_df["value"].values, index=dti, columns=["data"]
        ).sort_index()

    start_date = pd.to_datetime(start_date).tz_localize("Europe/Helsinki")
    end_date = pd.to_datetime(end_date).tz_localize("Europe/Helsinki")

    is_start_date_smaller_than_data = start_date < time_series.index[0]
    is_end_date_greater_than_data = end_date > time_series.index[-1]

    # dummy data will guarantee that the created daily data is uniform
    if is_start_date_smaller_than_data:
        time_series = time_series.append(
            pd.DataFrame(0, index=[start_date], columns=["data"])
        )

    if is_end_date_greater_than_data:
        time_series = time_series.append(
            pd.DataFrame(0, index=[end_date], columns=["data"])
        )

    if data_type == "EVENT":
        # ts_df = pd.DataFrame(time_series.resample("15Min").count(), columns=["data"])
        ts_df = time_series.resample("15Min").count()
    elif data_type == "MEASUREMENT":
        ts_df = time_series.resample(data_loading_args["measurement_frequency"]).sum()

    # use only the range specified by the user
    ts_df = ts_df[start_date : (end_date - pd.Timedelta(seconds=1))]

    ts_df = ts_df.replace(0, np.nan)

    return ts_df


def get_period_data(ts_df):
    periods_in_1_hour = int(nanos_in_1_hour / ts_df.index.freq.nanos)
    periods_in_1_day = int(nanos_in_1_day / ts_df.index.freq.nanos)

    return {
        "periods_in_1_hour": periods_in_1_hour,
        "periods_in_1_day": periods_in_1_day,
    }


def define_nights(ts_df):
    # it is plausible that nightly 15 min periods see 0 vehicles passing
    ts_df["data"][
        ((is_before_6_00(ts_df.index)) | (is_after_23_00(ts_df.index)))
        & (np.isnan(ts_df["data"]))
    ] = 0

    return ts_df


def separate_workdays_and_weekends(ts_df, period_data):
    # group by days
    df_by_day = ts_df.resample("D")

    # separate workdays and weekends
    workdays_df = pd.DataFrame()
    weekends_df = pd.DataFrame()
    for day, group in df_by_day:
        if day.weekday() <= friday:
            workdays_df = workdays_df.append(group)
        else:
            weekends_df = weekends_df.append(group)

    workdays_values = workdays_df.values

    # append or remove an hour in respect to daylight savings day
    # why only weekends? because this always happens on a Sunday
    weekends_values = np.array([])
    periods_in_1_hour = period_data["periods_in_1_hour"]
    for index, row in weekends_df.iterrows():
        if is_last_sunday_of_october(index) and is_4_00(index):
            # Reduce 2 hours into 1
            arr_length = len(weekends_values)
            last_2_hours = weekends_values[
                arr_length - 2 * periods_in_1_hour : arr_length
            ]
            weekends_values = np.delete(
                weekends_values, range(arr_length - 2 * periods_in_1_hour, arr_length)
            )

            last_1_hour = last_2_hours.reshape(2, periods_in_1_hour).sum(axis=0)
            weekends_values = np.append(weekends_values, last_1_hour)

        elif is_last_sunday_of_march(index) and is_2_00(index):
            # Add 1 dummy hour
            weekends_values = np.append(
                weekends_values, np.repeat(0, periods_in_1_hour)
            )

        weekends_values = np.append(weekends_values, row)

    periods_in_1_day = period_data["periods_in_1_day"]

    amount_of_windows = len(workdays_values) // periods_in_1_day
    tumbled_workdays = workdays_values.reshape((amount_of_windows, periods_in_1_day))
    amount_of_windows = len(weekends_values) // periods_in_1_day
    tumbled_weekends = weekends_values.reshape((amount_of_windows, periods_in_1_day))

    columns = [f"value{period+1}" for period in range(periods_in_1_day)]

    tumbled_workdays_df = pd.DataFrame()
    if not workdays_df.empty:
        index = workdays_df.groupby(workdays_df.index.date).groups
        tumbled_workdays_df = pd.DataFrame(
            tumbled_workdays, index=index, columns=columns
        )

    tumbled_weekends_df = pd.DataFrame()
    if not weekends_df.empty:
        index = weekends_df.groupby(weekends_df.index.date).groups
        tumbled_weekends_df = pd.DataFrame(
            tumbled_weekends, index=index, columns=columns
        )

    return workdays_df, tumbled_workdays_df, weekends_df, tumbled_weekends_df


def is_last_sunday_of_march(date):
    if date.month == march and date.weekday() == sunday:
        is_last_sunday = (date + datetime.timedelta(days=7)).month != march
        return is_last_sunday
    return False


def is_last_sunday_of_october(date):
    if date.month == october and date.weekday() == sunday:
        is_last_sunday = (date + datetime.timedelta(days=7)).month != october
        return is_last_sunday
    return False


def is_2_00(date):
    if date.hour == 2 and date.minute == 0:
        return True
    return False


def is_4_00(date):
    if date.hour == 4 and date.minute == 0:
        return True
    return False


def is_before_6_00(date):
    return date.hour < 6


def is_after_23_00(date):
    return (date.hour == 23) & (date.minute != 0)
