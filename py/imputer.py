import calendar
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from .preparer import (
    is_last_sunday_of_october,
    is_last_sunday_of_march,
    separate_workdays_and_weekends,
)


def impute_small_gaps(
    ts_df,
    period_data,
    workdays_df,
    tumbled_workdays_df,
    weekends_df,
    tumbled_weekends_df,
):
    mask = create_mask(ts_df, period_data["periods_in_1_hour"])

    workdays_df = extract_seasonality(
        period_data, tumbled_workdays_df, workdays_df, "workdays"
    )
    weekends_df = extract_seasonality(
        period_data, tumbled_weekends_df, weekends_df, "weekends"
    )

    days_df = workdays_df.append(weekends_df).sort_index()

    interpolated_values = (
        pd.Series(days_df["data"] - days_df["seasonality"]).interpolate()
        + days_df["seasonality"]
    )

    interpolated_values[interpolated_values < 0] = 0

    ts_df["imputed_data"] = interpolated_values[pd.Series(mask)]
    ts_df["imputed_data"] = round(ts_df["imputed_data"])
    return ts_df


def extract_seasonality(period_data, tumbled_days_df, days_df, type):
    # impute up to hour long gaps with seasonal decomposition + linear interpolation
    # Why? Simple linear interpolation does not know about the 7 and 16 oclock rush hours
    # Thus, it could cause artificial anomalies

    # use only days without nans to compute seasonality

    periods_in_1_hour = period_data["periods_in_1_hour"]
    periods_in_1_day = period_data["periods_in_1_day"]

    idx_of_no_nans = np.all(~np.isnan(tumbled_days_df), axis=1)
    non_nans = tumbled_days_df[idx_of_no_nans].values

    if len(non_nans) < 2:
        print(f"Not enough data for seasonal decomposition on {type}")
        print("Atleast two gapless days are needed")
        print("Using regular linear interpolation instead")
        days_df["seasonality"] = 0
        return days_df

    decomposition = seasonal_decompose(
        non_nans.flatten(), model="additive", period=periods_in_1_day
    )

    regular_daily_season = decomposition.seasonal[0:periods_in_1_day]
    # insert after 2:45
    long_daily_season = np.insert(
        regular_daily_season, 3 * periods_in_1_hour, np.repeat(0, periods_in_1_hour)
    )
    # delete after 2:45
    short_daily_season = np.delete(
        regular_daily_season, np.arange(3 * periods_in_1_hour, 4 * periods_in_1_hour)
    )

    seasonality = np.array([])
    for index in tumbled_days_df.index:
        if is_last_sunday_of_march(index):
            seasonality = np.append(seasonality, short_daily_season)
        elif is_last_sunday_of_october(index):
            seasonality = np.append(seasonality, long_daily_season)
        else:
            seasonality = np.append(seasonality, regular_daily_season)

    days_df["seasonality"] = seasonality

    return days_df


def impute_big_gaps(ts_df):
    # impute gaps with values from previous and upcoming weekdays

    # create temporary "imputed_data" series where anomalous days are removed
    # that way we won't base our imputed data on the anomalous data
    temp_df = ts_df.filter(
        ["imputed_data", "anomaly_label", "big_sensor_error"], axis=1
    )
    is_anomalous = temp_df["anomaly_label"] == 1
    temp_df["imputed_data"][is_anomalous] = np.nan
    # why not use the sensor_error label on ts_df? because we are ok with using the data that has small gaps
    is_sensor_error = temp_df["big_sensor_error"] == 1
    temp_df["imputed_data"][is_sensor_error] = np.nan

    df_by_time_and_weekday = temp_df.groupby(
        [temp_df.index.time, temp_df.index.weekday]
    )
    imputed_df_by_week = pd.DataFrame()
    for _, group in df_by_time_and_weekday:
        if group["imputed_data"].count() == 0:
            weekday = calendar.day_name[group.index[0].date().weekday()]
            raise ValueError(
                f"Not enough data to impute {weekday}s. Either you did not upload multiple {weekday}s or the data quality of every {weekday} is too low to be used for imputation"
            )
        group["imputed_data"] = group["imputed_data"].interpolate(
            limit_direction="both"
        )
        imputed_df_by_week = imputed_df_by_week.append(group)
        
    imputed_df_by_week = imputed_df_by_week.sort_index()

    (
        workdays_df,
        _tumbled_workdays_df,
        weekends_df,
        _tumbled_weekends_df,
    ) = separate_workdays_and_weekends(temp_df[["imputed_data"]])

    df_by_time = workdays_df.groupby(
        [workdays_df.index.time]
    )
    imputed_df_by_workday = pd.DataFrame()
    for _, group in df_by_time:
        group["imputed_data"] = group["imputed_data"].interpolate(
            limit_direction="both"
        )
        imputed_df_by_workday = imputed_df_by_workday.append(group)

    df_by_time = weekends_df.groupby(
        [weekends_df.index.time]
    )
    imputed_df_by_weekend = pd.DataFrame()
    for _, group in df_by_time:
        group["imputed_data"] = group["imputed_data"].interpolate(
            limit_direction="both"
        )
        imputed_df_by_weekend = imputed_df_by_weekend.append(group)

    imputed_df_by_day = imputed_df_by_workday.append(
        imputed_df_by_weekend
    ).sort_index()

    # take a (weighted) average of daily and weekly imputation

    imputed_df = (0.6*imputed_df_by_week + 0.4*imputed_df_by_day)

    # add anomalous days back to the data
    imputed_df["imputed_data"][is_anomalous] = ts_df["imputed_data"][is_anomalous]
    # missing sensor error data should also be added back
    is_not_nan = ~np.isnan(ts_df["imputed_data"][is_sensor_error])
    imputed_df["imputed_data"][is_sensor_error & is_not_nan] = ts_df["imputed_data"][
        is_sensor_error & is_not_nan
    ]

    imputed_df = imputed_df.sort_index()

    ts_df["imputed_data"] = np.ceil(imputed_df["imputed_data"])

    return ts_df


def create_mask(ts_df, max_gap):
    # https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe

    # group by change from non-nans to nans and vice versa
    continous_sequences = (
        (ts_df.notnull() != ts_df.shift().notnull()).cumsum().groupby("data")
    )

    # mark sequence as false if its too long
    mask = np.array([], dtype=bool)
    for _, sequence in continous_sequences:
        continuous_values_length = len(sequence)
        is_small_gap = continuous_values_length <= max_gap
        mask = np.append(mask, np.repeat(is_small_gap, continuous_values_length))
    mask = mask | ts_df["data"].notnull()

    return mask
