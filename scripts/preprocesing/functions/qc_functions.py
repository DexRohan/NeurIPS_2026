
"""Quality-control helper functions."""

import pandas as pd
import numpy as np


import scipy.stats as stats


from statsmodels.robust.scale import qn_scale as qn


from scipy.stats import t
from scipy.stats import norm


from geopy.distance import geodesic

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def QC_I1_datatype(data):
    """
    This function analyzes the data type in the sample.

    """
    values = data.count().sum()/data.size
    values = values.round(2)
    values = values*100
    columns = len(data.columns)

    print("")
    print("Shape: ", data.shape)
    print("Size: ", data.size, " - ", "non- nan values: ", data.count().sum())
    print("Percentage of non- nan values: ",values,"%")
    print("Columns: ",columns)
    print("Dtype: ", data.dtypes.value_counts())


def QC_I2_missingdata(data, cutoff):
    """
    This function excludes stations with more than 80% missing values to ensure data reliability and representativeness..

    """


    missing_percentage = data.isna().mean()

    print(missing_percentage.head())

    valid_stations = missing_percentage[missing_percentage <= cutoff].index


    data1 = data[valid_stations]


    return data1


def QC_I3_limittest(data):
    """
    This function eliminates data outside the following threshold: -40ºC and 60ºC.

    """

    data = data[((data > -40) & (data < 60 ))]

    return data


def QC_I4_slopetest(data):
    """
    This function eliminates data with a gradient higher than 20ºC/h.

    """

    slope = abs(data.diff(periods=1, axis=0))


    data = data[((slope < 20) | (pd.isna(slope)))]

    return data


def QC_I5_consistencytest(data):
    """
    This function eliminates data with a constant value longer than 6 hours.

    """

    slope = abs(data.diff(periods=1, axis=0))


    slope = ((slope < 0.005) | (pd.isna(slope)))


    slope1 = slope.rolling(6,min_periods=1).sum()


    for column in range(len(slope1.columns)):
        for i in range(6,len(slope1)):
            value = slope1.iloc[i,column]
            if value == 6.0:
                slope1.iloc[i-6:i+1,column] = np.nan


    slope2 = slope1.isna()

    data2 = data.mask(slope2==True,np.nan)

    return data2


def QC_G4_duplicate(data):
    """
    This function eliminates data with duplicate values or missing values according to id, long and lat.

    """
    data1 = data.copy()


    data1 = data1.replace('', np.nan)
    data1["a"] = data1[["module_final", "long", "lat"]].isnull().any(axis=1)


    data1["b"]= (data1["long"]==data1["lat"])


    data1["c"] = data1[["_id","module_final","long","lat"]].duplicated()


    data1["d"] = False


    data1["M1"] = data1["a"] | data1["b"] | data1["c"] | data1["d"]

    a = data1["a"].sum()
    b = data1["b"].sum()
    c = data1["c"].sum()
    d = data1["d"].sum()
    M1 = data1["M1"].sum()
    M1_perc = 1 - data1["M1"].sum() / data1["module_final"].count()

    d = {'Missing values': [a], 'Same values': [b], 'All duplicated': [c], 'Long-Lat duplicated': [d], 'M1': [M1], 'M1_perc': [M1_perc]}
    report_M1 = pd.DataFrame(data=d)

    print("Report of Flag values for step M1")
    print(report_M1)

    data1 = data1.drop(["a","b","c","d"], axis=1)

    return data1,report_M1


def apply_elevation_correction(weather_data, coordinates_data, heightcorrection=True, lapse_rate=0.0065,apply= True):
    """
    Apply or remove elevation-based temperature correction.

    Args:
        weather_data: DataFrame with temperature data (columns are station IDs/module_final)
        coordinates_data: DataFrame with columns 'module_final' and 'elev_meters'
        apply: If True, apply correction; if False, remove correction
        lapse_rate: Temperature change per meter (default 1°C per 100m = 1/100)

    Returns:
        Corrected weather DataFrame
    """

    mean_elev = coordinates_data["elev_meters"].mean()


    coordinates_data["elev_correction"] = (coordinates_data["elev_meters"] - mean_elev) * lapse_rate


    corrected_data = weather_data.copy()


    sign = 1 if apply else -1
    for column in corrected_data.columns:
        if column in coordinates_data["module_final"].values:
            correction_value = coordinates_data.loc[coordinates_data["module_final"] == column, "elev_correction"].values[0]
            if not np.isnan(correction_value):
                corrected_data[column] = corrected_data[column] + (sign * correction_value)

    return corrected_data



def QC_G5_distribution(weather_data,coordinates_data, heightcorrection, lapse_rate, z_score, t_distribution,low, high):
    """
    Normal distribution (or Student-t distribution if stations <100) is used to identify outliers:
    lower and upper ends of the distribution at each time step.

    z_score = alternatives:

    "z_score" = (t- mean(t))/std
    "mod_z_score_mad"=(t - median())/MAD
    "mod_z_score_qn"= (t - median())/Qn

    """


    if heightcorrection == True:
        weather_data1 = apply_elevation_correction(weather_data, coordinates_data,heightcorrection, lapse_rate, apply=True)
        print("Elevation correction applied to CWS temperature data")
    else:
        weather_data1 = weather_data.copy()
        print("Elevation correction not applied to CWS temperature data")



    weather_M2 = weather_data1

    weather_M2_Zscore = pd.DataFrame()


    if z_score=="z_score":

        for i in range(0,len(weather_M2)):
            row_data = weather_M2.iloc[i].dropna()
            if len(row_data) < 5:
                print(f"Row {i} has fewer than 5 valid values. Skipping.")
                appenddata = pd.DataFrame(index=weather_M2.iloc[[i]].index, columns=weather_M2.columns)
            else:
                appenddata = (weather_M2.iloc[[i]] - weather_M2.iloc[i].mean(skipna=True))/weather_M2.iloc[i].std(skipna=True,ddof=0)

            weather_M2_Zscore = pd.concat([weather_M2_Zscore,appenddata],axis=0)



    elif z_score=="mod_z_score_mad":

        for i in range(0,len(weather_M2)):
            row_data = weather_M2.iloc[i].dropna()
            if len(row_data) < 5:
                print(f"Row {i} has fewer than 5 valid values. Skipping.")
                appenddata = pd.DataFrame(index=weather_M2.iloc[[i]].index, columns=weather_M2.columns)
            else:
                appenddata = 0.6745*(weather_M2.iloc[[i]] - weather_M2.iloc[i].median(skipna=True))/weather_M2.iloc[i].mad(skipna=True)

            weather_M2_Zscore = pd.concat([weather_M2_Zscore,appenddata],axis=0)

    elif z_score=="mod_z_score_qn":



        for i in range(0,len(weather_M2)):
            row_data = weather_M2.iloc[i].dropna()


            if len(row_data) < 5:
                print(f"Row {i} has fewer than 5 valid values. Skipping.")
                appenddata = pd.DataFrame(index=weather_M2.iloc[[i]].index, columns=weather_M2.columns)
            else:
                appenddata = (weather_M2.iloc[[i]] - weather_M2.iloc[i].median(skipna=True))/qn(weather_M2.iloc[i].dropna())

            weather_M2_Zscore = pd.concat([weather_M2_Zscore,appenddata],axis=0)

    else:
        print("no match")

    weather_M2_Zscore_no_outliers = weather_M2_Zscore.copy()


    if t_distribution==True:

        for i in range(0,len(weather_M2_Zscore_no_outliers)):
            n = weather_M2_Zscore_no_outliers.iloc[i].count() -1
            weather_M2_Zscore_no_outliers.iloc[i] = weather_M2_Zscore_no_outliers.iloc[i].mask((weather_M2_Zscore_no_outliers.iloc[i]<t.ppf(low,df=n)) | (weather_M2_Zscore_no_outliers.iloc[i]>t.ppf(high,df=n)))
    elif t_distribution==False:

        for i in range(0,len(weather_M2_Zscore_no_outliers)):
            weather_M2_Zscore_no_outliers.iloc[i] = weather_M2_Zscore_no_outliers.iloc[i].mask((weather_M2_Zscore_no_outliers.iloc[i]<norm.ppf(low)) | (weather_M2_Zscore_no_outliers.iloc[i]>norm.ppf(high)))

    else:
        print("no match")


    weather_M2 = weather_M2[weather_M2_Zscore_no_outliers.notnull()==True]



    if heightcorrection == True:
        weather_M3  = apply_elevation_correction(weather_M2, coordinates_data, heightcorrection, lapse_rate, apply=False)
        print("Elevation correction removed from temperature data after outlier detection")
    else:
        weather_M3 = weather_M2.copy()

    return weather_M2_Zscore, weather_M2_Zscore_no_outliers,weather_M3


def QC_G5b_buddy_check(
    weather_data: pd.DataFrame,
    coordinates: pd.DataFrame,
    heightcorrection,
    lapse_rate,
    dist_threshold_m: float = 2000,
    alt_threshold_m: float = 100,
    min_buddies: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a buddy check on weather station data using a CrowdQC+ inspired method.

    For each station, it identifies nearby "buddies" based on distance and
    altitude. It then calculates a modified z-score for the station's readings
    based on the median and Qn scale of its buddies' readings. Readings with
    scores outside the 0.05 and 0.95 quantiles of a normal distribution
    are flagged as outliers and set to NaN.

    Args:
        weather_data: DataFrame with datetime index and station IDs as columns.
        coordinates: DataFrame with station metadata. Must include columns:
                     'module_final', 'lat', 'long', and 'elev'.
        dist_threshold_m: Maximum distance in meters to consider a station a buddy.
        alt_threshold_m: Maximum absolute elevation difference in meters
                           to consider a station a buddy.
        min_buddies: The minimum number of buddies required to perform the check.

    Returns:
        A tuple containing:
        - cleaned_data (pd.DataFrame): The weather data with outliers set to NaN.
        - outliers (pd.DataFrame): A DataFrame showing only the values that were removed.
    """

    lower_bound = norm.ppf(0.075)
    upper_bound = norm.ppf(0.925)



    if heightcorrection == True:
        weather_data1 = apply_elevation_correction(weather_data, coordinates,heightcorrection, lapse_rate, apply=True)
        print("Elevation correction applied to CWS temperature data")
    else:
        weather_data1 = weather_data.copy()
        print("Elevation correction not applied to CWS temperature data")


    required_cols = ['module_final', 'lat', 'long', 'elev_meters']
    if not all(col in coordinates.columns for col in required_cols):
        raise ValueError(f"Coordinates DataFrame must contain the following columns: {required_cols}")

    cleaned_data = weather_data1.copy()
    coords = coordinates.set_index('module_final')

    if not coordinates['module_final'].is_unique:
        dups = coordinates.loc[coordinates['module_final'].duplicated(), 'module_final'].unique()
        raise ValueError(f"Duplicate module_final in coordinates (showing up to 20): {dups[:20]}")

    all_stations = weather_data1.columns

    for station_j in tqdm(all_stations, desc="Buddy Check"):
        if station_j not in coords.index:
            warnings.warn(f"Station {station_j} not found in coordinates file. Skipping.")
            continue

        station_coords = (coords.loc[station_j, 'lat'], coords.loc[station_j, 'long'])
        station_elev = coords.loc[station_j, 'elev_meters']


        buddy_ids = []
        for buddy_candidate in all_stations:
            if station_j == buddy_candidate or buddy_candidate not in coords.index:
                continue

            candidate_coords = (coords.loc[buddy_candidate, 'lat'], coords.loc[buddy_candidate, 'long'])
            candidate_elev = coords.loc[buddy_candidate, 'elev_meters']

            distance = geodesic(station_coords, candidate_coords).meters
            alt_diff = abs(station_elev - candidate_elev)

            if distance <= dist_threshold_m and alt_diff <= alt_threshold_m:
                buddy_ids.append(buddy_candidate)


        if len(buddy_ids) < min_buddies:
            continue


        station_j_data = cleaned_data[station_j]
        buddy_data = weather_data1[buddy_ids]

        median_buddies = buddy_data.median(axis=1, skipna=True)


        qn_buddies = buddy_data.apply(lambda row: np.nan if row.notna().sum() < 2 else qn(row.dropna().to_numpy()), axis=1)

        qn_buddies.replace(0, np.nan, inplace=True)


        mod_z_score = (station_j_data - median_buddies) / qn_buddies


        outlier_mask = (mod_z_score < lower_bound) | (mod_z_score > upper_bound)

        cleaned_data.loc[outlier_mask, station_j] = np.nan


    if heightcorrection == True:
        cleaned_data2  = apply_elevation_correction(cleaned_data, coordinates, heightcorrection, lapse_rate, apply=False)
        print("Elevation correction removed from temperature data after outlier detection")
    else:
        cleaned_data2 = cleaned_data.copy()


    outliers = weather_data[weather_data != cleaned_data2]


    return cleaned_data2, outliers


def QC_G6_trust(weather_data, outliers, cutOff, time):

    """
    Flags and removes data from weather stations with >cutOff proportion of float values in the outliers DataFrame per time period.

    Parameters:
    -----------
    weather_data : pd.DataFrame
        A DataFrame with datetime index and stations as columns.
    outliers : pd.DataFrame
        A DataFrame with the same size and datetime as weather_data, containing float values eliminated from weather_data.
    cutOff : float
        Threshold for maximum allowed proportion of float values in outliers per time group (e.g., 0.25).
    time : str
        Pandas time frequency string to group by (e.g., 'M' for monthly).

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with suspicious station data removed.
    """

    cleaned_df = weather_data.copy()
    outliers1 = outliers.copy()


    for (group_time, group_df_weather), (_, group_df_outliers) in zip(
        cleaned_df.groupby(pd.Grouper(freq=time)),
        outliers1.groupby(pd.Grouper(freq=time))
    ):

        non_nan_fraction = group_df_outliers.notna().mean()


        stations_to_flag = non_nan_fraction[non_nan_fraction > cutOff].index
        print(f"Stations to flag: {stations_to_flag}")


        cleaned_df.loc[group_df_weather.index, stations_to_flag] = np.nan


    print(f"Columns before dropping NaN: {cleaned_df.shape[1]}")
    cleaned_df.dropna(axis=1, how='all', inplace=True)
    print(f"Columns after dropping NaN: {cleaned_df.shape[1]}")

    return cleaned_df


def QC_G7_tcorrelation(weather_data,cutOff):

    """Remove monthly station series whose Pearson correlation to the CWS median is below ``cutOff``."""

    weather_M4_help = weather_data.copy()


    weather_M4_help["month"] = weather_M4_help.index.month


    List_month = []
    List_month = list(weather_M4_help["month"].drop_duplicates().dropna())


    weather_M4_help = weather_M4_help.rename(columns={"month": "median"})
    weather_M4_help["median"] = np.nan


    for i in range(len(weather_M4_help)):
        weather_M4_help["median"].iloc[i] = weather_M4_help.drop(labels=["median"],axis=1).iloc[i].median(skipna=True)


    weather_M4 = pd.DataFrame()

    for i in List_month:
        weather_M4_help1 = pd.DataFrame()
        weather_M4_help1 = weather_M4_help[weather_M4_help.index.month == i].copy()
        correlation = abs(weather_M4_help1.corr(method="pearson"))

        if "median" in correlation.columns:
            correlation = correlation["median"].drop(labels=["median"], axis=0).to_frame()


        weather_M4_help1 = weather_M4_help1.drop(labels=["median"], axis=1)


        correlation1 = correlation[correlation["median"] < cutOff].reset_index()


        List_M4 = list(correlation1["index"].dropna())
        print("month", i, "-- Id to eliminate: ", List_M4)


        weather_M4_help1 = weather_M4_help1.drop(List_M4, axis=1)


        if not weather_M4_help1.empty:
            weather_M4 = pd.concat([weather_M4, weather_M4_help1])

    return weather_M4


def QC_G8_interpolation(data,threshold):
    """Interpolate short gaps up to the configured threshold."""



    CWS_UrbanClimate_all_QC_G81 = data.copy()


    valid_columns = CWS_UrbanClimate_all_QC_G81.columns[CWS_UrbanClimate_all_QC_G81.notna().sum() >= 2]
    invalid_columns = CWS_UrbanClimate_all_QC_G81.columns[CWS_UrbanClimate_all_QC_G81.notna().sum() < 2]


    if len(invalid_columns) > 0:
        print(f"Skipping interpolation for columns with fewer than 2 valid data points: {list(invalid_columns)}")


    CWS_UrbanClimate_all_QC_G81[valid_columns] = CWS_UrbanClimate_all_QC_G81[valid_columns].interpolate(
        method="cubicspline", limit=threshold, limit_area="inside"
    )


    for c in data:
        mask = data[c].isna()
        x = (mask.groupby((mask != mask.shift()).cumsum()).transform(lambda x: len(x) > threshold )* mask)
        CWS_UrbanClimate_all_QC_G81[c] = CWS_UrbanClimate_all_QC_G81.loc[~x, c]

    return CWS_UrbanClimate_all_QC_G81


def build_flagged_ta_outputs(
    raw_data: pd.DataFrame,
    cleaned_data_before_interp: pd.DataFrame,
    flag_suffix: str = "_is_outlier"
) -> tuple[pd.DataFrame, pd.DataFrame]:

    raw_data = raw_data.copy().sort_index()
    cleaned = cleaned_data_before_interp.copy()


    if raw_data.index.tz is not None:
        raw_data.index = raw_data.index.tz_convert(None)
    if cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_convert(None)


    raw_data.columns = raw_data.columns.astype(str)
    cleaned.columns = cleaned.columns.astype(str)


    cleaned_aligned = cleaned.reindex(
        index=raw_data.index,
        columns=raw_data.columns
    )


    outlier_mask = raw_data.notna() & cleaned_aligned.isna()
    flags_only = outlier_mask.astype("Int8")
    flag_columns = flags_only.rename(columns=lambda c: f"{c}{flag_suffix}")


    ordered_parts = []
    for col in raw_data.columns:
        ordered_parts.append(raw_data[[col]])
        ordered_parts.append(flag_columns[[f"{col}{flag_suffix}"]])

    flagged_ta = pd.concat(ordered_parts, axis=1)

    return flagged_ta, flags_only


