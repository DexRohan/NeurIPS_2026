
"""Datetime structure utilities."""


import pandas as pd
import numpy as np
import json

import datetime
from datetime import datetime, timedelta
from dateutil import tz

def st_datatype(data):
    """
    This function analyzes the data type in the sample.

    """
    values = data.count().sum()/data.size
    values = values.round(2)
    values = values*100
    columns = len(data.columns)

    print("")
    print("Shape: ", data.shape)
    print("Size: ", data.size, " - ", "number of nan values: ", data.size-data.count().sum())
    print("Percentage of valid values: ",values,"%")
    print("Columns: ",columns)
    print("Dtype: ", data.dtypes.value_counts())



def st_datatime(dataset1, datetime_structure, delta_direction, delta_time, local_timezone):

    print("processing dataset - time structure correction")

    dataset= dataset1.copy()


    if datetime_structure == 'local':
        from_zone = tz.gettz(local_timezone)
        to_zone = tz.gettz('UTC')
        dataset.index = dataset.index.tz_localize(from_zone, nonexistent='shift_forward', ambiguous=True).tz_convert(to_zone)
    elif datetime_structure == 'UTC':
        from_zone = tz.gettz('UTC')
        dataset.index = dataset.index.tz_localize(from_zone)
    else:
        raise ValueError("Invalid input for datetime structure. Please enter 'UTC' or 'local'.")


    if delta_direction == 'fordward':


        dataset_shifted = dataset.copy()

        dataset_shifted.index = dataset_shifted.index - pd.Timedelta(minutes=delta_time)


        dataset_shifted1 = dataset_shifted.resample('30min').mean()

        na_mask = dataset_shifted1.isna()



        gap_1_mask = na_mask & ~na_mask.shift(1, fill_value=False) & ~na_mask.shift(-1, fill_value=False)


        temp_data = dataset_shifted1.copy()
        temp_data[~gap_1_mask] = temp_data[~gap_1_mask].ffill()


        interpolated = temp_data.interpolate(method='time', limit=1)


        interpolated[~gap_1_mask] = dataset_shifted1[~gap_1_mask]


        dataset = interpolated


        dataset = dataset.resample('1h').first()

        print("Time index is shifted fordward in time")

    if delta_direction == 'backward':


        dataset_shifted = dataset.copy()

        dataset_shifted.index = dataset_shifted.index + pd.Timedelta(minutes=delta_time)


        dataset_shifted1 = dataset_shifted.resample('30min').mean()

        na_mask = dataset_shifted1.isna()



        gap_1_mask = na_mask & ~na_mask.shift(1, fill_value=False) & ~na_mask.shift(-1, fill_value=False)


        temp_data = dataset_shifted1.copy()
        temp_data[~gap_1_mask] = temp_data[~gap_1_mask].ffill()


        interpolated = temp_data.interpolate(method='time', limit=1)


        interpolated[~gap_1_mask] = dataset_shifted1[~gap_1_mask]


        dataset = interpolated


        dataset = dataset.resample('1h').first()

        print("Time index is shifted backward in time")

    elif delta_direction == False:

        print("Time index is not shifted fordward or backward in time")

    return dataset
