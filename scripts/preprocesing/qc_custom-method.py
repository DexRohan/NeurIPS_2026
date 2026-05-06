
"""Custom quality-control preprocessing workflow."""


import os
import sys
import pandas as pd
import datetime
from datetime import datetime, timedelta
from dateutil import tz

import pytz
import tzlocal

import numpy as np
import json


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
plt.style.use('default')
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


import scipy.stats as stats


from statsmodels.robust.scale import qn_scale as qn


from scipy.stats import t
from scipy.stats import norm


from geopy.distance import geodesic


plt.rcParams['axes.xmargin'] = 0


cwd = os.path.dirname(__file__)
os.chdir(cwd)


from functions import qc_functions as qc


projectdata_path = os.environ.get('QC_PROJECTDATA_JSON', 'projectdata.json')
if not os.path.exists(projectdata_path):
    raise FileNotFoundError(
        'Missing projectdata.json. Copy projectdata_template.json, update the paths to your non-redistributed data, '
        'or set QC_PROJECTDATA_JSON to the configuration file.'
    )
with open(projectdata_path, 'r', encoding='utf-8') as fp:
    data = json.load(fp)


city = data["city"]
lat = data["lat"]
long = data["long"]



first_date=data["first_date"]
last_date=data["last_date"]
year1 = datetime.strptime(first_date, '%d-%m-%Y %H:%M').year


color_net = data["color_net"]
color_wund =data["color_wund"]
color_ows =data["color_ows"]
color_cws = data["color_cws"]
color_outliers = data["color_outliers"]

cwd_data_str = data["cwd_data_str"]
cwd_data_meta = data["cwd_data_meta"]
cwd_data_qc = data["cwd_data_qc"]
cwd_data_qc_results = data["cwd_data_qc_results"]

name_coordinates_wunder = data["name_coordinates_wunder"]
name_ta_wunder = data["name_ta_wunder"]

name_coordinates_net = data["name_coordinates_net"]
name_ta_net = data["name_ta_net"]

name_coordinates_ows = data["name_coordinates_ows"]
name_ta_ows = data["name_ta_ows"]


print("QC-I1 ,",qc.QC_I1_datatype.__doc__)

print("QC-I2 ,",qc.QC_I2_missingdata.__doc__)

print("QC-I3 ,",qc.QC_I3_limittest.__doc__)

print("QC-I4 ,",qc.QC_I4_slopetest.__doc__)

print("QC-I5 ,",qc.QC_I5_consistencytest.__doc__)

print("QC-G4 ,",qc.QC_G4_duplicate.__doc__)

print("QC-G5 ,",qc.QC_G5_distribution.__doc__)

print("QC-G6 ,",qc.QC_G6_trust.__doc__)

print("QC-G7 ,",qc.QC_G7_tcorrelation.__doc__)

print("QC-G8 ,",qc.QC_G8_interpolation.__doc__)


os.chdir(cwd_data_meta)


CWS_coordinates_net = pd.read_csv(name_coordinates_net)
CWS_coordinates_wund = pd.read_csv(name_coordinates_wunder)


OWS_coordinates = pd.read_csv(name_coordinates_ows)


os.chdir(cwd_data_str)



CWS_UrbanClimate_net = pd.read_csv(name_ta_net, index_col='date',parse_dates=True)


CWS_UrbanClimate_wund = pd.read_csv(name_ta_wunder, index_col='date',parse_dates=True)


OWS_Climate = pd.read_csv(name_ta_ows, index_col='date',parse_dates=True)
OWS_Climate = OWS_Climate.apply(pd.to_numeric, errors='coerce')
OWS_Climate.index = pd.to_datetime(OWS_Climate.index, dayfirst=True)

if OWS_Climate.index.tz is not None:
    OWS_Climate.index = OWS_Climate.index.tz_convert(None)
else:

    pass


CWS_coordinates_net["type"] = "CWS - Citizen Weather Stations - Netatmo"
CWS_coordinates_wund["type"] = "CWS - Citizen Weather Stations - Wunderground"
OWS_coordinates["type"] = "OWS - Official Weather Stations"

print("All files have been correctly charged from : ",cwd_data_str)


CWS_UrbanClimate_net_raw_h = CWS_UrbanClimate_net.copy()
CWS_UrbanClimate_wund_raw_h = CWS_UrbanClimate_wund.copy()


if CWS_UrbanClimate_net_raw_h.index.tz is not None:
    CWS_UrbanClimate_net_raw_h.index = CWS_UrbanClimate_net_raw_h.index.tz_convert(None)

if CWS_UrbanClimate_wund_raw_h.index.tz is not None:
    CWS_UrbanClimate_wund_raw_h.index = CWS_UrbanClimate_wund_raw_h.index.tz_convert(None)


CWS_UrbanClimate_net_raw_h = CWS_UrbanClimate_net_raw_h.resample("h").first()
CWS_UrbanClimate_wund_raw_h = CWS_UrbanClimate_wund_raw_h.resample("h").first()


CWS_UrbanClimate_all_raw_h = pd.concat(
    [CWS_UrbanClimate_net_raw_h, CWS_UrbanClimate_wund_raw_h],
    axis=1
).sort_index()


def create_and_save_qc_plot(
    filepath: str,
    title: str,
    data_to_plot: dict,
    ylim: tuple = (-40, 60),
    xlim: tuple | None = None,
    major_locator=mdates.MonthLocator(interval=1)
):
    fig, ax = plt.subplots(figsize=(14, 3))

    for _, plot_args in data_to_plot.items():
        obj = plot_args["data"]

        if obj.empty:
            continue

        if isinstance(obj, pd.Series):
            x = pd.to_datetime(obj.index, utc=True).tz_convert(None)
            y = pd.to_numeric(obj, errors="coerce")

            ax.plot(
                x, y,
                color=plot_args["color"],
                alpha=plot_args["alpha"],
                linewidth=plot_args["linewidth"],
                label=plot_args.get("label"),
                rasterized=plot_args.get("rasterized", False)
            )

        elif isinstance(obj, pd.DataFrame):
            df = obj.copy()
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            df = df.apply(pd.to_numeric, errors="coerce")

            first = True
            for col in df.columns:
                ax.plot(
                    df.index, df[col],
                    color=plot_args["color"],
                    alpha=plot_args["alpha"],
                    linewidth=plot_args["linewidth"],
                    label=plot_args.get("label") if first else None,
                    rasterized=plot_args.get("rasterized", False)
                )
                first = False

        else:
            raise TypeError(f"Unsupported plot data type: {type(obj)}")

    ax.set(xlabel="Time", ylabel="Temperature (ºC)", title=title, ylim=ylim)

    if xlim is not None:
        left = pd.Timestamp(xlim[0])
        right = pd.Timestamp(xlim[1])
        if left.tz is not None:
            left = left.tz_convert(None)
        if right.tz is not None:
            right = right.tz_convert(None)
        ax.set_xlim(left, right)

    handles, labels = ax.get_legend_handles_labels()
    if any(lbl not in (None, "") for lbl in labels):
        ax.legend(loc=1)

    ax.xaxis.set_major_formatter(DateFormatter("%y-%m-%d"))
    ax.xaxis.set_major_locator(major_locator)

    plt.savefig(filepath, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {filepath}")


qc.QC_I1_datatype(CWS_UrbanClimate_net)
qc.QC_I1_datatype(CWS_UrbanClimate_wund)
qc.QC_I1_datatype(OWS_Climate)


qc.QC_I1_datatype(CWS_coordinates_net)
qc.QC_I1_datatype(CWS_coordinates_wund)
qc.QC_I1_datatype(OWS_coordinates)


print(OWS_coordinates.info())
print(OWS_Climate.info())
print("QC_I1 done")


before=first_date
after=last_date


plot_config_fig1 = {

    "netatmo_background": {
        'data': CWS_UrbanClimate_net, 'color': color_net, 'alpha': 0.4,
        'linewidth': 0.6, 'rasterized': True, 'label': None
    },
    "wunderground_background": {
        'data': CWS_UrbanClimate_wund, 'color': color_wund, 'alpha': 0.6,
        'linewidth': 0.6, 'rasterized': True, 'label': None
    },
    "ows_background": {
        'data': OWS_Climate, 'color': color_ows, 'alpha': 0.8,
        'linewidth': 0.6, 'rasterized': True, 'label': None
    },

    "netatmo_line": {
        'data': CWS_UrbanClimate_net.iloc[:, 0], 'color': color_net, 'alpha': 0.8,
        'linewidth': 1.5, 'label': "CWS - Netatmo"
    },
    "wunderground_line": {
        'data': CWS_UrbanClimate_wund.iloc[:, 0], 'color': color_wund, 'alpha': 0.8,
        'linewidth': 1.5, 'label': "CWS - Wunderground"
    },
    "ows_line": {
        'data': OWS_Climate.iloc[:, 0], 'color': color_ows, 'alpha': 1,
        'linewidth': 1.5, 'label': "Official weather stations"
    }
}
output_path_fig1 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure1.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig1,
    title="Raw CWS data",
    data_to_plot=plot_config_fig1
)
print("QC_I2 done")


CWS_UrbanClimate_net_QC_I3 = qc.QC_I3_limittest(CWS_UrbanClimate_net)
CWS_UrbanClimate_wund_QC_I3 = qc.QC_I3_limittest(CWS_UrbanClimate_wund)
OWS_Climate_QC_I3 = qc.QC_I3_limittest(OWS_Climate)


Outliers_net_I3 = CWS_UrbanClimate_net[(CWS_UrbanClimate_net - CWS_UrbanClimate_net_QC_I3 != 0)]
Outliers_wund_I3 = CWS_UrbanClimate_wund[(CWS_UrbanClimate_wund - CWS_UrbanClimate_wund_QC_I3 != 0)]
Outliers_OWS_I3 = OWS_Climate[(OWS_Climate - OWS_Climate_QC_I3 != 0)]



print("Outlier net: ",Outliers_net_I3.count().sum())
print("Outlier wund: ",Outliers_wund_I3.count().sum())
print("Outlier OWS: ",Outliers_OWS_I3.count().sum())

print("QC_I3 done")


CWS_UrbanClimate_net_QC_I4 = qc.QC_I4_slopetest(CWS_UrbanClimate_net_QC_I3)
CWS_UrbanClimate_wund_QC_I4 = qc.QC_I4_slopetest(CWS_UrbanClimate_wund_QC_I3)
OWS_Climate_QC_I4 = qc.QC_I4_slopetest(OWS_Climate_QC_I3)


Outliers_net_I4 = CWS_UrbanClimate_net_QC_I3[(CWS_UrbanClimate_net_QC_I3-CWS_UrbanClimate_net_QC_I4!=0)]
Outliers_wund_I4 = CWS_UrbanClimate_wund_QC_I3[(CWS_UrbanClimate_wund_QC_I3-CWS_UrbanClimate_wund_QC_I4!=0)]
Outliers_OWS_I4 = OWS_Climate_QC_I3[(OWS_Climate_QC_I3-OWS_Climate_QC_I4!=0)]


print("Outlier net: ",Outliers_net_I4.count().sum())
print("Outlier wund: ",Outliers_wund_I4.count().sum())
print("Outlier OWS: ",Outliers_OWS_I4.count().sum())

print("QC_I4 done")


CWS_UrbanClimate_net_QC_I5 = qc.QC_I5_consistencytest(CWS_UrbanClimate_net_QC_I4)
CWS_UrbanClimate_wund_QC_I5 = qc.QC_I5_consistencytest(CWS_UrbanClimate_wund_QC_I4)
OWS_Climate_QC_I5 = qc.QC_I5_consistencytest(OWS_Climate_QC_I4)


Outliers_net_I5 = CWS_UrbanClimate_net_QC_I4[(CWS_UrbanClimate_net_QC_I4-CWS_UrbanClimate_net_QC_I5!=0)]
Outliers_wund_I5 = CWS_UrbanClimate_wund_QC_I4[(CWS_UrbanClimate_wund_QC_I4-CWS_UrbanClimate_wund_QC_I5!=0)]
Outliers_OWS_I5 = OWS_Climate_QC_I4[(OWS_Climate_QC_I4-OWS_Climate_QC_I5!=0)]


print("Outlier net: ",Outliers_net_I5.count().sum())
print("Outlier wund: ",Outliers_wund_I5.count().sum())
print("Outlier OWS: ",Outliers_OWS_I5.count().sum())

print("QC_I5 done")


Outliers_net = CWS_UrbanClimate_net[(CWS_UrbanClimate_net-CWS_UrbanClimate_net_QC_I5!=0)]
Outliers_wund = CWS_UrbanClimate_wund[(CWS_UrbanClimate_wund-CWS_UrbanClimate_wund_QC_I5!=0)]
Outliers_OWS = OWS_Climate[(OWS_Climate-OWS_Climate_QC_I5!=0)]


plot_config_fig2 = {
    "netatmo_clean_bg": {'data': CWS_UrbanClimate_net_QC_I5, 'color': color_net, 'alpha': 0.4, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "wunderground_clean_bg": {'data': CWS_UrbanClimate_wund_QC_I5, 'color': color_wund, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "ows_clean_bg": {'data': OWS_Climate_QC_I5, 'color': color_ows, 'alpha': 0.8, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_bg": {'data': Outliers_net, 'color': color_outliers, 'alpha': 1, 'linewidth': 1, 'rasterized': True, 'label': None},

    "outliers_line": {'data': Outliers_net.iloc[:, 0], 'color': color_outliers, 'alpha': 0.8, 'linewidth': 1.5, 'label': "Outliers"},
    "netatmo_line": {'data': CWS_UrbanClimate_net_QC_I5.iloc[:, 0], 'color': color_net, 'alpha': 0.8, 'linewidth': 1.5, 'label': "CWS - Netatmo"},
    "wunderground_line": {'data': CWS_UrbanClimate_wund_QC_I5.iloc[:, 0], 'color': color_wund, 'alpha': 0.8, 'linewidth': 1.5, 'label': "CWS - Wunderground"},
    "ows_line": {'data': OWS_Climate_QC_I5.iloc[:, 0], 'color': color_ows, 'alpha': 1, 'linewidth': 1, 'label': "OWS"},
}
output_path_fig2 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure2.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig2,
    title="CWS data after individual QC procedures",
    data_to_plot=plot_config_fig2
)


print("Outlier CWS - Netatmo: ",Outliers_net.resample("YE").count().sum(axis=1).values)
print("Outlier CWS - Wunderground: ",Outliers_wund.resample("YE").count().sum(axis=1).values)
print("Outlier OWS: ",Outliers_OWS.resample("YE").count().sum(axis=1).values)

print("QC - Individual analysis finished")


plot_config_fig3 = {
    "netatmo_bg": {'data': CWS_UrbanClimate_net_QC_I5, 'color': color_net, 'alpha': 0.4, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "wunderground_bg": {'data': CWS_UrbanClimate_wund_QC_I5, 'color': color_wund, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "ows_bg": {'data': OWS_Climate_QC_I5, 'color': color_ows, 'alpha': 0.8, 'linewidth': 0.6, 'rasterized': True, 'label': None},

    "netatmo_line": {'data': CWS_UrbanClimate_net_QC_I5.iloc[:, 0], 'color': color_net, 'alpha': 0.8, 'linewidth': 1.5, 'label': "CWS - Netatmo"},
    "wunderground_line": {'data': CWS_UrbanClimate_wund_QC_I5.iloc[:, 0], 'color': color_wund, 'alpha': 0.8, 'linewidth': 1.5, 'label': "CWS - Wunderground"},
    "ows_line": {'data': OWS_Climate_QC_I5.iloc[:, 0], 'color': color_ows, 'alpha': 1, 'linewidth': 1.5, 'label': "Official weather stations"},
}
output_path_fig3 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure3.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig3,
    title="CWS data after individual QC procedures",
    data_to_plot=plot_config_fig3
)



print("QC_G1 done")


CWS_UrbanClimate_wund_QC_G2 = CWS_UrbanClimate_wund_QC_I5.copy()


CWS_UrbanClimate_net_QC_I5 = CWS_UrbanClimate_net_QC_I5.resample('h').first()
CWS_UrbanClimate_wund_QC_G2 = CWS_UrbanClimate_wund_QC_G2.resample('h').first()


CWS_UrbanClimate_all_QC_G2 = pd.concat([CWS_UrbanClimate_net_QC_I5,CWS_UrbanClimate_wund_QC_G2],axis=1)


CWS_coordinates_all = pd.concat([CWS_coordinates_net,CWS_coordinates_wund],axis=0).reset_index().drop("index",axis=1)


List_existingCWS = CWS_UrbanClimate_all_QC_G2.columns.values.tolist()


CWS_coordinates_all = CWS_coordinates_all[
    CWS_coordinates_all['module_final'].isin(List_existingCWS)
]

print("QC_G2 done - from now, CWSs are combined")


print("Check manually inconsistent data")

print(CWS_UrbanClimate_all_QC_G2.info())
print(OWS_Climate_QC_I5.info())
print(CWS_coordinates_all.info())
print(OWS_coordinates.info())


print("QC_G3 done")


data_G4,report_G4 = qc.QC_G4_duplicate(CWS_coordinates_all)


data_G4["list"] = data_G4.module_final[(data_G4["M1"]==True)]

List_G4 = []
List_G4 = list(data_G4["list"].dropna())


CWS_UrbanClimate_all_QC_G4 = CWS_UrbanClimate_all_QC_G2.drop(List_G4, axis=1)

CWS_coordinates_all_G4 = CWS_coordinates_all.loc[~CWS_coordinates_all['module_final'].isin(List_G4)].reset_index(drop=True)

print("QC_G4 done")


weather_Zscore, weather_Zscore_no_outliers, CWS_UrbanClimate_all_QC_G5 = qc.QC_G5_distribution(CWS_UrbanClimate_all_QC_G4,CWS_coordinates_all_G4,heightcorrection=True, lapse_rate=0.0065,z_score= "mod_z_score_qn", t_distribution=False,low = 0.1, high = 0.9)


Outliers_CWS_G5 = CWS_UrbanClimate_all_QC_G4[(CWS_UrbanClimate_all_QC_G4-CWS_UrbanClimate_all_QC_G5!=0)]


print(weather_Zscore.describe())


from pytz import UTC

initial_date3 = pd.Timestamp(f'{year1}-08-01', tz=UTC)
final_date3 = pd.Timestamp(f'{year1}-08-31', tz=UTC)



fig, ax = plt.subplots()
num_bins = 100
weather_Zscore.iloc[0].hist(ax=ax, bins=num_bins, alpha=0.75, range=[-10, 10], color="grey", label="Raw z-scores")
weather_Zscore_no_outliers.iloc[0].hist(ax=ax, bins=num_bins, alpha=0.75, range=[-8, 8], color=color_cws, label="Scores after QC_G5")
plt.legend(loc="upper left")
plt.ylabel("Frequency")
plt.xlabel("Modified Z-Score")
plt.title("Histogram of Z-Scores (First Timestep)")
output_path_fig7 = os.path.join(cwd_data_qc_results, f"02_Pre-processing_{city}_figure7.jpg")
plt.savefig(output_path_fig7, format='jpg', dpi=150, bbox_inches='tight')
plt.close(fig)


initial_date3 = pd.Timestamp(f'{year1}-08-01', tz='UTC')
final_date3 = pd.Timestamp(f'{year1}-08-31', tz='UTC')

plot_config_fig8 = {
    "cws_clean_bg": {'data': CWS_UrbanClimate_all_QC_G5, 'color': color_cws, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_bg": {'data': Outliers_CWS_G5, 'color': color_outliers, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_line": {'data': Outliers_CWS_G5.iloc[:, 0], 'color': color_outliers, 'alpha': 0.8, 'linewidth': 1.5, 'label': "Outliers"},
}
output_path_fig8 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure8.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig8,
    title="CWS data after QC_G5 (August)",
    data_to_plot=plot_config_fig8,
    ylim=(5, 45),
    xlim=(initial_date3, final_date3),
    major_locator=mdates.DayLocator(interval=5)
)
print("QC_G5 done")


print("Synchronizing coordinates for buddy check...")
list_existing_stations_after_g5 = CWS_UrbanClimate_all_QC_G5.columns.tolist()
coordinates_for_buddy_check = CWS_coordinates_all[
    CWS_coordinates_all['module_final'].isin(list_existing_stations_after_g5)
]


CWS_UrbanClimate_all_QC_G5b, Outliers_CWS_G5b = qc.QC_G5b_buddy_check(
    weather_data=CWS_UrbanClimate_all_QC_G5,
    coordinates=coordinates_for_buddy_check,
    heightcorrection=True, lapse_rate=0.0065
)

print(f"Buddy check completed. Identified {Outliers_CWS_G5b.count().sum()} new outliers.")
print("QC_G5b done")


Outliers1 = CWS_UrbanClimate_all_QC_G2[(CWS_UrbanClimate_all_QC_G2-CWS_UrbanClimate_all_QC_G5b!=0)]


CWS_UrbanClimate_all_QC_G6 = qc.QC_G6_trust(CWS_UrbanClimate_all_QC_G5b,Outliers1,cutOff=0.20,time='ME')


Outliers_CWS_G6 = CWS_UrbanClimate_all_QC_G5b[(CWS_UrbanClimate_all_QC_G5b-CWS_UrbanClimate_all_QC_G6!=0)]


plot_config_fig9 = {
    "cws_clean_bg": {'data': CWS_UrbanClimate_all_QC_G6, 'color': color_cws, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_bg": {'data': Outliers_CWS_G6, 'color': color_outliers, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_line": {'data': Outliers_CWS_G6.iloc[:, 0], 'color': color_outliers, 'alpha': 0.8, 'linewidth': 1.5, 'label': "Outliers"},
}
output_path_fig9 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure9.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig9,
    title="CWS data after QC_G6 (August)",
    data_to_plot=plot_config_fig9,
    ylim=(5, 45),
    xlim=(initial_date3, final_date3),
    major_locator=mdates.DayLocator(interval=5)
)


print("QC_G6 done")


CWS_UrbanClimate_all_QC_G7 = qc.QC_G7_tcorrelation(CWS_UrbanClimate_all_QC_G6,cutOff=0.9).sort_index()


Outliers_CWS_G7 = CWS_UrbanClimate_all_QC_G6[(CWS_UrbanClimate_all_QC_G6-CWS_UrbanClimate_all_QC_G7!=0)]


plot_config_fig10 = {
    "cws_clean_bg": {'data': CWS_UrbanClimate_all_QC_G7, 'color': color_cws, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_bg": {'data': Outliers_CWS_G7, 'color': color_outliers, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_line": {'data': Outliers_CWS_G7.iloc[:, 0], 'color': color_outliers, 'alpha': 0.8, 'linewidth': 1.5, 'label': "Outliers"},
}
output_path_fig10 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure10.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig10,
    title="CWS data after QC_G7 (August)",
    data_to_plot=plot_config_fig10,
    ylim=(5, 45),
    xlim=(initial_date3, final_date3),
    major_locator=mdates.DayLocator(interval=5)
)


print("QC_G7 done")


CWS_UrbanClimate_all_QC_G8 = qc.QC_G8_interpolation(CWS_UrbanClimate_all_QC_G7,1)



print("QC_G8 done - End of group QC steps")


CWS_UrbanClimate_all_QC_flagged, CWS_outlier_flags_only = qc.build_flagged_ta_outputs(
    raw_data=CWS_UrbanClimate_all_raw_h,
    cleaned_data_before_interp=CWS_UrbanClimate_all_QC_G7,
    flag_suffix="_is_outlier"
)


CWS_coordinates_all_flagged = pd.concat(
    [CWS_coordinates_net, CWS_coordinates_wund],
    axis=0
).reset_index(drop=True)

CWS_coordinates_all_flagged = CWS_coordinates_all_flagged[
    CWS_coordinates_all_flagged["module_final"].isin(CWS_UrbanClimate_all_raw_h.columns)
].copy()

print("Flagged TA outputs created")
print("Total flagged outliers:", int(CWS_outlier_flags_only.sum().sum()))


Outliers_G = CWS_UrbanClimate_all_QC_G2[(CWS_UrbanClimate_all_QC_G2-CWS_UrbanClimate_all_QC_G8!=0)]
CWS_UrbanClimate_all_QC = CWS_UrbanClimate_all_QC_G8.copy()
OWS_Climate_QC = OWS_Climate_QC_I5.copy()


plot_config_fig11 = {
    "cws_clean_bg": {'data': CWS_UrbanClimate_all_QC, 'color': color_cws, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "ows_clean_bg": {'data': OWS_Climate_QC, 'color': color_ows, 'alpha': 0.8, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "outliers_bg": {'data': Outliers_G, 'color': color_outliers, 'alpha': 0.4, 'linewidth': 0.6, 'rasterized': True, 'label': None},

    "outliers_line": {'data': Outliers_G.iloc[:, 0], 'color': color_outliers, 'alpha': 0.8, 'linewidth': 1.5, 'label': "Outliers"},
    "cws_line": {'data': CWS_UrbanClimate_all_QC.iloc[:, 0], 'color': color_cws, 'alpha': 0.8, 'linewidth': 1.5, 'label': "CWS"},
    "ows_line": {'data': OWS_Climate_QC.iloc[:, 0], 'color': color_ows, 'alpha': 1, 'linewidth': 1.5, 'label': "OWS"},
}
output_path_fig11 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure11.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig11,
    title="CWS data after group QC procedures",
    data_to_plot=plot_config_fig11
)


plot_config_fig12 = {
    "cws_clean_bg": {'data': CWS_UrbanClimate_all_QC, 'color': color_cws, 'alpha': 0.6, 'linewidth': 0.6, 'rasterized': True, 'label': None},
    "ows_clean_bg": {'data': OWS_Climate_QC, 'color': color_ows, 'alpha': 0.8, 'linewidth': 0.6, 'rasterized': True, 'label': None},

    "cws_line": {'data': CWS_UrbanClimate_all_QC.iloc[:, 0], 'color': color_cws, 'alpha': 0.8, 'linewidth': 1.5, 'label': "CWS"},
    "ows_line": {'data': OWS_Climate_QC.iloc[:, 0], 'color': color_ows, 'alpha': 1, 'linewidth': 1.5, 'label': "OWS"},
}
output_path_fig12 = os.path.join(cwd_data_qc_results, f"02-Pre-processing_{city}_figure12.jpg")
create_and_save_qc_plot(
    filepath=output_path_fig12,
    title="Final clean data after all QC procedures",
    data_to_plot=plot_config_fig12
)


size1=[CWS_UrbanClimate_net.size+CWS_UrbanClimate_wund.size,

       CWS_UrbanClimate_net_QC_I3.size+CWS_UrbanClimate_wund_QC_I3.size,
       CWS_UrbanClimate_net_QC_I4.size+CWS_UrbanClimate_wund_QC_I4.size,
       CWS_UrbanClimate_net_QC_I5.size+CWS_UrbanClimate_wund_QC_I5.size,
       CWS_UrbanClimate_all_QC_G4.size,
       CWS_UrbanClimate_all_QC_G5.size,
       CWS_UrbanClimate_all_QC_G5b.size,
       CWS_UrbanClimate_all_QC_G6.size,
       CWS_UrbanClimate_all_QC_G7.size,
       CWS_UrbanClimate_all_QC_G8.size]


number1=[CWS_UrbanClimate_net.stack().count()+CWS_UrbanClimate_wund.stack().count(),

         CWS_UrbanClimate_net_QC_I3.stack().count()+CWS_UrbanClimate_wund_QC_I3.stack().count(),
         CWS_UrbanClimate_net_QC_I4.stack().count()+CWS_UrbanClimate_wund_QC_I4.stack().count(),
         CWS_UrbanClimate_net_QC_I5.stack().count()+CWS_UrbanClimate_wund_QC_I5.stack().count(),
         CWS_UrbanClimate_all_QC_G4.stack().count(),
         CWS_UrbanClimate_all_QC_G5.stack().count(),
         CWS_UrbanClimate_all_QC_G5b.stack().count(),
         CWS_UrbanClimate_all_QC_G6.stack().count(),
         CWS_UrbanClimate_all_QC_G7.stack().count(),
         CWS_UrbanClimate_all_QC_G8.stack().count()]


percentage0= [(i / j)*100 for i, j in zip( number1,size1)]


number_CWS = [len(CWS_UrbanClimate_net.columns)+len(CWS_UrbanClimate_wund.columns),

              len(CWS_UrbanClimate_net_QC_I3.columns)+len(CWS_UrbanClimate_wund_QC_I3.columns),
              len(CWS_UrbanClimate_net_QC_I4.columns)+len(CWS_UrbanClimate_wund_QC_I4.columns),
              len(CWS_UrbanClimate_net_QC_I5.columns)+len(CWS_UrbanClimate_wund_QC_I5.columns),
              len(CWS_UrbanClimate_all_QC_G4.columns),
              len(CWS_UrbanClimate_all_QC_G5.columns),
              len(CWS_UrbanClimate_all_QC_G5b.columns),
              len(CWS_UrbanClimate_all_QC_G6.columns),
              len(CWS_UrbanClimate_all_QC_G7.columns),
              len(CWS_UrbanClimate_all_QC_G8.columns)]


helper1 = [CWS_UrbanClimate_net.stack().count()+CWS_UrbanClimate_wund.stack().count(),
           CWS_UrbanClimate_net.stack().count()+CWS_UrbanClimate_wund.stack().count(),

           CWS_UrbanClimate_net_QC_I3.stack().count()+CWS_UrbanClimate_wund_QC_I3.stack().count(),
           CWS_UrbanClimate_net_QC_I4.stack().count()+CWS_UrbanClimate_wund_QC_I4.stack().count(),
           CWS_UrbanClimate_net_QC_I5.stack().count()+CWS_UrbanClimate_wund_QC_I5.stack().count(),
           CWS_UrbanClimate_all_QC_G4.stack().count(),
           CWS_UrbanClimate_all_QC_G5.stack().count(),
           CWS_UrbanClimate_all_QC_G5b.stack().count(),
           CWS_UrbanClimate_all_QC_G6.stack().count(),
           CWS_UrbanClimate_all_QC_G7.stack().count()]

percentage1 = [(i / j -1)*100 for i, j in zip(number1,helper1)]

percentage2 = (number1/(CWS_UrbanClimate_net.stack().count()+CWS_UrbanClimate_wund.stack().count())-1)*100
percentage2 = percentage2.tolist()


statistics_QC = {'QC level': ["raw","QC_I4","QC_I5","QC_I6","QC_G3","QC_G4","QC_G5","QC_G6","QC_G7","QC_G8"],
                 'Size of DataFrame': size1,
                 'Number of available data': number1,
                 'Available data (%)': percentage0,
                 'Number of available CWS': number_CWS,
                 'Outliers identified (%)': percentage1,
                 "Accumulated outliers (%)": percentage2
                 }

df_statistics_QC = pd.DataFrame(data=statistics_QC)


print(OWS_Climate_QC_I5.size)
print(OWS_Climate_QC_I5.stack().count()/OWS_Climate_QC_I5.size)
print((OWS_Climate_QC_I5.stack().count()/OWS_Climate_QC_I4.stack().count()-1)*100)


size1=[OWS_Climate.size,
       OWS_Climate_QC_I3.size,
       OWS_Climate_QC_I4.size,
       OWS_Climate_QC_I5.size]

number1=[OWS_Climate.stack().count(),
         OWS_Climate_QC_I3.stack().count(),
         OWS_Climate_QC_I4.stack().count(),
         OWS_Climate_QC_I5.stack().count()]

percentage0= [(i / j)*100 for i, j in zip(number1,size1)]

number_CWS = [len(OWS_Climate_QC.columns),
              len(OWS_Climate_QC_I3.columns),
              len(OWS_Climate_QC_I4.columns),
              len(OWS_Climate_QC_I5.columns)]


helper1 = [OWS_Climate.stack().count(),
           OWS_Climate.stack().count(),
           OWS_Climate_QC_I3.stack().count(),
           OWS_Climate_QC_I4.stack().count()]

percentage1 = [(i / j -1)*100 for i, j in zip(number1,helper1)]
percentage2 = (number1/OWS_Climate.stack().count()-1)*100


statistics_QC_OWS = {'QC level': ["raw","QC_I3","QC_I4","QC_I5"],
                 'Size of DataFrame': size1,
                 'Number of available data': number1,
                 'Available data (%)': percentage0,
                 'Number of available CWS': number_CWS,
                 'Outliers identified (%)': percentage1,
                 "Accumulated outliers (%)": percentage2
                 }

df_statistics_QC_OWS = pd.DataFrame(data=statistics_QC_OWS)


List_helper = list(CWS_UrbanClimate_all_QC_G8.columns.values)
CWS_coordinates_all_QC = CWS_coordinates_all[
    CWS_coordinates_all['module_final'].isin(List_helper)
]


os.chdir(cwd_data_qc_results)


df_statistics_QC.to_csv(
    f"Pre-processing_{city}_{year1}_CWS_all_G8_g8_Statistics.csv",
    index=True,
    header=True
)
df_statistics_QC_OWS.to_csv(
    f"Pre-processing_{city}_{year1}_OWS_all_G8_g8_Statistics.csv",
    index=True,
    header=True
)


os.chdir(cwd_data_qc)

start_date = pd.to_datetime(first_date, format='%d-%m-%Y %H:%M')
end_date = pd.to_datetime(last_date, format='%d-%m-%Y %H:%M')


name_coordinates_cws_qc = f"Coordinates_{city}_CWS_str_qc.csv"
name_ta_cws_qc = f"ta_{city}_{start_date.year}-{end_date.year}_h_CWS_str_qc.csv"


name_coordinates_cws_qc_flagged = f"Coordinates_{city}_CWS_str_qc_flagged.csv"
name_ta_cws_qc_flagged = f"ta_{city}_{start_date.year}-{end_date.year}_h_CWS_str_qc_flagged.csv"
name_ta_cws_qc_flags_only = f"ta_{city}_{start_date.year}-{end_date.year}_h_CWS_str_qc_flags_only.csv"


name_coordinates_ows_qc = f"Coordinates_{city}_OWS_str_qc.csv"
name_ta_ows_qc = f"ta_{city}_{start_date.year}-{end_date.year}_h_OWS_str_qc.csv"


CWS_UrbanClimate_all_QC_G8.to_csv(name_ta_cws_qc, index=True, header=True)
CWS_coordinates_all_QC.to_csv(name_coordinates_cws_qc, index=False, header=True)


CWS_UrbanClimate_all_QC_flagged.to_csv(name_ta_cws_qc_flagged, index=True, header=True)
CWS_coordinates_all_flagged.to_csv(name_coordinates_cws_qc_flagged, index=False, header=True)


CWS_outlier_flags_only.to_csv(name_ta_cws_qc_flags_only, index=True, header=True)


OWS_Climate_QC_I5.to_csv(name_ta_ows_qc, index=True, header=True)
OWS_coordinates.to_csv(name_coordinates_ows_qc, index=False, header=True)

print("")
print("Saved clean CWS file:", name_ta_cws_qc)
print("Saved flagged CWS file:", name_ta_cws_qc_flagged)
print("Saved flags-only CWS file:", name_ta_cws_qc_flags_only)
print("Saved clean OWS file:", name_ta_ows_qc)
print("")
print("Total flagged CWS outliers:", int(CWS_outlier_flags_only.sum().sum()))
print("Rows in clean CWS file:", len(CWS_UrbanClimate_all_QC_G8))
print("Rows in flagged CWS file:", len(CWS_UrbanClimate_all_QC_flagged))
print("")
print("Pre-processing completed successfully")
