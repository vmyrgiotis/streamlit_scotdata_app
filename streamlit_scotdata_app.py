import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xarray as xr
from minio import Minio

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="Scotland merged NetCDF explorer", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-baseweb="select"] {
            max-width: 420px;
            margin-left: auto;
            margin-right: auto;
        }
        div[role="radiogroup"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            margin-left: 0.25rem;
            margin-right: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


MINIO_ACCESS_KEY = st.secrets["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = st.secrets["MINIO_SECRET_KEY"]

MINIO_ENDPOINT = "general-gensto.datalabs.ceh.ac.uk"
MINIO_BUCKET = "notebooks"
NETCDF_FILENAME = "scotland_merged_dataset_v2.nc"

INDEX_VAR = "LC_Type5"

VARIABLE_OPTIONS = [
    "Npp_500m",
    "t2m",
    "tp",
    "ocs_0_100cm",
    "LC_Type5",
]

LC_TYPE5_INFO = {
    0: {"name": "Water Bodies", "color": "#1c0dff"},
    1: {"name": "Evergreen Needleleaf Trees", "color": "#05450a"},
    2: {"name": "Evergreen Broadleaf Trees", "color": "#086a10"},
    3: {"name": "Deciduous Needleleaf Trees", "color": "#54a708"},
    4: {"name": "Deciduous Broadleaf Trees", "color": "#78d203"},
    5: {"name": "Shrub", "color": "#dcd159"},
    6: {"name": "Not Cultivated", "color": "#b6ff05"},
    7: {"name": "Cereal Croplands", "color": "#dade48"},
    8: {"name": "Broadleaf Croplands", "color": "#c24f44"},
    9: {"name": "Urban and Built-up Lands", "color": "#a5a5a5"},
    10: {"name": "Permanent Snow and Ice", "color": "#69fff8"},
    11: {"name": "Non-Vegetated Lands", "color": "#f9ffa4"},
}

vegetation_codes = [1, 2, 3, 4, 5, 7, 8]


@st.cache_resource
def open_dataset():
    if not MERGED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {MERGED_FILE}")
    return xr.open_dataset(MERGED_FILE, chunks="auto", cache=False)


def load_selected_dataset(var_name):
    ds = open_dataset()
    if var_name == INDEX_VAR:
        return ds[[var_name]]
    return ds[[var_name, INDEX_VAR]]


def apply_mask(ds, var_name):
    if var_name == INDEX_VAR:
        return ds[var_name]

    da = ds[var_name]
    lc = ds[INDEX_VAR]

    if var_name == "ocs_0_100cm":
        ds = ds.where((da >= 0) & (lc.isin(vegetation_codes)))
        return ds[var_name]

    if var_name == "Npp_500m":
        ds = ds.where((da >= 0) & (da <= 2) & (lc.isin(vegetation_codes)))
        return ds[var_name]

    if var_name == "tp":
        ds = ds.where((da >= 0) & (lc.isin(vegetation_codes)))
        return ds[var_name]

    if var_name == "t2m":
        ds = ds.where((da >= 0) & (lc.isin(vegetation_codes)))
        return ds[var_name]

    return da


def select_single_year(da, selected_year):
    if "time" not in da.dims or selected_year is None:
        return da

    da = da.sel(time=da["time"].dt.year == selected_year)

    ntime = da.sizes.get("time", 0)
    if ntime > 1:
        da = da.mean(dim="time", skipna=True)
    elif ntime == 1:
        da = da.isel(time=0, drop=True)

    return da


@st.cache_data
def time_da_to_df(var_name):
    ds = load_selected_dataset(var_name)
    da = apply_mask(ds, var_name).astype("float32")

    if "time" not in da.dims:
        return pd.DataFrame(columns=["year", var_name])

    dims_to_mean = [d for d in da.dims if d != "time"]
    ts = da.mean(dim=dims_to_mean, skipna=True).compute()

    df = pd.DataFrame({
        "year": ts["time"].dt.year.values,
        var_name: ts.values,
    })
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", var_name])

    if not df.empty:
        df["year"] = df["year"].astype(int)

    return df


@st.cache_data
def time_var_by_lc_df(var_name):
    ds = load_selected_dataset(var_name)
    da = apply_mask(ds, var_name).astype("float32")

    if "time" not in da.dims:
        return pd.DataFrame(columns=["year", var_name, "LC_Type5_name"])

    lc = ds[INDEX_VAR]
    series_frames = []

    for code, info in LC_TYPE5_INFO.items():
        masked = da.where(lc == code)
        dims_to_mean = [d for d in masked.dims if d != "time"]
        ts = masked.mean(dim=dims_to_mean, skipna=True).compute()

        tmp = pd.DataFrame({
            "year": ts["time"].dt.year.values,
            var_name: ts.values,
        })
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp = tmp.dropna(subset=["year", var_name])

        if not tmp.empty:
            tmp["year"] = tmp["year"].astype(int)
            tmp["LC_Type5_code"] = code
            tmp["LC_Type5_name"] = info["name"]
            series_frames.append(tmp)

    if series_frames:
        return pd.concat(series_frames, ignore_index=True)

    return pd.DataFrame(columns=["year", var_name, "LC_Type5_code", "LC_Type5_name"])


@st.cache_data
def time_lc_fraction_df():
    ds = open_dataset()[[INDEX_VAR]]
    da = ds[INDEX_VAR]

    series_frames = []

    for code, info in LC_TYPE5_INFO.items():
        masked = xr.where(da == code, 1, 0)
        dims_to_mean = [d for d in masked.dims if d != "time"]
        ts = masked.mean(dim=dims_to_mean, skipna=True).compute()

        tmp = pd.DataFrame({
            "year": ts["time"].dt.year.values,
            "fraction": ts.values,
        })
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp = tmp.dropna(subset=["year", "fraction"])

        if not tmp.empty:
            tmp["year"] = tmp["year"].astype(int)
            tmp["LC_Type5_name"] = info["name"]
            series_frames.append(tmp)

    if series_frames:
        return pd.concat(series_frames, ignore_index=True)

    return pd.DataFrame(columns=["year", "fraction", "LC_Type5_name"])


def make_map(da, selected_var, selected_year=None):
    if "lat" not in da.coords or "lon" not in da.coords:
        st.error(f"{selected_var} must have 'lat' and 'lon' coordinates in the NetCDF file.")
        return

    lat0 = da["lat"].isel(lat=0).compute().item()
    latn = da["lat"].isel(lat=-1).compute().item()
    if lat0 > latn:
        da = da.sortby("lat")

    title = f"{selected_var}" if selected_year is None else f"{selected_var} ({selected_year})"
    plot_da = da.astype("float32").compute()

    if selected_var == INDEX_VAR:
        sorted_items = sorted(LC_TYPE5_INFO.items())
        color_scale = [item[1]["color"] for item in sorted_items]
        tickvals = [item[0] for item in sorted_items]
        ticktext = [item[1]["name"] for item in sorted_items]

        fig = px.imshow(
            plot_da.values,
            origin="lower",
            aspect="auto",
            x=plot_da["lon"].values,
            y=plot_da["lat"].values,
            labels={"x": "Longitude", "y": "Latitude", "color": selected_var},
            color_continuous_scale=color_scale,
            zmin=min(tickvals),
            zmax=max(tickvals),
            title=title,
        )
        fig.update_coloraxes(
            colorbar=dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
            )
        )
    else:
        fig = px.imshow(
            plot_da.values,
            origin="lower",
            aspect="auto",
            x=plot_da["lon"].values,
            y=plot_da["lat"].values,
            labels={"x": "Longitude", "y": "Latitude", "color": selected_var},
            color_continuous_scale="Viridis",
            title=title,
        )

    fig.update_xaxes(range=[-8, 2])
    fig.update_yaxes(range=[54, 61])
    st.plotly_chart(fig, use_container_width=True)


st.markdown("<h1 style='text-align: center;'>Scotland merged NetCDF explorer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Explore spatial maps and annual time series, including LC_Type5 class means for variables with a time dimension.</p>",
    unsafe_allow_html=True,
)

try:
    ds0 = open_dataset()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not open dataset: {e}")
    st.stop()

available_vars = [v for v in VARIABLE_OPTIONS if v in ds0.data_vars]

selected_var = st.selectbox("Select variable", options=available_vars, index=0)

st.markdown(
    f"<p style='text-align: center;'>Selected variable: <b>{selected_var}</b></p>",
    unsafe_allow_html=True,
)

ds = load_selected_dataset(selected_var)
da = apply_mask(ds, selected_var)
has_time = ("time" in da.dims) and (selected_var != "ocs_0_100cm")

tab_labels = ["Map"]
if has_time:
    tab_labels.append("Time series")

tabs = st.tabs(tab_labels)

with tabs[0]:
    selected_year = None

    if "time" in da.dims:
        years = da["time"].dt.year.compute().values
        unique_years = sorted(np.unique(years))

        selected_year = st.selectbox(
            "Select year for map",
            options=[int(y) for y in unique_years],
            index=0,
        )

        da_map = select_single_year(da, selected_year)
    else:
        da_map = da

    make_map(da_map, selected_var, selected_year)

if has_time:
    with tabs[1]:
        if selected_var == INDEX_VAR:
            ts_plot = time_lc_fraction_df()
            if not ts_plot.empty:
                color_map = {info["name"]: info["color"] for _, info in LC_TYPE5_INFO.items()}
                fig_ts = px.line(
                    ts_plot,
                    x="year",
                    y="fraction",
                    color="LC_Type5_name",
                    markers=True,
                    color_discrete_map=color_map,
                    labels={"fraction": "Spatial mean class presence"},
                    title="LC_Type5 class fraction through time",
                )
                fig_ts.update_traces(mode="lines+markers")
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.warning("No time series data available.")
        else:
            view_mode = st.radio(
                "Time series view",
                options=["Overall mean", f"Mean by {INDEX_VAR}"],
                horizontal=True,
            )

            if view_mode == "Overall mean":
                ts_plot = time_da_to_df(selected_var)
                if not ts_plot.empty:
                    fig_ts = px.line(
                        ts_plot,
                        x="year",
                        y=selected_var,
                        markers=True,
                        title=f"{selected_var} through time",
                    )
                    fig_ts.update_traces(mode="lines+markers")
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.warning("No time series data available.")
            else:
                ts_plot = time_var_by_lc_df(selected_var)
                if not ts_plot.empty:
                    color_map = {info["name"]: info["color"] for _, info in LC_TYPE5_INFO.items()}
                    fig_ts = px.line(
                        ts_plot,
                        x="year",
                        y=selected_var,
                        color="LC_Type5_name",
                        markers=True,
                        color_discrete_map=color_map,
                        title=f"{selected_var} through time by {INDEX_VAR}",
                    )
                    fig_ts.update_traces(mode="lines+markers")
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.warning("No LC_Type5 time series data available for this variable.")
