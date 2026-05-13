import streamlit as st
import pandas as pd
import plotly.express as px
from minio import Minio
import tempfile
import os
import xarray as xr
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load MinIO credentials from Streamlit secrets
MINIO_ACCESS_KEY = st.secrets["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = st.secrets["MINIO_SECRET_KEY"]

# --- MinIO client and NetCDF file download ---
minio_client = Minio(
    "general-gensto.datalabs.ceh.ac.uk",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True
)
# Specify bucket and object name
bucket_name = "notebooks"
object_name = "scotland_merged_dataset_v2.nc"

# --- Variable options ---
data_vars = [
    'Npp_500m',
    't2m',
    'tp',
    'ocs_0_100cm'
]
index_var = 'LC_Type5'

# LC_Type5 code to name and color mapping
lc_type5_info = {
    0:  {"name": "Water Bodies", "color": "#1c0dff"},
    1:  {"name": "Evergreen Needleleaf Trees", "color": "#05450a"},
    2:  {"name": "Evergreen Broadleaf Trees", "color": "#086a10"},
    3:  {"name": "Deciduous Needleleaf Trees", "color": "#54a708"},
    4:  {"name": "Deciduous Broadleaf Trees", "color": "#78d203"},
    5:  {"name": "Shrub", "color": "#dcd159"},
    6:  {"name": "Not Cultivated", "color": "#b6ff05"},
    7:  {"name": "Cereal Croplands", "color": "#dade48"},
    8:  {"name": "Broadleaf Croplands", "color": "#c24f44"},
    9:  {"name": "Urban and Built-up Lands", "color": "#a5a5a5"},
    10: {"name": "Permanent Snow and Ice", "color": "#69fff8"},
    11: {"name": "Non-Vegetated Lands", "color": "#f9ffa4"},
}
# Vegetation codes only (exclude 0, 9, 10, 11)
vegetation_codes = [1, 2, 3, 4, 5, 7, 8]

st.title('Map & Time Series Explorer')

# Variable selection
selected_var = st.selectbox('Select variable to plot:', data_vars)

# Only open the dataset and load data after variable selection
@st.cache_data(show_spinner=True)
def load_xarray_and_df(var_name):
    # Download the NetCDF file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
        minio_client.fget_object(bucket_name, object_name, tmp.name)
        tmp_path = tmp.name
    # Open only the selected variable and LC_Type5 if needed
    # Only load LC_Type5 if selected_var is Npp_500m or LC_Type5
    with xr.open_dataset(tmp_path, engine=None) as ds_all:
        all_vars = set(ds_all.data_vars)
    if var_name == index_var or var_name == 'Npp_500m':
        vars_to_keep = set([var_name, index_var]) & all_vars
    else:
        vars_to_keep = set([var_name]) & all_vars
    drop_vars = list(all_vars - vars_to_keep)
    ds = xr.open_dataset(tmp_path, engine=None, drop_variables=drop_vars)
    # Apply filtering as requested
    if 'Npp_500m' in ds and 'LC_Type5' in ds:
        ds['Npp_500m'] = ds['Npp_500m'].where(ds['LC_Type5'].isin(vegetation_codes))
    if 'ocs_0_100cm' in ds:
        ds['ocs_0_100cm'] = ds['ocs_0_100cm'].where(ds['ocs_0_100cm'] > 0)
    # Convert time to datetime64 if needed
    if 'time' in ds:
        try:
            ds['time'] = ds.indexes['time'].to_datetimeindex()
        except Exception:
            ds['time'] = ds['time'].astype(str)
    # For time series plot, use DataFrame
    sel_vars = [var_name]
    if index_var in ds.data_vars and (var_name == index_var or var_name == 'Npp_500m'):
        sel_vars.append(index_var)
    df = ds[sel_vars].to_dataframe().reset_index()
    df = df.dropna(subset=[var_name])
    # Clean up temp file after loading
    os.remove(tmp_path)
    return ds, df

ds, df = load_xarray_and_df(selected_var)


# Handle ocs_0_100cm (no time dimension)
if selected_var == 'ocs_0_100cm' or 'time' not in ds[selected_var].dims:
    da = ds[selected_var]
    selected_year = None
else:
    # Year selection with slider (annual data)
    unique_years = sorted({pd.to_datetime(t).year for t in ds['time'].values})
    selected_year = st.slider('Select year:', min_value=min(unique_years), max_value=max(unique_years), value=min(unique_years), format="%d")
    # Select data for the selected year (xarray)
    year_mask = pd.to_datetime(ds['time'].values).year == selected_year
    da = ds[selected_var].isel(time=year_mask)

# Map plot using plotly imshow, centered on Scotland
if selected_var == 'ocs_0_100cm' or 'time' not in ds[selected_var].dims:
    st.subheader(f'Map of {selected_var}')
else:
    st.subheader(f'Map of {selected_var} for {selected_year}')
if da.size > 0:
    import numpy as np
    # da: dims (lat, lon) or (time, lat, lon) with time filtered
    if 'time' in da.dims:
        da2d = da.squeeze('time')
    else:
        da2d = da
    # Only filter out non-vegetation codes if Npp_500m is selected
    if selected_var == 'Npp_500m' and index_var in da2d.coords:
        mask = np.isin(da2d[index_var].values, vegetation_codes)
        if mask.shape == da2d.shape:
            da2d = da2d.where(mask)
    # For tp and t2m, plot all valid data (no mask)
    # Ensure lat is sorted ascending for imshow
    if np.any(np.diff(da2d['lat'].values) < 0):
        da2d = da2d.sortby('lat')
    # Remove NaNs for imshow
    plot_data = np.where(np.isnan(da2d.values), None, da2d.values)
    fig = px.imshow(
        plot_data,
        origin='lower',
        aspect='auto',
        labels={'x': 'Longitude', 'y': 'Latitude', 'color': selected_var},
        x=da2d['lon'].values,
        y=da2d['lat'].values,
        color_continuous_scale='Viridis',
        title=f'{selected_var}' + (f' ({selected_year})' if selected_year else ''),
    )
    # Set map extent to Scotland
    fig.update_xaxes(range=[-8, 2])
    fig.update_yaxes(range=[54, 61])
    st.plotly_chart(fig, use_container_width=True)
else:
    if selected_var == 'ocs_0_100cm' or 'time' not in ds[selected_var].dims:
        st.warning('No data available.')
    else:
        st.warning('No data for selected year.')

# --- Memory Optimization ---
# Downcast numeric columns only if they exist
for col in ['lat', 'lon', selected_var]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
if index_var in df.columns:
    df[index_var] = pd.to_numeric(df[index_var], downcast='integer')


# Time series plot per LC_Type5 (annual) only for selected variables and vegetation codes
if selected_var in ['Npp_500m', 't2m', 'tp']:
    ts_df = df.copy()
    ts_df['year'] = pd.to_datetime(ts_df['time']).dt.year
    if selected_var == 'Npp_500m':
        # Filter to vegetation codes only for Npp_500m
        ts_df = ts_df[ts_df[index_var].isin(vegetation_codes)]
        # Map LC_Type5 code to name and color
        ts_df['LC_Type5_name'] = ts_df[index_var].map(lambda x: lc_type5_info.get(x, {}).get('name', str(x)))
        color_map = {lc_type5_info[k]['name']: lc_type5_info[k]['color'] for k in vegetation_codes}
        ts_df = ts_df.groupby(['year', 'LC_Type5_name'])[selected_var].mean().reset_index()
        st.subheader(f'Time Series of {selected_var} (per LC Type)')
        if not ts_df.empty:
            fig_ts = px.line(
                ts_df,
                x='year',
                y=selected_var,
                color='LC_Type5_name',
                markers=True,
                color_discrete_map=color_map
            )
            st.plotly_chart(fig_ts)
        else:
            st.warning('No time series data available.')
    else:
        # For t2m and tp, show overall mean time series (no vegetation filter)
        ts_df = ts_df.groupby('year')[selected_var].mean().reset_index()
        st.subheader(f'Time Series of {selected_var} (Scotland mean)')
        if not ts_df.empty:
            fig_ts = px.line(
                ts_df,
                x='year',
                y=selected_var,
                markers=True
            )
            st.plotly_chart(fig_ts)
        else:
            st.warning('No time series data available.')

st.info('Edit the code if your NetCDF variable names differ.')
