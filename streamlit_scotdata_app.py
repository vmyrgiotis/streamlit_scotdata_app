import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os
import xarray as xr
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



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

st.title('Scotland NPP Data Explorer')

# 1. User selects variable
selected_var = st.selectbox('Select variable to plot:', data_vars)



# 2. Load grid_info.nc directly from GitHub raw URL using xarray
# grid_url = "https://github.com/vmyrgiotis/streamlit_scotdata_app/blob/master/grid_info.nc"
grid_ds = xr.open_dataset("grid_info.nc")
lat = grid_ds['lat'].values
lon = grid_ds['lon'].values

# 3. Load Parquet file for selected variable from GitHub raw URL
# scotland_merged_dataset.parquet is stored on the GitHub repo (raw URL)
# github_raw_url = "https://github.com/vmyrgiotis/streamlit_scotdata_app/blob/master/scotland_merged_dataset.parquet"
cols = ['lat', 'lon', selected_var]
df = pd.read_parquet('scotland_merged_dataset.parquet', columns=cols)

# 4. Read grid info from NetCDF
grid_ds = xr.open_dataset(grid_path)
lat = grid_ds['lat'].values
lon = grid_ds['lon'].values

# 5. Project data onto grid using lat/lon
# Assumes df has columns 'lat', 'lon', and the variable
import numpy as np
grid_data = np.full((len(lat), len(lon)), np.nan)
lat_idx = {v: i for i, v in enumerate(lat)}
lon_idx = {v: i for i, v in enumerate(lon)}

for _, row in df.iterrows():
    i = lat_idx.get(row['lat'])
    j = lon_idx.get(row['lon'])
    if i is not None and j is not None:
        grid_data[i, j] = row[selected_var]

# 6. Plot
fig = px.imshow(
    grid_data,
    origin='lower',
    aspect='auto',
    labels={'x': 'Longitude', 'y': 'Latitude', 'color': selected_var},
    x=lon,
    y=lat,
    color_continuous_scale='Viridis',
    title=f'{selected_var} projected on grid'
)
fig.update_xaxes(range=[-8, 2])
fig.update_yaxes(range=[54, 61])
st.plotly_chart(fig, use_container_width=True)


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
