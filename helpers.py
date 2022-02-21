import pandas as pd
import datetime as dt
import plotly.express as px

def convert_epoch_column_to_datetime(epoch_column, turkey_time=True):
    """
    Convert epoch time to datetime object
    """
    utc = pd.to_datetime(epoch_column, unit="s").dt.tz_localize('UTC')
    if turkey_time:
        return utc.dt.tz_convert('Europe/Istanbul')
    return utc

def convert_to_epoch(df, date_col, hour_col):
    df["epoch"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col], unit='H')
    df['epoch'] = (df['epoch'] - dt.datetime(1970,1,1)).dt.total_seconds().astype(int)
    # df["epoch"] = df["epoch"].astype('int64')//1e9
    return df


def draw_map(df, lat_column="lat", lon_column="lon"):
    df = df[[lat_column, lon_column]].drop_duplicates().reset_index(drop=True)
    fig = px.scatter_geo(df, lat=lat_column,lon=lon_column)
    fig.update_layout(title = 'World map', title_x=0.5)
    fig.show()