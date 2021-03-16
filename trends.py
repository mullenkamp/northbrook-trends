# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:20:30 2018

@author: michaelek
"""
import os
import numpy as np
import pandas as pd
from fbprophet import Prophet, plot
import io
import pickle
import yaml
from tethysts import Tethys
import xarray as xr
import copy
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
from seaborn import set_style, despine, set_context, color_palette
from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )

pd.options.display.max_columns = 10

#####################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)


owner = 'Environment Canterbury'
product_code = 'quality_controlled_data'

flow_ref = ['66204', '66214', '66417', '66213', '66210']
gwl_ref = ['M35/2679', 'M35/0366', 'M35/0472', 'M35/0538', 'M35/0312', 'M35/0222', 'M34/0232', 'M34/0207']
gwl_rec_ref = ['M35/2679', 'M35/0366']
# precip_ref = ['322410', '322211', '322110', '321212', '321310']
precip_ref = ['322110', '321310']

s1 = '1500fc71aab1fb7d0d4786a0'

groups = {1: ['M35/2679', 'M35/0366', 'M35/0472', 'M35/0538'], 2: ['M35/0312', 'M35/0222'], 3: ['M34/0232', 'M34/0207'], 4: ['66204', '66213', '66210'], 5: ['66417'], 6: ['66214'], 7: ['322110', '321310']}

change_points = {'M35/2679': 0.3, 'M35/0366': 0.05, 'M35/0472': 0.05, 'M35/0538': 0.05, 'M35/0312': 0.5, 'M35/0222': 0.1, 'M34/0232': 0.1, 'M34/0207': 0.3, '66204': 0.1, '66213': 0.1, '66417': 0.1, '66214': 0.1, '66210': 0.1, '322110': 0.1, '321310': 0.1}

y_axis_labels = {1: 'Depth below ground (m)', 2: 'Depth below ground (m)', 3: 'Depth below ground (m)', 4: 'Flow (m3/s)', 5: 'Flow (m3/s)', 6: 'Flow (m3/s)', 7: 'Depth (mm)'}
x_axis_labels = {'yearly_trend': 'Date', 'seasonality': 'Day of year', 'ts_plot': 'Date'}


flow_stns_shp = 'flow_sites_2021-03-16.shp'
gwl_stns_shp = 'gwl_sites_2021-03-16.shp'
precip_stns_shp = 'precip_sites_2021-03-16.shp'

##################################
### Get data

tethys1 = Tethys(param['remotes'])

## Datasets
datasets = tethys1.datasets.copy()

flow_ds = [ds for ds in datasets if (ds['owner'] == owner) and (ds['product_code'] == product_code) and (ds['parameter'] == 'streamflow') and (ds['frequency_interval'] == '1H')][0]

gwl_ds = [ds for ds in datasets if (ds['owner'] == owner) and (ds['product_code'] == product_code) and (ds['parameter'] == 'groundwater_depth') and (ds['feature'] == 'groundwater') and (ds['method'] == 'field_activity')][0]

gwl_rec_ds = [ds for ds in datasets if (ds['owner'] == owner) and (ds['product_code'] == product_code) and (ds['parameter'] == 'groundwater_depth') and (ds['feature'] == 'groundwater') and (ds['method'] != 'field_activity')][0]

precip_ds = [ds for ds in datasets if (ds['owner'] == owner) and (ds['product_code'] == product_code) and (ds['parameter'] == 'precipitation') and (ds['frequency_interval'] == '1H')][0]

## Stations
flow_stns0 = tethys1.get_stations(flow_ds['dataset_id'])
flow_stns = [s for s in flow_stns0 if s['ref'] in flow_ref]

gwl_stns0 = tethys1.get_stations(gwl_ds['dataset_id'])
gwl_stns = [s for s in gwl_stns0 if s['ref'] in gwl_ref]

gwl_rec_stns0 = tethys1.get_stations(gwl_rec_ds['dataset_id'])
gwl_rec_stns = [s for s in gwl_rec_stns0 if s['ref'] in gwl_rec_ref]

precip_stns0 = tethys1.get_stations(precip_ds['dataset_id'])
precip_stns = [s for s in precip_stns0 if s['ref'] in precip_ref]

## Results
flow_stn_ids = [s['station_id'] for s in flow_stns]
flow_dict = tethys1.get_bulk_results(flow_ds['dataset_id'], flow_stn_ids, remove_height=True)

flow_list = []
for stn_id, results in flow_dict.items():
    res1 = results.to_dataframe().reset_index()
    res1['station_id'] = stn_id
    res1['dataset_id'] = results.attrs['dataset_id']
    res1.rename(columns={'streamflow': 'value'}, inplace=True)
    flow_list.append(res1)

flow1 = pd.concat(flow_list)


gwl_stn_ids = [s['station_id'] for s in gwl_stns]
gwl_dict = tethys1.get_bulk_results(gwl_ds['dataset_id'], gwl_stn_ids, remove_height=True)

gwl_list = []
for stn_id, results in gwl_dict.items():
    res1 = results.to_dataframe().reset_index()
    res1['station_id'] = stn_id
    res1['dataset_id'] = results.attrs['dataset_id']
    res1.rename(columns={'groundwater_depth': 'value'}, inplace=True)
    gwl_list.append(res1)

gwl1 = pd.concat(gwl_list)

precip_stn_ids = [s['station_id'] for s in precip_stns]
precip_dict = tethys1.get_bulk_results(precip_ds['dataset_id'], precip_stn_ids, remove_height=True)

precip_list = []
for stn_id, results in precip_dict.items():
    res1 = results.to_dataframe().reset_index()
    res1['station_id'] = stn_id
    res1['dataset_id'] = results.attrs['dataset_id']
    res1.rename(columns={'precipitation': 'value'}, inplace=True)
    precip_list.append(res1)

precip1 = pd.concat(precip_list)


all_data = pd.concat([flow1, gwl1, precip1])
# all_data = pd.concat([flow1, gwl1])
all_stns = flow_stns.copy()
all_stns.extend(gwl_stns)
all_stns.extend(precip_stns)


def stns_dict_to_gdf(stns):
    """

    """
    stns1 = copy.deepcopy(stns)
    geo1 = [shapely.geometry.Point(s['geometry']['coordinates']) for s in stns1]

    [s.update({'min': s['stats']['min'], 'max': s['stats']['max'], 'from_date': s['stats']['from_date'], 'to_date': s['stats']['to_date']}) for s in stns1]
    [(s.pop('stats'), s.pop('geometry'), s.pop('results_object_key'), s.pop('virtual_station')) for s in stns1]
    [s.pop('properties') for s in stns1 if 'properties' in s]

    df1 = pd.DataFrame(stns1)
    # df1['from_date'] = pd.to_datetime(df1['from_date'])
    # df1['to_date'] = pd.to_datetime(df1['to_date'])
    # df1['modified_date'] = pd.to_datetime(df1['modified_date'])

    stns_gpd1 = gpd.GeoDataFrame(df1, crs=4326, geometry=geo1)

    return stns_gpd1


flow_stns_gdf = stns_dict_to_gdf(flow_stns)
gwl_stns_gdf = stns_dict_to_gdf(gwl_stns)
precip_stns_gdf = stns_dict_to_gdf(precip_stns)

flow_stns_gdf.to_file(os.path.join(base_dir, flow_stns_shp))
gwl_stns_gdf.to_file(os.path.join(base_dir, gwl_stns_shp))
precip_stns_gdf.to_file(os.path.join(base_dir, precip_stns_shp))

#############################################
### Testing

# data = flow1[flow1.station_id == s1].drop('station_id', axis=1).rename(columns={'time': 'ds', 'streamflow': 'y'}).set_index('ds')
# data1 = data.resample('D').mean().reset_index()

# m = Prophet()
# m.fit(data1)

# future = m.make_future_dataframe(periods=365)

# forecast = m.predict(future)
# fig2 = m.plot_components(forecast)
# # plot.plot_components_plotly(m, forecast)

# s2 = 'M35/0366'
# s2_id = [s['station_id'] for s in gwl_stns if s['ref'] == s2][0]
# data = gwl1[gwl1.station_id == s2_id].drop('station_id', axis=1).rename(columns={'time': 'ds', 'groundwater_depth': 'y'}).set_index('ds')
# data1 = data.resample('D').mean().interpolate('time').reset_index()

# m = Prophet()
# m.fit(data1)

# future = m.make_future_dataframe(periods=30)

# forecast = m.predict(future)
# fig2 = m.plot_components(forecast)

trend_dict = {i: {} for i in groups}
trend_m_dict = {i: {} for i in groups}
# trend_y_dict = {i: {} for i in groups}
for g, refs in groups.items():
    print(g)
    for ref in refs:
        print(ref)
        stn = [s for s in all_stns if s['ref'] == ref][0]
        stn_id = stn['station_id']
        ds_id = stn['dataset_id']
        data = all_data[(all_data.station_id == stn_id) & (all_data.dataset_id == ds_id)].drop(['station_id', 'dataset_id'], axis=1).rename(columns={'time': 'ds', 'value': 'y'}).set_index('ds')
        chng_pt = change_points[ref]
        # chng_pt = 0.1
        if not data.empty:
            ## Yearly
            # if g in [1, 2, 3]:
            #     data1 = data.resample('D').mean()
            #     data2 = data1.interpolate('time', limit=120).dropna()
            #     data2 = data2.resample('A-JUN').mean()
            # elif g == 7:
            #     data2 = data.resample('A-JUN').sum()
            #     # data2 = data1.interpolate('time', limit=1).dropna()
            # else:
            #     data2 = data.resample('A-JUN').mean()
            #     # data2 = data1.interpolate('time', limit=1).dropna()

            # m = Prophet(changepoint_prior_scale=chng_pt, changepoint_range=0.9, interval_width=0.95)
            # m.fit(data2.reset_index())
            # future = m.make_future_dataframe(periods=1, freq='A-JUN')
            # forecast = m.predict(future)
            # trend_y_dict[g].update({ref: (m, forecast)})

            ## Monthly
            if g in [1, 2, 3]:
                data1 = data.resample('D').mean()
                data2 = data1.interpolate('time', limit=120).dropna()
                data2 = data2.resample('M').mean()
            elif g == 7:
                data1 = data.resample('M').sum()
                data2 = data1.interpolate('time', limit=3).dropna()
            else:
                data1 = data.resample('M').mean()
                data2 = data1.interpolate('time', limit=3).dropna()

            m = Prophet(changepoint_prior_scale=chng_pt, changepoint_range=0.9, interval_width=0.95)
            m.fit(data2.reset_index())
            future = m.make_future_dataframe(periods=1, freq='M')
            forecast = m.predict(future)
            trend_m_dict[g].update({ref: (m, forecast)})

            # Daily
            if g == 7:
                data1 = data.resample('D').sum()
                data2 = data1.interpolate('time', limit=120).dropna()
            else:
                data1 = data.resample('D').mean()
                data2 = data1.interpolate('time', limit=120).dropna()

            m = Prophet(changepoint_prior_scale=chng_pt, changepoint_range=0.9, interval_width=0.95)
            m.fit(data2.reset_index())
            future = m.make_future_dataframe(periods=1)
            forecast = m.predict(future)
            trend_dict[g].update({ref: (m, forecast)})

            # m.plot(forecast)
            # m.plot_components(forecast)

# group = 4

# for f in trend_dict[group]:
#     m1, fore = trend_dict[group][f]
#     fig = plot.plot_components(m1, fore)
#     fig2 = plot.plot(m1, fore)


group = 7

for f in trend_m_dict[group]:
    m1, fore = trend_m_dict[group][f]
    fig = plot.plot_components(m1, fore)
    fig2 = plot.plot(m1, fore)

# group = 4

# for f in trend_y_dict[group]:
#     m1, fore = trend_y_dict[group][f]
#     fig = plot.plot_components(m1, fore)
#     fig2 = plot.plot(m1, fore)

################################################3
### Plots

## Yearly trends
set_style("white")
set_style("ticks")
# set_context('poster')
set_context('talk')
pal1 = color_palette()

x_label = x_axis_labels['yearly_trend']
data_label = 'Yearly Trend'

# for g in groups:
#     for f in trend_m_dict[g]:
#         y_label = y_axis_labels[g]
#         m1, fore = trend_m_dict[g][f]

#         data1 = fore[['ds', 'trend']].rename(columns={'ds': x_label, 'trend': data_label}).set_index(x_label).copy()

#         ax1 = data1.plot(ylabel=y_label, color=pal1[0], title='Site: ' + f, figsize=(15, 10))
#         plt.tight_layout()
#         plot1 = ax1.get_figure()

#         despine()

#         plot1.savefig(os.path.join(base_dir, 'plots', f.replace('/', '_')+'_yearly_trend.png'))



for g in groups:
    y_label = y_axis_labels[g]

    data_list = []
    for f in trend_m_dict[g]:

        m1, fore = trend_m_dict[g][f]

        data1 = fore[['ds', 'trend']].rename(columns={'ds': x_label, 'trend': f}).set_index(x_label).copy()

        data_list.append(data1)

    data_df = pd.concat(data_list, axis=1)

    ax1 = data_df.plot(ylabel=y_label, color=pal1, title=data_label, figsize=(15, 10))
    plt.tight_layout()
    plot1 = ax1.get_figure()

    despine()

    plot1.savefig(os.path.join(base_dir, 'plots', 'group_{g}_yearly_trend_from_monthly.png'.format(g=g)))


for g in groups:
    y_label = y_axis_labels[g]

    data_list = []
    for f in trend_dict[g]:

        m1, fore = trend_dict[g][f]

        data1 = fore[['ds', 'trend']].rename(columns={'ds': x_label, 'trend': f}).set_index(x_label).copy()

        data_list.append(data1)

    data_df = pd.concat(data_list, axis=1)

    ax1 = data_df.plot(ylabel=y_label, color=pal1, title=data_label, figsize=(15, 10))
    plt.tight_layout()
    plot1 = ax1.get_figure()

    despine()

    plot1.savefig(os.path.join(base_dir, 'plots', 'group_{g}_yearly_trend_from_daily.png'.format(g=g)))

## seasonal trends
set_style("white")
set_style("ticks")
# set_context('poster')
set_context('talk')
pal1 = color_palette()

x_label = x_axis_labels['seasonality']
data_label = 'Seasonality'

# for g in groups:
#     for f in trend_m_dict[g]:
#         y_label = y_axis_labels[g]
#         m1, fore = trend_m_dict[g][f]

#         data1 = fore[['ds', 'trend']].rename(columns={'ds': x_label, 'trend': data_label}).set_index(x_label).copy()

#         ax1 = data1.plot(ylabel=y_label, color=pal1[0], title='Site: ' + f, figsize=(15, 10))
#         plt.tight_layout()
#         plot1 = ax1.get_figure()

#         despine()

#         plot1.savefig(os.path.join(base_dir, 'plots', f.replace('/', '_')+'_seasonality.png'))

for g in groups:
    y_label = y_axis_labels[g]

    data_list = []
    for f in trend_dict[g]:

        m1, fore = trend_dict[g][f]

        data1 = fore[['ds', 'yearly']].rename(columns={'ds': x_label, 'yearly': f}).set_index(x_label).copy()
        data2 = data1.groupby(data1.index.dayofyear).mean()

        data_list.append(data2)

    data_df = pd.concat(data_list, axis=1)

    ax1 = data_df.plot(ylabel=y_label, color=pal1, title=data_label, figsize=(15, 10))
    plt.tight_layout()
    plot1 = ax1.get_figure()

    despine()

    plot1.savefig(os.path.join(base_dir, 'plots', 'group_{g}_seasonality.png'.format(g=g)))


## TS plots
set_style("white")
set_style("ticks")
# set_context('poster')
set_context('talk')
pal1 = color_palette()

x_label = x_axis_labels['ts_plot']
# data_label = 'Seasonality'

for g in groups:
    for f in trend_dict[g]:
        y_label = y_axis_labels[g]
        m, fcst = trend_dict[g][f]

        fig, ax = plt.subplots(figsize=(15, 10))
        # fig = plot.plot(m1, fore, ax=ax1, ylabel=y_label, xlabel=x_label)
        # ax1.legend(loc='upper right')

        fcst_t = fcst['ds'].dt.to_pydatetime()
        ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.')
        ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2')
        # ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
        # ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
        if m.uncertainty_samples:
            ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                            color='#0072B2', alpha=0.2)
        # Specify formatting to workaround matplotlib issue #12925
        locator = AutoDateLocator(interval_multiples=False)
        formatter = AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.legend(['Measured Data', 'Modelled Data', 'Uncertainty Bounds'], loc='upper right')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # data1 = fore[['ds', 'trend']].rename(columns={'ds': x_label, 'trend': data_label}).set_index(x_label).copy()

        # ax1 = data1.plot(ylabel=y_label, color=pal1[0], title='Site: ' + f, figsize=(15, 10))
        plt.title('Site: ' + f)
        despine()
        plt.tight_layout()
        plot1 = ax.get_figure()
        plt.close()

        plot1.savefig(os.path.join(base_dir, 'plots', f.replace('/', '_')+'_ts_plot.png'))

# for g in groups:
#     y_label = y_axis_labels[g]

#     data_list = []
#     for f in trend_dict[g]:

#         m1, fore = trend_dict[g][f]

#         data1 = fore[['ds', 'yearly']].rename(columns={'ds': x_label, 'yearly': f}).set_index(x_label).copy()
#         data2 = data1.groupby(data1.index.dayofyear).mean()

#         data_list.append(data2)

#     data_df = pd.concat(data_list, axis=1)

#     ax1 = data_df.plot(ylabel=y_label, color=pal1, title=data_label, figsize=(15, 10))
#     plt.tight_layout()
#     plot1 = ax1.get_figure()

#     despine()

#     plot1.savefig(os.path.join(base_dir, 'plots', 'group_{g}_seasonality.png'.format(g=g)))










