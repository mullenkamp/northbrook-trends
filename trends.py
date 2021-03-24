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
# import yaml
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

# with open(os.path.join(base_dir, 'parameters.yml')) as param:
#     param = yaml.safe_load(param)

remotes = [{'bucket': 'ecan-env-monitoring', 'connection_config': 'https://b2.tethys-ts.xyz'}]

mcmc_samples = 0 # Change to 300 to add uncertainty analysis to the outputs; adds a lot more processing time (~1 hour).

owner = 'Environment Canterbury'
product_code = 'quality_controlled_data'

flow_ref = ['66204', '66214', '66417', '66213', '66210']
gwl_ref = ['M35/2679', 'M35/0366', 'M35/0472', 'M35/0538', 'M35/0312', 'M35/0222', 'M34/0232', 'M34/0207']
gwl_rec_ref = ['M35/2679', 'M35/0366']
# precip_ref = ['322410', '322211', '322110', '321212', '321310']
precip_ref = ['322110', '321310']


groups = {1: ['M35/2679', 'M35/0366', 'M35/0472', 'M35/0538'], 2: ['M35/0312', 'M35/0222'], 3: ['M34/0232', 'M34/0207'], 4: ['66204', '66213', '66210'], 5: ['66417'], 6: ['66214'], 7: ['322110', '321310']}

change_points = {'M35/2679': 0.3, 'M35/0366': 0.05, 'M35/0472': 0.05, 'M35/0538': 0.05, 'M35/0312': 0.5, 'M35/0222': 0.1, 'M34/0232': 0.1, 'M34/0207': 0.3, '66204': 0.1, '66213': 0.1, '66417': 0.1, '66214': 0.1, '66210': 0.1, '322110': 0.1, '321310': 0.1}

y_axis_labels = {1: 'Depth below ground (m)', 2: 'Depth below ground (m)', 3: 'Depth below ground (m)', 4: 'Flow (m3/s)', 5: 'Flow (m3/s)', 6: 'Flow (m3/s)', 7: 'Depth (mm)'}
x_axis_labels = {'yearly_trend': 'Date', 'seasonality': 'Day of year', 'ts_plot': 'Date'}


flow_stns_shp = 'flow_sites_2021-03-16.shp'
gwl_stns_shp = 'gwl_sites_2021-03-16.shp'
precip_stns_shp = 'precip_sites_2021-03-16.shp'

###################################
### Extra functions


def stns_dict_to_gdf(stns):
    """

    """
    stns1 = copy.deepcopy(stns)
    geo1 = [shapely.geometry.Point(s['geometry']['coordinates']) for s in stns1]

    [s.update({'min': s['stats']['min'], 'max': s['stats']['max'], 'from_date': s['stats']['from_date'], 'to_date': s['stats']['to_date']}) for s in stns1]
    [(s.pop('stats'), s.pop('geometry'), s.pop('results_object_key'), s.pop('virtual_station')) for s in stns1]
    [s.pop('properties') for s in stns1 if 'properties' in s]

    df1 = pd.DataFrame(stns1)

    stns_gpd1 = gpd.GeoDataFrame(df1, crs=4326, geometry=geo1)

    return stns_gpd1


def set_y_as_percent(ax):
    yticks = 100 * ax.get_yticks()
    yticklabels = ['{0:.4g}%'.format(y) for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax


def plot_trends(m, fcst, x_label, y_label, legend_label, color='#0072B2', ax=None, uncertainty=True, plot_cap=False, figsize=(15, 10)):
    """
    Plot a particular component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    ax: Optional matplotlib Axes to plot on.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    matplotlib ax
    """
    set_style("white")
    set_style("ticks")
    # set_context('poster')
    set_context('talk')

    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst['ds'].dt.to_pydatetime()
    artists += ax.plot(fcst_t, fcst['trend'], ls='-', c=color, label=legend_label)
    if 'cap' in fcst and plot_cap:
        artists += ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(
            fcst_t, fcst['trend_lower'], fcst['trend_upper'],
            color=color, alpha=0.2)]
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if 'trend' in m.component_modes['multiplicative']:
        ax = set_y_as_percent(ax)
    plt.tight_layout()
    despine()
    return ax


def plot_seasonality(m, fcst, x_label, y_label, legend_label, color='#0072B2', ax=None, uncertainty=True, plot_cap=False, figsize=(15, 10)):
    """
    Plot a particular component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    ax: Optional matplotlib Axes to plot on.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    matplotlib ax
    """
    set_style("white")
    set_style("ticks")
    set_context('talk')

    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    ## Prepare data
    fcst1 = fcst.groupby(fcst['ds'].dt.dayofyear).mean()

    fcst_t = fcst1.index
    artists += ax.plot(fcst_t, fcst1['yearly'], ls='-', c=color, label=legend_label)
    if 'cap' in fcst1 and plot_cap:
        artists += ax.plot(fcst_t, fcst1['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst1 and plot_cap:
        ax.plot(fcst_t, fcst1['floor'], ls='--', c='k')
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(
            fcst_t, fcst1['yearly_lower'], fcst1['yearly_upper'],
            color=color, alpha=0.2)]
    # Specify formatting to workaround matplotlib issue #12925
    # locator = AutoDateLocator(interval_multiples=False)
    # formatter = AutoDateFormatter(locator)
    # ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if 'trend' in m.component_modes['multiplicative']:
        ax = set_y_as_percent(ax)
    plt.tight_layout()
    despine()
    return ax


def plot_results(m, fcst, x_label, y_label, title_label, color='#0072B2', ax=None, uncertainty=True, figsize=(15, 10)):
    """

    """
    set_style("white")
    set_style("ticks")
    set_context('talk')

    if not ax:
        fig, ax = plt.subplots(figsize=(15, 10))
    fcst_t = fcst['ds'].dt.to_pydatetime()
    ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.')
    ax.plot(fcst_t, fcst['yhat'], ls='-', c=color)
    if uncertainty and m.uncertainty_samples:
        ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                        color=color, alpha=0.2)
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.legend(['Measured Data', 'Modelled Data', 'Uncertainty Bounds'], loc='upper right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title_label)
    despine()
    plt.tight_layout()

    return ax

##################################
### Get data

tethys1 = Tethys(remotes)

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

# Prepare results
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

# Combine results and stations
all_data = pd.concat([flow1, gwl1, precip1])
# all_data = pd.concat([flow1, gwl1])
all_stns = flow_stns.copy()
all_stns.extend(gwl_stns)
all_stns.extend(precip_stns)

### Export site locations
flow_stns_gdf = stns_dict_to_gdf(flow_stns)
gwl_stns_gdf = stns_dict_to_gdf(gwl_stns)
precip_stns_gdf = stns_dict_to_gdf(precip_stns)

flow_stns_gdf.to_file(os.path.join(base_dir, flow_stns_shp))
gwl_stns_gdf.to_file(os.path.join(base_dir, gwl_stns_shp))
precip_stns_gdf.to_file(os.path.join(base_dir, precip_stns_shp))

#############################################
### Analysis

# trend_dict = {i: {} for i in groups}
trend_m_dict = {i: {} for i in groups}
trend_w_dict = {i: {} for i in groups}
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
            if g != 7:
                # Weekly
                data1 = data.resample('D').mean()
                data2 = data1.interpolate('time', limit=120).dropna()
                data2 = data2.resample('W').mean()

                m = Prophet(changepoint_prior_scale=chng_pt, changepoint_range=0.9, interval_width=0.95, mcmc_samples=mcmc_samples)
                m.fit(data2.reset_index())
                future = m.make_future_dataframe(periods=1, freq='W')
                forecast = m.predict(future)
                trend_w_dict[g].update({ref: (m, forecast)})

                # Monthly
                data2 = data2.resample('M').mean()

                m = Prophet(changepoint_prior_scale=chng_pt, changepoint_range=0.9, interval_width=0.95, mcmc_samples=mcmc_samples)
                m.fit(data2.reset_index())
                future = m.make_future_dataframe(periods=1, freq='W')
                forecast = m.predict(future)
                trend_m_dict[g].update({ref: (m, forecast)})
            else:
                data1 = data.resample('M').sum()
                data2 = data1.interpolate('time', limit=3).dropna()

                m = Prophet(changepoint_prior_scale=chng_pt, changepoint_range=0.9, interval_width=0.95, mcmc_samples=mcmc_samples)
                m.fit(data2.reset_index())
                future = m.make_future_dataframe(periods=1, freq='M')
                forecast = m.predict(future)
                trend_m_dict[g].update({ref: (m, forecast)})


### Save results

# group = 7

# for f in trend_dict[group]:
#     m1, fore = trend_dict[group][f]
#     fig = plot.plot_components(m1, fore)
#     fig2 = plot.plot(m1, fore)


# group = 1

# for f in trend_m_dict[group]:
#     m1, fore = trend_m_dict[group][f]
#     fig = plot.plot_components(m1, fore)
#     fig2 = plot.plot(m1, fore)

# group = 1

# for f in trend_w_dict[group]:
#     m1, fore = trend_w_dict[group][f]
#     fig = plot.plot_components(m1, fore)
#     fig2 = plot.plot(m1, fore)

################################################3
### Plots

pal1 = color_palette()

now1 = pd.Timestamp.now().strftime('%Y-%m-%d')

plot_dir = 'plots_' + now1
plot_path = os.path.join(base_dir, plot_dir)

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

data_groups = {'weekly': trend_w_dict, 'monthly': trend_m_dict}

for freq, trend_dict, in data_groups.items():
    ax1 = None
    for g, results in trend_dict.items():
        sites = list(results.keys())
        if sites:
            y_label = y_axis_labels[g]

            colors_dict = {sites[i]: pal1[i] for i in range(len(sites))}

            ## Trend plot
            x_label = x_axis_labels['yearly_trend']

            for f in sites:
                m1, fore = results[f]

                color = colors_dict[f]

                ax1 = plot_trends(m1, fore, x_label, y_label, f, ax=ax1, color=color)

            plot1 = ax1.get_figure()
            plt.legend()

            plot1.savefig(os.path.join(plot_path, 'group_{g}_trend_from_{freq}_data.png'.format(g=g, freq=freq)))
            plt.close()
            ax1 = None

            ## Seasonality plot
            if freq == 'weekly':
                x_label = x_axis_labels['seasonality']

                for f in sites:
                    m1, fore = results[f]

                    color = colors_dict[f]

                    ax1 = plot_seasonality(m1, fore, x_label, y_label, f, ax=ax1, color=color)

                plot1 = ax1.get_figure()
                plt.legend()

                plot1.savefig(os.path.join(plot_path, 'group_{g}_seasonality_from_{freq}_data.png'.format(g=g, freq=freq)))
                plt.close()
                ax1 = None

            ## Results plot
            x_label = x_axis_labels['ts_plot']

            for f in sites:
                m1, fore = results[f]

                color = colors_dict[f]

                ax1 = plot_results(m1, fore, x_label, y_label, 'Site: ' + f, color)
                plot1 = ax1.get_figure()
                site_ref = f.replace('/', '-')
                plot1.savefig(os.path.join(plot_path, 'group_{g}_{site}_results_from_{freq}_data.png'.format(g=g, freq=freq, site=site_ref)))
                plt.close()
                ax1 = None





