# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:20:30 2018

@author: michaelek
"""
import os
import numpy as np
import pandas as pd
import io
import pickle
import yaml
from tethysts import Tethys
import xarray as xr

pd.options.display.max_columns = 10

#####################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)


##################################
### Get data

tethys1 = Tethys(param['remotes'])



















def main():

    #####################################
    ### Parameters
    print('load parameters')

    base_dir = os.path.realpath(os.path.dirname(__file__))

    with open(os.path.join(base_dir, 'parameters.yml')) as param:
        param = yaml.safe_load(param)

    ts_local_tz = 'Etc/GMT-12'

    source = param['source']
    datasets = source['datasets'].copy()
    file_remote = param['remote']['file']
    s3_remote = param['remote']['s3']
    period = source['period']
    processing_code = source['processing_code']

    #####################################
    ### Do the work

    try:
        ### Initalize
        run_date = pd.Timestamp.today(tz='utc').round('s').tz_localize(None)
        # run_date_local = run_date.tz_convert(ts_local_tz).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')

        print('Start:')
        print(run_date)

        # ancillary_variables = ['modified_date']

        ### Create dataset_ids
        dataset_list = tu.process_datasets(datasets)

        ### Determine the last run date and process the old data if old enough
        # last_run_date_key = tu.process_buffer_last_run_date(dataset_list[0]['dataset_id'], dataset_list, run_date, s3_remote, 14)
        run_date_dict = tu.process_run_date(processing_code, dataset_list, s3_remote, run_date)

        ## Create the data_dict
        data_dict = {d['dataset_id']: [] for d in dataset_list}

        for meas in datasets:
            print('----- Starting new dataset group -----')
            print(meas)

            ### Pull out stations
            remote = source['site_data'][meas].copy()
            remote_ds = remote.pop('dataset_id')
            sites = source['sites'][meas].copy()

            t1 = Tethys([remote])
            remote_stns = t1.get_stations(remote_ds)
            remote_stns1 = [s for s in remote_stns if s['station_id'] in sites]
            rem_keys = ['dataset_id', 'stats', 'results_object_key', 'modified_date']
            [[s.pop(k) for k in rem_keys] for s in remote_stns1]

            ## Final station processing
            stns_dict = tu.process_stations_base(remote_stns1)

            ####################################
            ### Get ts data

            for stn_id, stn in stns_dict.items():
                print(stn['name'])

                url = source['data_url'][meas].format(ref=stn['ref'], period=period)

                resp = requests.get(url)

                b_io = io.BytesIO(resp.content)

                data1 = pd.read_csv(b_io, compression='zip')
                data1.columns = ['ref', 'time', datasets[meas][0]['parameter']]
                data1.drop('ref', axis=1, inplace=True)
                data1['time'] = pd.to_datetime(data1['time'].str.replace('.', '').str.upper(), format='%d/%m/%Y %I:%M:%S %p').dt.round('T')
                data1['height'] = 0
                data1['time'] = data1['time'].dt.tz_localize(ts_local_tz).dt.tz_convert('utc').dt.tz_localize(None)
                data1 = data1.iloc[((data1['time'].dt.hour == 0).idxmax()+1):].copy()

                mod_date = pd.Timestamp.today(tz='utc').round('s').tz_localize(None)

                ###########################################
                ## Package up into the data_dict
                if not data1.empty:
                    tu.prepare_results(data_dict, datasets[meas], stn, data1, mod_date, mod_date, other_closed='right')

        ########################################
        ### Save results and stations
        tu.update_results_s3(processing_code, data_dict, run_date_dict, s3_remote, threads=20, public_url=file_remote['connection_config'])

    except Exception as err:
        # print(err)
        print(traceback.format_exc())
        tu.email_msg(param['remote']['email']['sender_address'], param['remote']['email']['sender_password'], param['remote']['email']['receiver_address'], 'Failure on tethys-extraction-ecan-env', traceback.format_exc(), param['remote']['email']['smtp_server'])

    try:

        ### Aggregate all stations for the dataset
        print('Aggregate all stations for the dataset and all datasets in the bucket')

        s3 = tu.s3_connection(s3_remote['connection_config'], 30)

        for ds in dataset_list:
            ds_new = tu.put_remote_dataset(s3, s3_remote['bucket'], ds)
            ds_stations = tu.put_remote_agg_stations(s3, s3_remote['bucket'], ds['dataset_id'])

        ### Aggregate all datasets for the bucket
        ds_all = tu.put_remote_agg_datasets(s3, s3_remote['bucket'])

        ### Timings
        end_run_date = pd.Timestamp.today(tz='utc').round('s').tz_localize(None)

        print('-- Finished!')
        print(end_run_date)

    except Exception as err:
        # print(err)
        print(traceback.format_exc())
        tu.email_msg(param['remote']['email']['sender_address'], param['remote']['email']['sender_password'], param['remote']['email']['receiver_address'], 'Failure on tethys-extraction-ecan-env agg processes', traceback.format_exc(), param['remote']['email']['smtp_server'])



#################################################
### Schedule Running

# sleep(1)
main()
# schedule.every().day.at('06:00').do(process_fenz)
# schedule.every().day.at('18:00').do(process_fenz)
#
# while True:
#     schedule.run_pending()
#     sleep(1)
