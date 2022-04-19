

import pandas as pd


def get_profiles_activities():
    datas = pd.read_csv('./utils/datas/forticloud-traffic-forward-2022-03-29_1121.log.xlsx', sep='delimiter', header=None, engine='python')
    data_source = datas.iloc[43, :].values
    type = datas.iloc[42, :].values
    source_ip = datas.iloc[10, :].values
    destination_ip = datas.iloc[15, :].values


