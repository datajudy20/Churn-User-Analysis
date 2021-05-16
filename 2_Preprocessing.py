"""
2단계
저장된 CSV 파일 데이터 전처리 (기본 변수 하나하나 탐색)
"""

import pandas as pd
import numpy as np
import time

start_time = time.time()

wowah = pd.read_csv('F:/통계논문/wow_data_2.csv', encoding='utf-8',
                    dtype={'char': int,
                           ' level': int,
                           ' race': object,
                           ' charclass':object,
                           ' guild': int,
                           ' dummy1': object,
                           ' dummy2': object,
                           ' zone' : object,
                           ' timestamp': object})
wowah.drop(columns=[' dummy1', ' dummy2'], inplace=True)
wowah.rename({'char': 'player',
              ' level': 'level',
              ' race': 'race',
              ' charclass': 'class',
              ' guild': 'guild',
              ' dummy1' : 'dummy1',
              ' dummy2':'dummy2',
              ' zone' : 'zone',
              ' timestamp': 'timestamp'}, axis=1, inplace=True)
wowah['zone'].replace({'Dalaran競技場': 'Dalaran Arena'}, inplace=True)
wowah.reset_index(drop=True, inplace=True)

print(wowah.info())
#print(wowah.head())
#print(wowah.tail())
print(wowah.shape)

print('player', np.unique(wowah['player']))        ## 0-86244
print('level', np.unique(wowah['level']))          ## 1-70
print('race', np.unique(wowah['race']))


### class
wowah.drop(wowah[wowah['player']==2400].index, inplace=True)
wowah['class'] = wowah['class'].apply(lambda x: x.lstrip())
print('class', np.unique(wowah['class']))

### race+class+player
wowah['class_id'] = np.array(pd.factorize(wowah['class'])[0])
print(pd.factorize(wowah['class'])[1])
wowah['race_id'] = np.array(pd.factorize(wowah['race'])[0])
print(pd.factorize(wowah['race'])[1])
wowah['class_id'] = wowah['class_id'].astype(str)
wowah['race_id'] = wowah['race_id'].astype(str)
wowah['sym'] = pd.Series(index = wowah.index, data='#')
wowah['char'] = wowah[['player', 'sym', 'race_id', 'class_id']].astype(str).sum(axis=1)
wowah.drop(['race_id', 'class_id', 'sym'], axis=1, inplace=True)

print('Dataframe size:', wowah.shape)
print('Data on {:.0f} players and {:.0f} their charachters available'.format(len(wowah['player'].unique()), len(wowah['char'].unique())))


### guild
print(np.unique(wowah.loc[wowah['guild']==0, 'char']))
#df_tmp = wowah.loc[wowah['char'].isin(np.unique(wowah.loc[wowah['guild']==0, 'char'])), ['char','guild']]
#print(pd.Series(df_tmp.guild.values,index=df_tmp.char).to_dict())
print('guild', np.unique(wowah['guild']))


### zone
wowah.drop(wowah[wowah['player']==2029].index, inplace=True)
wowah.drop(wowah[wowah['player'].isin(np.unique(wowah.loc[wowah['zone'].isin([' 北方海岸', ' 未知', ' 監獄', ' 達納蘇斯', ' 龍骨荒野',
                                    '1231崔茲', '1608峽谷', '北方海岸', '時光洞穴', '未知',
                                    '毒牙沼澤', '監獄', '達納蘇斯', '麥克那爾']), 'player']))].index, inplace=True)
print('zone', np.unique(wowah['zone']))
print('Data on {:.0f} players and {:.0f} their charachters available'.format(len(wowah['player'].unique()), len(wowah['char'].unique())))


### timestamp
def time_transform(x):
    y = x.split()[0]
    return y[:-2] + '20' + y[-2:]

wowah['date'] = wowah['timestamp'].apply(time_transform)
wowah['time'] = wowah['timestamp'].apply(lambda x: x.split()[1][:-3])
wowah['timestamp'] = pd.to_datetime(wowah['timestamp'], format='%m/%d/%y %H:%M:%S')
wowah['date'] =  pd.to_datetime(wowah['date'], format='%m/%d/%Y')
wowah['time'] = pd.to_datetime(wowah['time'], format='%H:%M').dt.time

data = wowah.sort_values(by=['date', 'time'], axis=0)

data.to_csv("F:/통계논문/start_data.csv", mode='w')
print(data.info())
print(data.head(10))
print(data.tail(10))

spent_time = time.time() - start_time
mins_spent = int(spent_time / 60)
secs_remainder = int(spent_time % 60)
print('Time of process: ', mins_spent, ':', secs_remainder)