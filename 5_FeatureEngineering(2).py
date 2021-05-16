"""
5단계
추가 feature engineering
base columns : char, race, class, maxlevel, guild_ox, play_day7_mean, play_day5_mean, play_day2_mean, y
add columns : guild_change, guild_level, play_sec7_mean, play_sec5_mean, play_sec2_mean, char_nth, orgrimmar
"""

import pandas as pd
import numpy as np
import time
import datetime
import warnings
warnings.filterwarnings(action='ignore')

start_time = time.time()
data1 = pd.read_csv('F:/통계논문/start_data.csv', encoding='utf-8',
                    dtype={'player': int,
                           'level': int,
                           'guild': int,
                           'race': object,
                           'class':object,
                           'zone' : object,
                           'timestamp': object,
                           'char': object,
                           'date': object,
                           'time': object})
print(data1.info())
print(data1.head())
print(data1.tail())


data3 = pd.read_csv("F:/통계논문/start_data3.csv", encoding='utf-8',
                    dtype={'char': object,
                           'race_id': int,
                           'race': object,
                           'class_id': int,
                           'class': object,
                           'max_level': int,
                           'y' : int,
                           'date_length' : int,
                           'guild_ox': int,
                           'play_day7_mean' : float,
                           'play_day5_mean' : float,
                           'play_day2_mean' : float})

print(data3.info())
print(data3.head())
print(data3.tail())
print(data3.describe())


### 평균 접속 시간 (평일/주말)
def play_hour(char, option):
    times = data1.loc[data1['char'] == char, ['timestamp', 'date', 'time']]
    times['timestamp'] = times['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    times['date'] = times['timestamp'].apply(lambda x: x.date())
    times['weekday'] = times['date'].apply(lambda x: datetime.datetime(x.year, x.month, x.day).isocalendar()[2])

    times['delta'] = (times.timestamp - times.timestamp.shift())
    times['time_gap'] = times['delta'].apply(lambda x: x / np.timedelta64(1, 's'))

    ## 일주일동안 접속 시간
    if option == 7:
        hours = []
        idx = times[times['time_gap'] > 2000].index
        for i in range(len(idx)):
            if i == 0:
                hours.append(times.loc[0: idx[i] - 1, 'time_gap'].sum(skipna=True))
            else:
                hours.append(times.loc[idx[i - 1] + 1: idx[i] - 1, 'time_gap'].sum(skipna=True))
        result = np.mean(np.array(hours))

    ## 평일동안 접속 시간
    elif option == 5:
        hours = []
        times = times.loc[times['weekday'].isin([1, 2, 3, 4, 5]), :]
        idx = times[times['time_gap'] > 2000].index
        for i in range(len(idx)):
            if i == 0:
                hours.append(times.loc[0: idx[i] - 1, 'time_gap'].sum(skipna=True))
            else:
                hours.append(times.loc[idx[i - 1] + 1: idx[i] - 1, 'time_gap'].sum(skipna=True))
        result = np.mean(np.array(hours))

    ## 주말동안 접속 시간
    elif option == 2:
        hours = []
        times = times.loc[times['weekday'].isin([6, 7]), :]
        idx = times[times['time_gap'] > 2000].index
        for i in range(len(idx)):
            if i == 0:
                hours.append(times.loc[0: idx[i] - 1, 'time_gap'].sum(skipna=True))
            else:
                hours.append(times.loc[idx[i - 1] + 1: idx[i] - 1, 'time_gap'].sum(skipna=True))
        result = np.mean(np.array(hours))
    
    return result

data3['play_sec7_mean'] = data3['char'].apply(lambda x: play_hour(x, 7))
print('success - 0')

data3['play_sec5_mean'] = data3['char'].apply(lambda x: play_hour(x, 5))
print('success - 1')

data3['play_sec2_mean'] = data3['char'].apply(lambda x: play_hour(x, 2))
print('success - 2')



### guild 변경 횟수
def guild(char, option):
       guild_unique = data1.loc[data1['char']==char, 'guild'].unique()
       if option == 'change':
              if guild_unique[0] == -1:
                     guild_unique = np.delete(guild_unique, 0)

              result = len(guild_unique)-1

              if result <0:
                     result = 0

       elif option == 'level':
              if (len(guild_unique) == 1) & (guild_unique[0] == -1):
                     result = -1
              else:
                     tmp = data1.loc[data1['char'] == char, ['guild', 'level']]
                     result = tmp.iloc[(tmp.guild.values != -1).argmax(), 1]

       return result

data3['guild_change'] = data3['char'].apply(lambda x: guild(x, 'change'))
print('success - 3')

data3['guild_level'] = data3['char'].apply(lambda x: guild(x, 'level'))
print('success - 4')



### 몇번째 char
def char_nth(char):
       tmp = data1.loc[data1['player'] == int(char.split('#')[0])]
       if len(tmp.char.unique()) == 1:
              result = 1
       else:
              result = list(tmp.char.unique()).index(char) + 1

       return result


data3['char_nth'] = data3['char'].apply(lambda x: char_nth(x))
print('success - 5')



### zone 체류
def zone_stay(char):
       zones = data1.loc[data1['char'] == char, 'zone']
       zones_unique = list(zones.unique())
       if 'Orgrimmar' in zones_unique:
              if 'Durotar' in zones_unique:
                     result = pd.Series(zones).value_counts()['Orgrimmar'] + pd.Series(zones).value_counts()['Durotar']
              else:
                     result = pd.Series(zones).value_counts()['Orgrimmar']

       else:
              if 'Durotar' in zones_unique:
                     result = pd.Series(zones).value_counts()['Durotar']
              else:
                     result = 0

       return result


data3['orgrimmar'] = data3['char'].apply(lambda x: zone_stay(x))
print('success - 6')


#print('zone : ', np.unique(data1.zone.unique()))
#zone = list(data1.zone)
#print(pd.Series(zone).value_counts()[0:20])



print('success all')



spent_time = time.time() - start_time
mins_spent = int(spent_time / 60)
secs_remainder = int(spent_time % 60)
print('Time of process: ', mins_spent, ':', secs_remainder)



