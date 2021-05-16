"""
4단계
캐릭터 별 최고 레벨 달성 여부 분석용 데이터에 추가 전처리 및 feature engineering
전처리 : 첫 기록부터 65 ~ 70 레벨인 캐릭터와 게임 이용 기간 50일보다 작은 char 삭제
base columns : char, race, class, maxlevel, y
add columns : guild_ox, play_day7_mean, play_day5_mean, play_day2_mean
"""

import pandas as pd
import numpy as np
import time
import datetime

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


data2 = pd.read_csv('F:/통계논문/start_data2.csv', encoding='utf-8',
                    dtype={'char': object,
                           'race_id': int,
                           'race': object,
                           'class_id': int,
                           'class': object,
                           'max_level': int,
                           'y' : int})
print(data2.info())
print(data2.head())
print(data2.tail())
print(data2['y'].value_counts())


### 첫 기록부터 65-70레벨인 char 삭제
def start_70(char):
       levels = data1.loc[data1['char']==char, 'level']
       result = levels.values[0]
       return result

data2['start_level'] = data2['char'].apply(lambda x: start_70(x))
data2.drop(data2[data2['start_level']>64].index, inplace=True)
print(data2['y'].value_counts())
print('success - 0')


### 게임 이용 기간이 50일보다 작은 char 삭제
def date_len(char):
       dates = data1.loc[data1['char']==char, 'date']
       result = datetime.datetime.strptime(dates.values[-1], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(dates.values[0], '%Y-%m-%d %H:%M:%S')
       return result.days

data2['date_length'] = data2['char'].apply(lambda x: date_len(x))
data2.drop(data2[data2['date_length']<=50].index, inplace=True)
print(data2['y'].value_counts())
print('success - 1')


### y값 비율 맞추기 (1:1)
sample_n = data2['y'].value_counts()[1]
tmp1 = data2.loc[data2['y']==1]
tmp0 = data2.loc[data2['y']==0].sample(n=sample_n+50)
data = pd.concat([tmp0,tmp1])
print(data.info())
print(data.head())
print(data.tail())
print(data['y'].value_counts())
print('success - 2')

"""
### 길드 유무 변수 만들기
def guild_ox(char):
       guild_list = np.unique(data1.loc[data1['char']==char, 'guild'])
       if len(guild_list)>1:
              result = 1
       elif guild_list[0] != -1:
              result = 1
       else:
              result = 0
       return result

data['guild_ox'] = data['char'].apply(lambda x: guild_ox(x))
print('success - 3')


### 평균 접속 일수 (일주일)
def play_day(char, option):
       times= data1.loc[data1['char']==char, ['timestamp', 'date', 'time']]
       times['date'] = times['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())
       times['year'] = times['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
       times['n_week'] = times['date'].apply(lambda x: datetime.datetime(x.year, x.month, x.day).isocalendar()[1])
       times['weekday'] = times['date'].apply(lambda x: datetime.datetime(x.year, x.month, x.day).isocalendar()[2])

       ## 일주일동안 접속 일수
       if option == 7:
              grouped = times['weekday'].groupby([times['year'], times['n_week']])
              result = grouped.unique().apply(lambda x: len(x)).mean()

       ## 평일동안 접속 일수
       elif option == 5:
              times = times.loc[times['weekday'].isin([1, 2, 3, 4, 5]), :]
              if len(times) == 0:
                     result = 0
              else:
                     grouped = times['weekday'].groupby([times['year'], times['n_week']])
                     result = grouped.unique().apply(lambda x: len(x)).mean()

       ## 주말동안 접속 일수
       elif option == 2:
              times = times.loc[times['weekday'].isin([6,7]), :]
              if len(times) == 0:
                     result = 0
              else:
                     grouped = times['weekday'].groupby([times['year'], times['n_week']])
                     result = grouped.unique().apply(lambda x: len(x)).mean()

       return result
              
data['play_day7_mean'] = data['char'].apply(lambda x: play_day(x, 7))
data['play_day5_mean'] = data['char'].apply(lambda x: play_day(x, 5))
data['play_day2_mean'] = data['char'].apply(lambda x: play_day(x, 2))
print('success - 4')


data.to_csv("F:/통계논문/start_data3.csv", mode='w', index=False)"""
print('success all')


spent_time = time.time() - start_time
mins_spent = int(spent_time / 60)
secs_remainder = int(spent_time % 60)
print('Time of process: ', mins_spent, ':', secs_remainder)