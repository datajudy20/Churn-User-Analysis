"""
3단계
전처리 완료한 데이터로 분석용 데이터 만들기 (캐릭터 별 최고 레벨 달성 여부)
columns : char, race, class, maxlevel, y
"""

import pandas as pd
import numpy as np
import time

start_time = time.time()

data = pd.read_csv('F:/통계논문/start_data.csv', encoding='utf-8',
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

print('Dataframe size:', data.shape)
print(data.info())
print(data.head(10))
print(data.tail(10))
#print('Data on {:.0f} players and {:.0f} their charachters available'.format(len(data['player'].unique()), len(data['char'].unique())))

race_unique = ['Orc', 'Tauren', 'Troll', 'Undead', 'Blood Elf']
class_unique = ['Hunter', 'Warrior', 'Warlock', 'Rogue', 'Shaman', 'Druid', 'Mage', 'Priest', 'Paladin']
char_unique = np.unique(data['char'])

df = pd.DataFrame(char_unique, columns=['char'])
df['race_id'] = df['char'].apply(lambda x: x[-2])
df['race'] = df['race_id'].apply(lambda x: race_unique[int(x)])
df['class_id'] = df['char'].apply(lambda x: x[-1])
df['class'] = df['class_id'].apply(lambda x: class_unique[int(x)])
print('-------------------------------')

def max_level(char):
       levels = data.loc[data['char']==char, 'level']
       return levels.values[-1]

df['max_level'] = df['char'].apply(lambda x: max_level(x))
df['y'] = df['max_level'].apply(lambda x: 1 if (x>=70) else 0 )
print(df)
df.to_csv("F:/통계논문/start_data2.csv", mode='w')


spent_time = time.time() - start_time
mins_spent = int(spent_time / 60)
secs_remainder = int(spent_time % 60)
print('Time of process: ', mins_spent, ':', secs_remainder)
