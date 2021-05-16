"""
6단계
데이터 탐색을 위한 시각화 - feature engineering으로 생성된 변수
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

"""import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
font_location = 'C:\windows\Fonts\H2GTRM.TTF'
font_name = fm.FontProperties(fname=font_location).get_name()
plt.rc('font', family=font_name)"""

from matplotlib import font_manager, rc
#font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/서울한강 장체B.ttf").get_name()
font_name = font_manager.FontProperties(fname="C:/Users/datajudy20/AppData/Local/Microsoft/Windows/Fonts/서울한강 장체B.ttf").get_name()

rc('font', family=font_name)

print('# 설정되어있는 폰트 사이즈')
plt.rcParams['font.size']=12
print (plt.rcParams['font.size'])
print('# 설정되어있는 폰트 글꼴')
print (plt.rcParams['font.family'] )
plt.rc('axes', unicode_minus=False)

start_time = time.time()

data = pd.read_csv("F:/통계논문/data_model.csv", encoding='utf-8',
                    dtype={'char': object,
                           'char_nth' : int,
                           'race_id': int,
                           'race': object,
                           'class_id': int,
                           'class': object,
                           'max_level': int,
                           'y' : int,
                           'date_length' : int,
                           'guild_ox': int,
                           'guild_change' : int,
                           'guild_level' : int,
                           'play_sec7_mean' : float,
                           'play_sec5_mean' : float,
                           'play_sec2_mean' : float,
                           'play_day7_mean' : float,
                           'play_day5_mean' : float,
                           'play_day2_mean' : float,
                           'orgrimmar' : int})


print(data['y'].value_counts())

print(data.info())
print(data.head())
print(data.tail())
print(data.describe())
print(data.columns)

### train / test 분리 ###
X = data.drop(columns=['char', 'race_id', 'class_id', 'max_level',
                       'date_length', 'start_level','y']).copy()
y = data['y']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values,
                                                    shuffle=True, test_size=0.25,
                                                    random_state=0)
X_train = pd.DataFrame(X_train, columns=X.columns)
y_train = pd.DataFrame(y_train, columns=['y'])
print(X_train)
print(y_train)
data = pd.concat([X_train, y_train], axis=1)
data[['guild_ox', 'char_nth']] = data[['guild_ox', 'char_nth']].apply(pd.to_numeric)
data[['guild_change', 'guild_level', 'orgrimmar',
      'play_day7_mean', 'play_day5_mean', 'play_day2_mean',
      'play_sec7_mean', 'play_sec5_mean', 'play_sec2_mean']] = data[['guild_change', 'guild_level', 'orgrimmar',
                                                                     'play_day7_mean', 'play_day5_mean', 'play_day2_mean',
                                                                     'play_sec7_mean', 'play_sec5_mean', 'play_sec2_mean']].apply(pd.to_numeric)
print(data.info())
print(data.head())


cols_cont = ['guild_change', 'guild_level', 'orgrimmar',
             'play_day7_mean', 'play_day5_mean', 'play_day2_mean',
             'play_sec7_mean', 'play_sec5_mean', 'play_sec2_mean']

cols_catg = ['race', 'class', 'guild_ox', 'char_nth']


### 반응변수 그래프 ###
fig, ax = plt.subplots()
data['y'].value_counts().plot.pie(autopct='%1.1f%%',
                                  startangle=90, fontsize=15, use_index=False,
                                  colors=['sienna','peru'],
                                  labels=['이탈 고객', '최고 레벨\n도달 고객'])
ax.set(ylabel='', aspect='equal')
plt.title('반응변수 분포 비율')
fig.savefig('F:/통계논문/image/piechart_y.png')
plt.show()


### 설명변수 그래프 - pie chart ###
fig, ax = plt.subplots()
data['race'].value_counts().plot.pie(autopct='%1.1f%%',
                                     startangle=90, fontsize=15,
                                     use_index=False, cmap='Oranges')
ax.set(ylabel='', aspect='equal')
plt.title('race 분포 비율')
fig.savefig('F:/통계논문/image/piechart_race.png')
plt.show()

fig, ax = plt.subplots()
data['class'].value_counts().plot.pie(autopct='%1.1f%%',
                                     startangle=90, fontsize=12,
                                     use_index=False, cmap='Oranges')
ax.set(ylabel='', aspect='equal')
plt.title('class 분포 비율')
fig.savefig('F:/통계논문/image/piechart_class.png')
plt.show()


fig, ax = plt.subplots()
data['guild_ox'].value_counts().plot.pie(autopct='%1.1f%%',
                                         startangle=90, fontsize=15, use_index=False,
                                         colors=['sienna','peru'],
                                         labels=['길드 가입 고객', '길드 미가입 고객'])
ax.set(ylabel='', aspect='equal')
plt.title('guild_ox 분포 비율')
fig.savefig('F:/통계논문/image/piechart_guild_ox.png')
plt.show()


print(data['char_nth'].value_counts())
tmp = pd.cut(data['char_nth'], bins=[0,1,2,10])
fig, ax = plt.subplots()
tmp.value_counts().plot.pie(explode=[0.1,0.2,0.3],autopct='%1.1f%%',
                            labels=['첫번째', '두번째', '세번째 이상'],
                            colors=['sienna','peru','darkorange'])
ax.set(ylabel='', aspect='equal')
plt.title('char_nth 분포 비율')
fig.savefig('F:/통계논문/image/piechart_char_nth.png')
plt.show()


fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
data['race'].value_counts().plot.pie(autopct='%1.1f%%',
                                     startangle=90, fontsize=12,
                                     use_index=False, cmap='Oranges')
#ax.set(ylabel='', aspect='equal')
plt.ylabel('')
plt.title('race 분포 비율')

plt.subplot(1,2,2)
data['class'].value_counts().plot.pie(autopct='%1.1f%%',
                                     startangle=90, fontsize=12,
                                     use_index=False, cmap='Oranges')
#ax.set(ylabel='', aspect='equal')
plt.ylabel('')
plt.title('class 분포 비율')
fig.savefig('F:/통계논문/image/piechart2_race+class.png')
plt.show()


fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
data['guild_ox'].value_counts().plot.pie(autopct='%1.1f%%',
                                         startangle=90, fontsize=12, use_index=False,
                                         colors=['sienna','peru'],
                                         labels=['길드 가입 고객', '길드 미가입 고객'])
#ax.set(ylabel='', aspect='equal')
plt.ylabel('')
plt.title('guild_ox 분포 비율')

plt.subplot(1,2,2)
tmp = pd.cut(data['char_nth'], bins=[0,1,2,10])
#fig, ax = plt.subplots()
tmp.value_counts().plot.pie(autopct='%1.1f%%',
                            fontsize=12,
                            labels=['첫번째', '두번째', '세번째 이상'],
                            colors=['sienna','peru','darkorange'])
#ax.set(ylabel='', aspect='equal')
plt.ylabel('')
plt.title('char_nth 분포 비율')
fig.savefig('F:/통계논문/image/piechart2_guild_ox+char_nth.png')
plt.show()



### 설명변수 그래프 - histogram ###
fig = plt.figure(figsize=(17,20))
for i, col in enumerate(cols_cont):
       plt.subplot(3, 3, i+1)
       plt.hist(data[col], color='sienna', bins=10)
       plt.title(str(col)+ ' 히스토그램')

fig.savefig('F:/통계논문/image/histogram2_contvar.png')
plt.show()



### 반응변수에 따른 변수 차이  - count plot ###

df1 = data.groupby('race')['y'].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
fig=plt.figure()
g = sns.catplot(x='race', y='percent', hue='y', kind='bar',
                data=df1, palette=['sienna','peru'])
g.ax.set_ylim(0,70)
plt.title('반응변수에 따른 race')
plt.show()

df1 = data.groupby('class')['y'].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
fig=plt.figure()
g = sns.catplot(x='class', y='percent', hue='y', kind='bar',
                data=df1, palette=['sienna','peru'])
g.ax.set_ylim(0,80)
plt.title('반응변수에 따른 class')
plt.show()


cols = ['race','class']
data['race_class'] = data[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
print(data['race_class'].unique())

df1 = data.groupby('race_class')['y'].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

fig=plt.figure(figsize=(20,10))
g = sns.catplot(y='race_class', x='percent', hue='y', kind='bar',
                data=df1, palette=['sienna','peru'])
#g.ax.set_ylim(0,80)
plt.title('반응변수에 따른 race와 class 조합')

#fig.savefig('F:/통계논문/image/countplot_class.png')
plt.show()



fig=plt.figure(figsize=(20,10))
sns.countplot(data=data, y='race_class', hue='y',
              palette=['sienna','peru'])
#fig.tight_layout()
plt.title('반응변수에 따른 race와 class 조합')
fig.savefig('F:/통계논문/image/countplot_race_class_y.png')
plt.show()


df1 = data.groupby('guild_ox')['y'].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

fig=plt.figure(figsize=(20,10))
g = sns.catplot(x='guild_ox', y='percent', hue='y', kind='bar',
                data=df1, palette=['sienna','peru'])
g.ax.set_ylim(0,100)
plt.title('반응변수에 따른 guild 가입 여부')
plt.show()


df1 = data.groupby('char_nth')['y'].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

fig=plt.figure(figsize=(20,10))
g = sns.catplot(x='char_nth', y='percent', hue='y', kind='bar',
                data=df1, palette=['sienna','peru'])
g.ax.set_ylim(0,100)
plt.title('반응변수에 따른 char 개수 분포')
plt.show()


fig, ax = plt.subplots()
sns.countplot(data=data, x='char_nth', hue='y',
              palette=['sienna','peru'])
plt.ylim(0,40)
plt.legend(loc='upper right')
fig.tight_layout()
fig.savefig('F:/통계논문/image/countplot_char_nth_y.png')
plt.show()



### 반응변수에 따른 변수 차이  - box plot ###
fig = plt.figure(figsize=(17,20))
for i, col in enumerate(cols_cont):
       plt.subplot(3, 3, i+1)
       sns.boxplot(x=data['y'], y=data[col], palette=['sienna','peru'])
       plt.title(str(col)+ ' 상자 그림')

fig.savefig('F:/통계논문/image/boxplot2_contvar.png')
plt.show()



# 그림 사이즈 지정
fig, ax = plt.subplots(figsize=(9,9))

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(data.iloc[:,3:].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
sns.heatmap(data.iloc[:,3:].corr(),
            cmap = 'RdYlBu_r',
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1,   # 컬러바 범위 -1 ~ 1
            annot_kws={"size": 12}
           )
plt.title('상관관계 heatmap')
fig.tight_layout()
fig.savefig('F:/통계논문/image/heatmap_corr.png')
plt.show()


"""fig=plt.figure(figsize=(15,15))
g = sns.pairplot(data, hue="y", markers=["o", "s"], palette=['orangered','saddlebrown'])
plt.title("Pair Plot")
fig.tight_layout()
plt.savefig('F:/통계논문/image/pairplot.png')
plt.show()"""

