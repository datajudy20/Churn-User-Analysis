"""
7단계
이탈 유저와 잔존 유저 간에 설명변수 평균 차이 통계 검정
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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
x = data.loc[data['y'] == 0, :].copy()
y = data.loc[data['y'] == 1, :].copy()


class Testcont:

    def __init__(self, var):
        self.var = var

    def shap(self):
        shap_x = stats.shapiro(x[self.var])
        shap_y = stats.shapiro(y[self.var])


        if (shap_x.pvalue >= 0.05) & (shap_y.pvalue >= 0.05):
            print(str(self.var) + '변수는 정규성을 만족한다')
            self.levene()
        else:
            print(str(self.var) + '변수는 정규성을 만족하지 않는다')
            self.mann()

    def mann(self):
        test = stats.mannwhitneyu(x[self.var], y[self.var])

        if test.pvalue < 0.05:
            print('유의수준 0.05 하에서 ' + str(self.var) + '변수는 두 집단 사이에서 순위합이 같지 않다  (p-value = ' + str(test.pvalue) + ')')

        else:
            print(str(self.var) + '변수는 두 집단 사이에서 순위합이 같다  (p-value = ' + str(test.pvalue) + ')', end='\n\n\n')
            print('-' * 80)

    def levene(self):
        test = stats.bartlett(x[self.var], y[self.var])
        print('< 등분산성 검정 levene test >', end='\n  ')
        if test.pvalue < 0.05:
            print(str(self.var) + '변수는 두 집단 사이에서 분산이 같지 않다 => 이분산')
            self.Ttest(False)
        else:
            print(str(self.var) + '변수는 두 집단 사이에서 분산이 같다 => 등분산')
            self.Ttest(True)

    def Ttest(self, tf):
        test = stats.ttest_ind(x[self.var], y[self.var], equal_var=tf)
        print('< 평균 차이 검정 t test >', end='\n  ')
        if test.pvalue < 0.05:
            print('유의수준 0.05 하에서 ' + str(self.var) + '변수는 두 집단 사이에서 평균에 차이가 있다  (p-value = ' + str(test.pvalue) + ')')
            self.boxplot_var(0.05)
        else:
            print(str(self.var) + '변수는 두 집단 사이에서 평균에 차이가 없다  (p-value = ' + str(test.pvalue) + ')', end='\n\n\n')
            print('-' * 80)

    def boxplot_var(self, alpha):
        plt.figure(figsize=(5, 7))
        plt.boxplot((x[self.var].to_numpy(), y[self.var].to_numpy()),
                    sym='o', labels=['y = 0', 'y = 1'], meanline=True, showmeans=True)
        plt.title('{0} boxplot)'.format(self.var))
        plt.show()
        #print('\n\n')
        print('-' * 80)


col_plus = ['guild_change', 'guild_level', 'play_day7_mean', 'play_day5_mean',
            'play_day2_mean', 'play_sec7_mean', 'play_sec5_mean', 'play_sec2_mean', 'orgrimmar']
for col in col_plus:
    Testcont(col).levene()


mann = stats.mannwhitneyu(x['char_nth'], y['char_nth'])
print(mann)
if mann.pvalue < 0.05:
    print('유의수준 0.05 하에서 char_nth 변수는 두 집단 사이에서 순위합이 같지 않다  (p-value = ' + str(mann.pvalue) + ')')
else:
    print('char_nth 변수는 두 집단 사이에서 순위합이 같다  (p-value = ' + str(mann.pvalue) + ')')

