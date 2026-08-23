import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#1.数据准备和读取
cols = ['unit_number', 'time_in_cycles'] + \
       ['op_setting_' + str(i) for i in range(1, 4)] + \
       ['sensor_' + str(i) for i in range(1, 22)]
df = pd.read_csv("C:/Users/Peter/Desktop/Engine Health Analysis/data/train_FD001.txt", sep="\s+", header=None,names=cols)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
#2.数据探索（寿命，缺失值）
print(df.shape)
print(df.head(5))
print(df.columns.tolist())
print(df.isnull().sum())
life = df.groupby('unit_number')['time_in_cycles'].max()
print('最长寿命',life.max())
print('最短寿命',life.min())
print('平均寿命',life.mean())
print('中位数',life.median())
#发动机的循环次数分布
print(life.describe())
plt.hist(life, bins=20, color='steelblue', edgecolor='white')
plt.title('发动机寿命分布（循环数）', fontsize=25)
plt.xlabel('寿命（循环）', fontsize=20)
plt.ylabel('发动机数量', fontsize=20)
plt.savefig(r'C:\Users\Peter\Desktop\Engine Health Analysis\images\100个发动机寿命预览.png',dpi = 150)
plt.show()
# 3.单发动机探索（相关性 + 传感器趋势）
#相关性
engine1= df[df['unit_number'] == 1]
sensor = [f'sensor_{i}'for i in range(1,22)]
corr_matrix = engine1[sensor+['time_in_cycles']].corr()
corr_sensor_time = corr_matrix['time_in_cycles'].drop('time_in_cycles')
print(corr_sensor_time.sort_values())
#发动机传感器的总体趋势
fig,axes =plt.subplots(3,7,figsize=(28,9))
for i, ax in enumerate(axes.flat,start=1):
    sensor =f'sensor_{i}'
    x= engine1['time_in_cycles']
    y= engine1[sensor]
    ax.plot(x,y)
    ax.set_title(sensor)
plt.tight_layout()
plt.savefig(r'C:\Users\Peter\Desktop\Engine Health Analysis\images\发动机1各个传感器的参数.png',dpi = 150)
plt.show()
# 4.多发动机验证
engine1 = df[df['unit_number']==1]
engine5 = df[df['unit_number']==5]
engine10 = df[df['unit_number']==10]
engine20 = df[df['unit_number']==20]
four_engine = [engine1,engine5,engine10,engine20]
title_list = ['engine1','engine5','engine10','engine20']
fig,axes = plt.subplots(2,2,figsize=(12,10))
for i,ax in enumerate(axes.flat):
    engine = four_engine[i]
    x = engine['time_in_cycles']
    y11 = engine['sensor_11']

    ax.plot(x,y11,label='sensor_11')
    ax.set_title(title_list[i])
    ax.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\Peter\Desktop\Engine Health Analysis\images\多发动机sensor11的参数指标.png',dpi = 150)
plt.show()
fig,axes = plt.subplots(2,2,figsize=(12,10))
for i,ax in enumerate(axes.flat):
    engine = four_engine[i]
    x = engine['time_in_cycles']
    y11 = engine['sensor_12']

    ax.plot(x,y11,label='sensor_12')
    ax.set_title(title_list[i])
    ax.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\Peter\Desktop\Engine Health Analysis\images\多发动机sensor12的参数指标.png',dpi = 150)
plt.show()