# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.数据加载和列命名
cols = ['unit_number', 'time_in_cycles'] + \
       ['op_setting_' + str(i) for i in range(1, 4)] + \
       ['sensor_' + str(i) for i in range(1, 22)]

df = pd.read_csv("C:/Users/Peter/Desktop/Engine Health Analysis/data/train_FD001.txt",
                 sep="\s+", header=None, names=cols)
#2.构造标签（RUL）
df['RUL'] = df.groupby('unit_number')['time_in_cycles'].transform('max')-df['time_in_cycles']
df['RUL'] = df['RUL'].clip(upper=125)  #设置上限
print(df['RUL'].describe())
#3.划分特征（X）和标签（y）
X = df[[f'sensor_{i}'for i in range(1, 22)]]
y = df['RUL']
# 4.划分训练集和测试集
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)
# 5.模型训练
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
# 6.预测和评估
pred = model.predict(X_test)
print(type(pred))
RMSE = mean_squared_error(y_test,pred)**0.5
print(RMSE)
bias=(pred-y_test).mean()
print(bias)
#预测和真实值的对比散点图
plt.figure(figsize=(8,8))
plt.scatter(y_test,pred)
plt.plot([0,125],[0,125],'r--',label='理想线y=x')
plt.xlabel('RUL', fontsize=14)
plt.ylabel('Predicted RUL', fontsize=14)
plt.title('预测值和真实值的对比散点图', fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('C:/Users/Peter/Desktop/Engine Health Analysis/images/rul_pred_vs_true.png', dpi=150)
plt.show()
# 预测模型在低 RUL（接近失效）区间表现更准，高 RUL 区间存在系统性高估，但对维护决策影响有限——低 RUL 的准确预警才是实际需求。
#model是已经训练完成的随机森林实例对象
importance = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
print(importance.head(10))
#特征重要性的柱状图
importance.head(10).plot(kind='bar', figsize=(8,8),color='steelblue')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Feature Importance', fontsize=24)
plt.xlabel('传感器', fontsize=14)
plt.ylabel('重要性', fontsize=14)
plt.tight_layout()
plt.savefig('C:/Users/Peter/Desktop/Engine Health Analysis/images/feature_importance.png', dpi=150)
plt.show()
# sensor_11（高压压气机出口静压相关）对 RUL 预测贡献最大（56%），与相关性分析一致；sensor_7 虽与循环强相关，但信息被 sensor_11 覆盖，特征重要性较低。


