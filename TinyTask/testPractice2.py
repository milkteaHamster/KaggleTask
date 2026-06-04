import pandas as pd

df = pd.DataFrame({
    # 'gender': ['男', '女', '男', '女'],
     'city':   ['北京', '上海', '上海', '北京'],
    # 'click':  [1, 0, 1, 1]
})

# 年龄分桶
df['age'] = [22, 35, 28, 45]
df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 100], labels=['青年', '中年', '中老年'])

# 桶 × 城市 交叉
df['age_city'] = df['age_bin'].astype(str) + '_' + df['city']
print(df[['age', 'age_bin', 'city', 'age_city']])
