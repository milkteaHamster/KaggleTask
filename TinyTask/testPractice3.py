import pandas as pd
import numpy as np

# ============================================================
# Spaceship Titanic - Kaggle 入门竞赛
# 任务: 预测乘客是否被传送到另一个维度 (二分类)
# ============================================================

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ========== 1. 数据概览 ==========
print("=== 训练集形状 ===")
print(train.shape)
print("\n=== 列信息 ===")
print(train.info())
print("\n=== 前5行 ===")
print(train.head())
print("\n=== 目标变量分布 ===")
print(train['Transported'].value_counts(normalize=True))

# ========== 2. 缺失值分析 ==========
print("\n=== 缺失值比例 ===")
for col in train.columns:
    miss = train[col].isna().sum()
    if miss > 0:
        print(f"{col}: {miss} ({miss/len(train):.2%})")

# ========== 3. Cabin 拆分解码 (类似练习2的分桶交叉) ==========
# Cabin格式: "deck/num/side" 如 "B/0/P"
train[['Deck', 'CabinNum', 'Side']] = train['Cabin'].str.split('/', expand=True)
test[['Deck', 'CabinNum', 'Side']] = test['Cabin'].str.split('/', expand=True)

print("\n=== Deck 分布 ===")
print(train['Deck'].value_counts())
print("\n=== Side 分布 ===")
print(train['Side'].value_counts())

# ========== 4. PassengerId 拆分 (组内编号) ==========
train[['GroupId', 'GroupNum']] = train['PassengerId'].str.split('_', expand=True)
test[['GroupId', 'GroupNum']] = test['PassengerId'].str.split('_', expand=True)

# 每组人数
group_size = train['GroupId'].value_counts()
print("\n=== 组人数分布 (前10) ===")
print(group_size.head(10))

# ========== 5. Name 拆分 ==========
train['LastName'] = train['Name'].str.split().str[-1]
test['LastName'] = test['Name'].str.split().str[-1]

# ========== 6. 特征工程: CryoSleep 冷冬眠乘客消费应为0 ==========
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    fill_mask = (train[col].isna()) & (train['CryoSleep'] == True)
    train.loc[fill_mask, col] = 0

# ========== 7. 填充缺失值 ==========
# 数值型用中位数填充
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

# 类别型用众数填充
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

# ========== 8. 交叉特征: Deck × Side × HomePlanet ==========
train['Deck_Side'] = train['Deck'].astype(str) + '_' + train['Side'].astype(str)
test['Deck_Side'] = test['Deck'].astype(str) + '_' + test['Side'].astype(str)

train['Home_Deck'] = train['HomePlanet'].astype(str) + '_' + train['Deck'].astype(str)
test['Home_Deck'] = test['HomePlanet'].astype(str) + '_' + test['Deck'].astype(str)

print("\n=== Deck_Side 交叉分布 (前10) ===")
print(train['Deck_Side'].value_counts().head(10))

# ========== 9. 总消费特征 ==========
train['TotalSpend'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']
test['TotalSpend'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']

# ========== 10. CryoSleep 与消费的关系 ==========
print("\n=== CryoSleep vs 平均消费 ===")
print(train.groupby('CryoSleep')['TotalSpend'].mean())

# ========== 11. 编码为数值 ==========
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['Transported'] = le.fit_transform(train['Transported'])  # True->1, False->0

binary_cols = ['CryoSleep', 'VIP']
for col in binary_cols:
    train[col] = train[col].astype(bool).astype(int)
    test[col] = test[col].astype(bool).astype(int)

# ========== 12. 多模型对比 ==========
feature_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
                'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'Deck', 'Side', 'TotalSpend']

train_encoded = pd.get_dummies(train[feature_cols])
test_encoded = pd.get_dummies(test[feature_cols])
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               HistGradientBoostingClassifier, VotingClassifier)

X, y = train_encoded, train['Transported']

# 定义候选模型
models = {
    'LogisticRegression':    LogisticRegression(max_iter=1000, random_state=42),
    'KNN (k=5)':             KNeighborsClassifier(n_neighbors=5),
    'SVM (rbf)':             SVC(kernel='rbf', random_state=42),
    'DecisionTree':          DecisionTreeClassifier(max_depth=10, random_state=42),
    'RandomForest':          RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting':      GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'HistGradientBoosting':  HistGradientBoostingClassifier(max_iter=200, random_state=42),
}

# 逐个评估
print("\n=== 多模型 5折交叉验证准确率对比 ===\n")
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = (scores.mean(), scores.std())
    print(f"  {name:<24s}  {scores.mean():.4f}  (+/- {scores.std():.4f})")

# 找出最佳模型
best_name = max(results, key=lambda k: results[k][0])
print(f"\n最佳: {best_name} = {results[best_name][0]:.4f}")

# ========== 13. 软投票集成 (Voting Ensemble) ==========
print("\n=== 软投票集成 (Soft Voting) ===")
voting = VotingClassifier(estimators=[
    ('rf',   models['RandomForest']),
    ('gb',   models['GradientBoosting']),
    ('hgb',  models['HistGradientBoosting']),
], voting='soft')

scores = cross_val_score(voting, X, y, cv=5, scoring='accuracy')
print(f"  Voting (RF+GB+HGB):     {scores.mean():.4f}  (+/- {scores.std():.4f})")

# ========== 14. 模型选型小结 ==========
print("""
=== 模型选型小结 ===
┌──────────────────────────┬──────────────────────────────────┐
│ 模型                      │ 适用场景                          │
├──────────────────────────┼──────────────────────────────────┤
│ LogisticRegression       │ 基线模型，可解释性强               │
│ KNN                      │ 小数据、局部模式                    │
│ SVM (rbf)                │ 高维数据、非线性边界                │
│ DecisionTree             │ 可解释、特征重要性                  │
│ RandomForest             │ 鲁棒、抗过拟合、特征多             │
│ GradientBoosting         │ 精度高、逐步纠错                   │
│ HistGradientBoosting     │ 速度更快、大规模数据               │
│ Voting (Soft)            │ 取长补短、降低方差                 │
└──────────────────────────┴──────────────────────────────────┘
""")

print("\n训练完成！")
