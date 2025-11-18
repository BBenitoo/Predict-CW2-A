#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:05:54 2025

@author: a1
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
RNG = 42
np.random.seed(RNG)
# 1. Load dataset
# ---------------------------
cols = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight",
        "VisceraWeight", "ShellWeight", "Rings"]
df = pd.read_csv("abalone.data.csv", header=None, names=cols)

# Basic cleaning
df["Sex"] = df["Sex"].astype(str).str.strip() # force elements in 'sex' into string and eliminate spaces
for c in df.columns:
    if c != "Sex":
        df[c] = pd.to_numeric(df[c], errors="coerce") # force elements in all columns(except 'sex') into numerical values
df = df[df["Sex"].isin(["M", "F", "I"])] # eliminate rows containing error value in column 'sex'
df = df.dropna().reset_index(drop=True) # drop rows containing 'NaN' and reset index numbers

# 2. Construct target variable (Age)
# ---------------------------
df['Age'] = df['Rings'] + 1.5
df = df.drop(columns=['Rings'])

# 3. Define X / y
# ---------------------------
X = df.drop(columns=['Age'])  #features
y = df['Age'] # target

# 3.1 Feature engineering
def add_features(X_):
    X_ = X_.copy()
    eps = 1e-8

    # Use ellipsoid volume approximation instead of cylinder
    # Volume of ellipsoid: V = (4/3) * π * a * b * c
    # where a, b, c are semi-axes: Length/2, Diameter/2, Height/2
    vol = (4/3) * np.pi * (X_["Length"] / 2) * (X_["Diameter"] / 2) * (X_["Height"] / 2)
    vol = vol.replace(0, np.nan)

    X_["DensityWhole"] = X_["WholeWeight"] / (vol + eps)
    X_["MeatRatio"] = X_["ShuckedWeight"] / (X_["WholeWeight"] + eps)
    X_["VisceraRatio"] = X_["VisceraWeight"] / (X_["WholeWeight"] + eps)
    X_["ShellRatio"] = X_["ShellWeight"] / (X_["WholeWeight"] + eps)
    X_["Length_Diameter"] = X_["Length"] * X_["Diameter"]
    X_["Height_Shell"] = X_["Height"] * X_["ShellWeight"]

    X_ = X_.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_

X = add_features(X)

# 3.2 IQR outlier clipping - moved to ColumnTransformer to avoid data leakage
from sklearn.base import BaseEstimator, TransformerMixin

class IQRClipTransformer(BaseEstimator, TransformerMixin):
    """IQR-based outlier clipping transformer"""
    def __init__(self):
        self.low_ = None
        self.high_ = None
    
    def fit(self, X, y=None):
        q1 = np.quantile(X, 0.25, axis=0)
        q3 = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1               # iqr stands for the spread of the middle 50% of the data
        self.low_ = q1 - 1.5 * iqr
        self.high_ = q3 + 1.5 * iqr
        return self
    
    def transform(self, X):
        # Clip each column independently using broadcasting
        X_clipped = np.clip(X, self.low_, self.high_)
        return X_clipped

# 4. ColumnTransformer (unified preprocessing)
# ---------------------------
categorical_cols = ["Sex"]
numeric_cols = [c for c in X.columns if c != "Sex"]

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('num', Pipeline([
        ('iqr_clip', IQRClipTransformer()),
        ('scaler', StandardScaler())
    ]), numeric_cols)
])

# Full preprocessing pipeline
#full_preprocess = Pipeline([('ct', preprocess)])

# 5. Train/Test Split with stratification on binned Age
# ---------------------------
age_bins = pd.qcut(y, q=10, duplicates='drop') # 10 bins
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RNG,
    stratify=age_bins       #Stratified Sampling
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# 6. Cross-validation splitter for modelling
cv_bins = pd.qcut(y_train, q=5, duplicates='drop')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
pipe_rf = Pipeline([
    ('prep', preprocess),
    ('reg', RandomForestRegressor(random_state=42))
])
'''
# 更广泛的参数搜索
param_grid_rf = {
    'reg__n_estimators': [410],
    'reg__max_depth': [None, 10, 20, 30],  # None表示不限制深度，适合探索是否过拟合
    'reg__min_samples_split': [2, 5, 10],  # 更广泛探索分裂条件
    'reg__min_samples_leaf': [1, 2, 4],    # 包含更小叶子节点，适合中等数据量
    'reg__max_features': ['sqrt', 'log2', 0.5, 0.8],  # 增加 log2 和 0.5，探索更少特征时的表现
    'reg__criterion': ['squared_error', 'absolute_error']

}

grid_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    scoring='r2',
    cv=10,
    n_jobs=-1,
    verbose=2
)

start_time = time.time()
grid_rf.fit(X_train, y_train)
end_time = time.time()
rf_train_time = end_time - start_time

print("Best Parameters:", grid_rf.best_params_)
print("Best CV R²:", grid_rf.best_score_)
print("Training Time (s):", rf_train_time)

# 记录所有结果
cv_results = pd.DataFrame(grid_rf.cv_results_)
cv_results_sorted = cv_results.sort_values(by='mean_test_score', ascending=False)
print("\nTop 5 parameter combinations:")
print(cv_results_sorted[['params', 'mean_test_score']].head(5))


best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2  = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Test Performance ---")
print("MSE:", rf_mse)
print("MAE:", rf_mae)
print("R² :", rf_r2)
'''
# === 使用已知最佳参数构建最终模型 ===
final_rf = Pipeline([
    ('prep', preprocess),
    ('reg', RandomForestRegressor(
        n_estimators=410,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        criterion='squared_error',
        random_state=42,
        n_jobs=-1
    ))
])

start_time = time.time()
final_rf.fit(X_train, y_train)
rf_train_time = time.time() - start_time

y_pred_rf = final_rf.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2  = r2_score(y_test, y_pred_rf)

print("\n--- 最终随机森林模型性能 ---")
print("训练时间（秒）:", rf_train_time)
print("MSE:", rf_mse)
print("MAE:", rf_mae)
print("R² :", rf_r2)

# === 异常样本分析模块 ===
import seaborn as sns
from scipy.stats import zscore

# 1. 残差计算
residuals = y_test - y_pred_rf

# 2. 残差分布图
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=50, kde=True, color='steelblue')
plt.title("Residual Distribution")
plt.xlabel("Residual (True - Predicted)")
plt.grid(True)
plt.show()

# 3. 残差 vs 真实值散点图
plt.figure(figsize=(8,5))
plt.scatter(y_test, residuals, alpha=0.5, color='darkorange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Age")
plt.ylabel("Residual")
plt.title("Residuals vs True Age")
plt.grid(True)
plt.show()

# 4. 标准化残差检测异常点
z_resid = zscore(residuals)
outlier_mask = np.abs(z_resid) > 3
num_outliers = np.sum(outlier_mask)
print(f"\n异常样本数量（|Z-score| > 3）：{num_outliers}")

# 5. 打印异常样本的特征值（最多10个）
if num_outliers > 0:
    print("\n异常样本特征（Top 10）：")
    outlier_indices = np.argsort(np.abs(z_resid))[-10:]
    print(X_test.iloc[outlier_indices])

# 6. 分段 R² 分析
print("\n分段 R² 分析：")
bins = pd.qcut(y_test, q=4, duplicates='drop')
for b in bins.unique():
    mask = bins == b
    r2_local = r2_score(y_test[mask], y_pred_rf[mask])
    print(f"区间 {b} 的 R²：{r2_local:.3f}")


















