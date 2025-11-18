#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:50:21 2025

@author: a1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 09:48:47 2025

@author: a1
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import zscore

RNG = 42
np.random.seed(RNG)

# === Step 1: Load and clean dataset ===
cols = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight",
        "VisceraWeight", "ShellWeight", "Rings"]
df = pd.read_csv("abalone.data.csv", header=None, names=cols)
df["Sex"] = df["Sex"].astype(str).str.strip()
for c in df.columns:
    if c != "Sex":
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["Sex"].isin(["M", "F", "I"])]
df = df.dropna().reset_index(drop=True)

# === Step 2: Construct target variable (log-transformed Age) ===
df['Age'] = df['Rings'] + 1.5
df['LogAge'] = np.log1p(df['Age'])  # log(1 + Age)
df = df.drop(columns=['Rings'])

# === Step 3: Define X / y ===
X = df.drop(columns=['Age', 'LogAge'])
y = df['LogAge']  # log-transformed target

# === Step 3.1: Feature engineering ===
def add_features(X_):
    X_ = X_.copy()
    eps = 1e-8
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

# === Step 3.2: IQR outlier clipping transformer ===
class IQRClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.low_ = None
        self.high_ = None
    def fit(self, X, y=None):
        q1 = np.quantile(X, 0.25, axis=0)
        q3 = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.low_ = q1 - 1.5 * iqr
        self.high_ = q3 + 1.5 * iqr
        return self
    def transform(self, X):
        return np.clip(X, self.low_, self.high_)

# === Step 4: Preprocessing pipeline ===
categorical_cols = ["Sex"]
numeric_cols = [c for c in X.columns if c != "Sex"]

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('num', Pipeline([
        ('iqr_clip', IQRClipTransformer()),
        ('scaler', StandardScaler())
    ]), numeric_cols)
])

# === Step 5: Train/Test split ===
age_bins = pd.qcut(df['Age'], q=10, duplicates='drop')
X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
    X, y, df['Age'],
    test_size=0.2,
    random_state=RNG,
    stratify=age_bins
)


# 定义样本权重：线性提升高龄权重（也可用指数/分段）
age_train = y_train.copy()
age_min, age_max = age_train.min(), age_train.max()
w = 1.0 + 2.0*(age_train - age_min) / (age_max - age_min)  # 权重范围约 [1, 3]
# ...（前面部分保持不变，直到你定义完 final_rf_w 并输出全局加权训练性能）...

# === Step 6: 分段建模 + 段内加权优化 ===
bins = [0, 8, 12, np.inf]
labels = ["low", "mid", "high"]
df["AgeSegment"] = pd.cut(df["Age"], bins=bins, labels=labels)

segment_models = {}
segment_preds = []
segment_true = []

for label in labels:
    mask = df["AgeSegment"] == label
    X_seg = X[mask]
    y_seg = y[mask]
    age_seg = df.loc[mask, "Age"]

    X_train_seg, X_test_seg, y_train_seg, y_test_seg, age_train_seg, age_test_seg = train_test_split(
        X_seg, y_seg, age_seg,
        test_size=0.2,
        random_state=RNG,
        stratify=pd.qcut(age_seg, q=5, duplicates='drop')
    )

    # 段内权重策略
    seg_min, seg_max = age_train_seg.min(), age_train_seg.max()
    if label == "high":
        w_seg = 1.0 + 2.5 * (age_train_seg - seg_min) / (seg_max - seg_min)
        reg = RandomForestRegressor(
            n_estimators=120, max_depth=None, min_samples_split=8,
            min_samples_leaf=2, max_features='sqrt', criterion='squared_error',
            random_state=RNG, n_jobs=-1
        )
    elif label == "low":
        w_seg = 1.0 + 0.8 * (seg_max - age_train_seg) / (seg_max - seg_min)
        reg = RandomForestRegressor(
            n_estimators=40, max_depth=12, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt', criterion='squared_error',
            random_state=RNG, n_jobs=-1
        )
    else:
        w_seg = None
        reg = RandomForestRegressor(
            n_estimators=40, max_depth=None, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt', criterion='squared_error',
            random_state=RNG, n_jobs=-1
        )

    model = Pipeline([("prep", preprocess), ("reg", reg)])
    t0 = time.time()
    if w_seg is not None:
        model.fit(X_train_seg, y_train_seg, reg__sample_weight=w_seg.loc[y_train_seg.index])
    else:
        model.fit(X_train_seg, y_train_seg)
    train_time = time.time() - t0

    y_pred_seg = model.predict(X_test_seg)
    r2 = r2_score(y_test_seg, y_pred_seg)
    mae = mean_absolute_error(y_test_seg, y_pred_seg)
    mse = mean_squared_error(y_test_seg, y_pred_seg)

    print(f"\n--- Segment [{label}] ---")
    print("样本数：", len(X_seg))
    print("训练时间（秒）：", train_time)
    print("R²：", r2)
    print("MAE：", mae)
    print("MSE：", mse)

    segment_models[label] = model
    segment_preds.append(y_pred_seg)
    segment_true.append(y_test_seg)

# === Step 7: 合并所有预测结果，评估整体性能 ===
y_all = np.concatenate(segment_true)
y_pred_all = np.concatenate(segment_preds)

print("\n=== 分段建模整体性能（含段内加权优化） ===")
print("R²：", r2_score(y_all, y_pred_all))
print("MAE：", mean_absolute_error(y_all, y_pred_all))
print("MSE：", mean_squared_error(y_all, y_pred_all))

final_rf_w = Pipeline([
    ("prep", preprocess),
    ("reg", RandomForestRegressor(
        n_estimators=410,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features="sqrt",
        criterion="squared_error",
        random_state=RNG,
        n_jobs=-1
    ))
])

t0 = time.time()
final_rf_w.fit(X_train, y_train, reg__sample_weight=w)  # 关键：传入 sample_weight
train_time = time.time() - t0

y_pred_w = final_rf_w.predict(X_test)
print("\n--- RF + 高龄加权 性能 ---")
print("训练时间（秒）：", train_time)
print("MSE：", mean_squared_error(y_test, y_pred_w))
print("MAE：", mean_absolute_error(y_test, y_pred_w))
print("R²：", r2_score(y_test, y_pred_w))

