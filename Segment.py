#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:29:57 2025

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

RNG = 42
np.random.seed(RNG)

# === Step 1: 数据加载与清洗 ===
cols = ["Sex","Length","Diameter","Height","WholeWeight","ShuckedWeight","VisceraWeight","ShellWeight","Rings"]
df = pd.read_csv("abalone.data.csv", header=None, names=cols)
df["Sex"] = df["Sex"].astype(str).str.strip()
for c in df.columns:
    if c != "Sex": df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["Sex"].isin(["M","F","I"])].dropna().reset_index(drop=True)

# === Step 2: 构造目标变量 Age ===
df["Age"] = df["Rings"] + 1.5
X = df.drop(columns=["Rings", "Age"])
y = df["Age"]

# === Step 3: 特征工程 ===
def add_features(X_):
    X_ = X_.copy(); eps = 1e-8
    vol = (4/3)*np.pi*(X_["Length"]/2)*(X_["Diameter"]/2)*(X_["Height"]/2)
    vol = vol.replace(0, np.nan)
    X_["DensityWhole"] = X_["WholeWeight"]/(vol+eps)
    X_["MeatRatio"] = X_["ShuckedWeight"]/(X_["WholeWeight"]+eps)
    X_["VisceraRatio"] = X_["VisceraWeight"]/(X_["WholeWeight"]+eps)
    X_["ShellRatio"] = X_["ShellWeight"]/(X_["WholeWeight"]+eps)
    X_["Length_Diameter"] = X_["Length"]*X_["Diameter"]
    X_["Height_Shell"] = X_["Height"]*X_["ShellWeight"]
    return X_.replace([np.inf,-np.inf], np.nan).fillna(0.0)

X = add_features(X)

# === Step 4: IQR 裁剪预处理器 ===
class IQRClipTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        q1 = np.quantile(X, 0.25, axis=0); q3 = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.low_ = q1 - 1.5*iqr; self.high_ = q3 + 1.5*iqr
        return self
    def transform(self, X):
        return np.clip(X, self.low_, self.high_)

# === Step 5: 预处理管道 ===
categorical_cols = ["Sex"]
numeric_cols = [c for c in X.columns if c != "Sex"]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("num", Pipeline([("iqr_clip", IQRClipTransformer()), ("scaler", StandardScaler())]), numeric_cols)
])

# === Step 6: 分段建模逻辑 ===
# 分段定义（你可以自定义边界）
bins = [0, 8, 12, np.inf]
labels = ["low", "mid", "high"]
df["AgeSegment"] = pd.cut(df["Age"], bins=bins, labels=labels)

# 分段训练与预测
segment_models = {}
segment_preds = []
segment_true = []

for label in labels:
    mask = df["AgeSegment"] == label
    X_seg = X[mask]
    y_seg = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_seg, y_seg, test_size=0.2, random_state=RNG
    )

    model = Pipeline([
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
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n--- Segment [{label}] ---")
    print("样本数：", len(X_seg))
    print("训练时间（秒）：", train_time)
    print("R²：", r2)
    print("MAE：", mae)
    print("MSE：", mse)

    segment_models[label] = model
    segment_preds.append(y_pred)
    segment_true.append(y_test)

# === Step 7: 合并所有预测结果，评估整体性能 ===
y_all = np.concatenate(segment_true)
y_pred_all = np.concatenate(segment_preds)

print("\n=== 分段建模整体性能 ===")
print("R²：", r2_score(y_all, y_pred_all))
print("MAE：", mean_absolute_error(y_all, y_pred_all))
print("MSE：", mean_squared_error(y_all, y_pred_all))
