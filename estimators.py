#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
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

# === Step 2: 构造目标变量 ===
df["Age"] = df["Rings"] + 1.5
df["LogAge"] = np.log1p(df["Age"])
X = df.drop(columns=["Age","LogAge","Rings"])
y = df["LogAge"]

# === Step 3: 特征工程 ===
def add_features(X_):
    X_ = X_.copy(); eps = 1e-8
    vol = (4/3)*np.pi*(X_["Length"]/2)*(X_["Diameter"]/2)*(X_["Height"]/2)
    vol = vol.replace(0, np.nan)
    X_["DensityWhole"] = X_["WholeWeight"]/(vol+eps)
    X_["MeatRatio"] = X_["ShuckedWeight"]/(X_["WholeWeight"]+eps)
    X_["VisceraRatio"] = X_["VisceraWeight"]/(X_["WholeWeight"]+eps)
    X_["ShellRatio"] = X_["ShellWeight"]/(X_["WholeWeight"]+eps)
    return X_.replace([np.inf,-np.inf], np.nan).fillna(0.0)

X = add_features(X)

# === Step 4: IQR裁剪预处理器 ===
class IQRClipTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        q1 = np.quantile(X, 0.25, axis=0); q3 = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.low_ = q1 - 1.5*iqr; self.high_ = q3 + 1.5*iqr
        return self
    def transform(self, X): return np.clip(X, self.low_, self.high_)

# === Step 5: 预处理管道 ===
categorical_cols = ["Sex"]
numeric_cols = [c for c in X.columns if c != "Sex"]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("num", Pipeline([("iqr_clip", IQRClipTransformer()), ("scaler", StandardScaler())]), numeric_cols)
])

X_prepared = preprocess.fit_transform(X)

# === Step 6: OOB误差 vs n_estimators ===
n_list = list(range(50, 1001, 50))
oob_errors = []

for n in n_list:
    rf = RandomForestRegressor(
        n_estimators=n,
        oob_score=True,
        bootstrap=True,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        criterion="squared_error",
        random_state=RNG,
        n_jobs=-1
    )
    rf.fit(X_prepared, y)
    oob_r2 = rf.oob_score_
    oob_errors.append(1 - oob_r2)  # 越小越好

# === Step 7: 绘图 ===
plt.figure(figsize=(10,6))
plt.plot(n_list, oob_errors, marker='o', linestyle='-')
plt.xlabel("n_estimators")
plt.ylabel("1 - OOB R² (OOB误差)")
plt.title("OOB Error vs n_estimators (RandomForestRegressor)")
plt.grid(True)
plt.tight_layout()
plt.show()
