import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RNG = 42
np.random.seed(RNG)

# 1. Load dataset
cols = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight",
        "VisceraWeight", "ShellWeight", "Rings"]
df = pd.read_csv("abalone.data.csv", header=None, names=cols)

# Basic cleaning
df["Sex"] = df["Sex"].astype(str).str.strip()
for c in df.columns:
    if c != "Sex":
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["Sex"].isin(["M", "F", "I"])]
df = df.dropna().reset_index(drop=True)

# 2. Construct target variable (Age)
df['Age'] = df['Rings'] + 1.5
df = df.drop(columns=['Rings'])

# 3. Define X / y
X = df.drop(columns=['Age'])  # features
y = df['Age']  # target

# 3.1 Feature engineering
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

# 3.2 IQR outlier clipping
class IQRClipTransformer(BaseEstimator, TransformerMixin):
    """IQR-based outlier clipping transformer"""
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
        X_clipped = np.clip(X, self.low_, self.high_)
        return X_clipped

# 4. ColumnTransformer (unified preprocessing)
categorical_cols = ["Sex"]
numeric_cols = [c for c in X.columns if c != "Sex"]
numeric_pipe = Pipeline([
    ('iqr_clip', IQRClipTransformer()),
    ('scaler', StandardScaler())
])

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('num', numeric_pipe, numeric_cols)
])

# 5. Train/Test Split with stratification on binned Age
age_bins = pd.qcut(y, q=10, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG, stratify=age_bins
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# 6. Cross-validation splitter
cv_bins = pd.qcut(y_train, q=5, duplicates='drop')
cv_bins_codes = cv_bins.cat.codes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

# Ridge regression pipeline
pipe_ridge = Pipeline([
    ('prep', preprocess),
    ('reg', Ridge())
])

param_grid_ridge = {
    'reg__alpha': np.logspace(-4, 3, 30),   
    'reg__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
    'reg__fit_intercept': [True, False],
    'reg__tol': [1e-3, 1e-4, 1e-5]
}

# GridSearch
grid_ridge = GridSearchCV(
    estimator=pipe_ridge,
    param_grid=param_grid_ridge,
    scoring='r2',
    cv=cv.split(X_train, cv_bins_codes),
    n_jobs=-1
)

# Training
start_train = time.time()
grid_ridge.fit(X_train, y_train)
end_train = time.time()
print(f"Training Time: {end_train - start_train:.4f} seconds")

best_ridge = grid_ridge.best_estimator_
print("Best Parameters:", grid_ridge.best_params_)
print("Best CV R²:", grid_ridge.best_score_)

# Prediction
start_pred = time.time()
y_pred_ridge = best_ridge.predict(X_test)
end_pred = time.time()
print(f"Prediction Time: {end_pred - start_pred:.4f} seconds")

# Coefficients
cat_features = best_ridge.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
num_features = numeric_cols
feature_names = np.concatenate([cat_features, num_features])

coefficients = best_ridge.named_steps['reg'].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print("\nRidge Coefficient Interpretation:")
print(coef_df)

# Test Evaluation
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)

print("\n--- Ridge Test Performance ---")
print(f"R²  : {ridge_r2:.4f}")
print(f"MSE : {ridge_mse:.4f}")
print(f"RMSE: {ridge_rmse:.4f}")
print(f"MAE : {ridge_mae:.4f}")

# Scatter Plot
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_ridge, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age (Ridge Regression)")
plt.grid(alpha=0.3)
plt.show()

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    best_ridge,
    X_train,
    y_train,
    cv=cv.split(X_train, cv_bins_codes),
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(9, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label="Training R²")
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label="Validation R²")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.title("Learning Curve (Ridge Regression)")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

cv_scores = cross_val_score(
    best_ridge, X_train, y_train,
    cv=cv.split(X_train, cv_bins_codes),
    scoring='r2'
)

print("\n5-Fold CV R² Scores:", np.round(cv_scores, 4))
print("Mean CV R²:", cv_scores.mean().round(4))
