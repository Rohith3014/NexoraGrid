import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error

# Test classifier
X = np.random.rand(100, 5)
y = np.random.randint(0, 6, size=100)
clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, eval_metric="mlogloss")
clf.fit(X, y)
preds = clf.predict(X)
print("XGBClassifier fit & predict success! Accuracy:", accuracy_score(y, preds))

# Test regressor
y_reg = np.random.rand(100) * 50
reg = xgb.XGBRegressor(n_estimators=10, max_depth=3)
reg.fit(X, y_reg)
reg_preds = reg.predict(X)
print("XGBRegressor fit & predict success! MSE:", mean_squared_error(y_reg, reg_preds))
