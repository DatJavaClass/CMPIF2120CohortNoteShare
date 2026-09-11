import matplotlib, numpy as np, pandas as pd, seaborn, sklearn, xgboost as xgb

X=np.array([[0.0], [1.0], [2.0], [3.0]])
y=np.array([0, 0, 1, 1])
model=xgb.XGBClassifier(device="cpu", n_estimators=2, tree_method="hist")
model.fit(X, y)
predictions=model.predict(X)
assert len(predictions) == len(y)
print(f"PyPrimePortable verified with XGBoost {xgb.__version__} on CPU")
