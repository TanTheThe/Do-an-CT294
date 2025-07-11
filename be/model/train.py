import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('abalone/abalone.data', header=None)

df.columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]

df = pd.get_dummies(df, columns=["Sex"])

X = df.drop("Rings", axis=1)
y = df["Rings"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_configs = {
    "decision_tree": {
        "model": DecisionTreeRegressor(),
        "params": {
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10]
        }
    },
    "knn": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ['uniform', 'distance']
        }
    },
    "random_forest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    }
}

results = {}
trained_models = {}
metrics_dict = {}
os.makedirs('model', exist_ok=True)
all_best_params = {}

for name, config in model_configs.items():
    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_scaled, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    all_best_params[name] = best_params

    mse_scores, mae_scores, r2_scores = [], [], []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)

    metrics_dict[name] = {
        "MSE": round(avg_mse, 4),
        "MAE": round(avg_mae, 4),
        "R2": round(avg_r2, 4)
    }

    results[name] = avg_mse
    trained_models[name] = best_model
    print(f"Model: {name} - MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")

best_model_name = min(results, key=results.get)
best_model = trained_models[best_model_name]
print(f"\nModel tốt nhất là '{best_model_name}' với MSE trung bình = {results[best_model_name]:.4f}")

joblib.dump(best_model, 'model/best_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

with open('model/best_params_all.json', 'w') as f:
    json.dump(all_best_params, f, indent=4)

with open('model/best_model_name.txt', 'w') as f:
    f.write(best_model_name)

with open('model/model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

df_plot = pd.DataFrame(metrics_dict).T
df_plot.plot(kind='bar', figsize=(12, 6))
plt.title("Comparison of Model Performance")
plt.xlabel("Models")
plt.ylabel("Scores")
plt.legend(["MSE", "MAE", "R2"])
plt.tight_layout()
plt.savefig("model/model_comparison.png")
