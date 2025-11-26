import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from skopt import BayesSearchCV
import matplotlib.pyplot as plt

# === 1. Cargar el dataset ===
file_path = 'C:/Users/camil/OneDrive/Documentos/RandomForestDropOutUIS/dataset.csv'  
df = pd.read_csv(file_path)

# === 2. Variable objetivo ===
# Convertimos 'Target' a binaria: 1 = Dropout, 0 = Graduate (ignoramos 'Enrolled')
df = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
df['Target_Binary'] = (df['Target'] == 'Dropout').astype(int)

# === 3. One-hot encoding de variables categóricas ===
df_encoded = pd.get_dummies(df.drop(columns=['Target']), drop_first=True)

# Variables predictoras y target
X = df_encoded.drop(columns=['Target_Binary'])
y = df_encoded['Target_Binary']

# === 4. División train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 5. Escalado ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Optimización bayesiana de hiperparámetros ===
param_space = {
    'learning_rate': (0.01, 0.1, 'uniform'),
    'max_depth': (3, 7),
    'min_child_weight': (1, 5),
    'subsample': (0.7, 1.0),
    'colsample_bytree': (0.7, 1.0),
    'n_estimators': (100, 500),
    'scale_pos_weight': (1, 3)  # importante por el desbalance
}

xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss'  # mantiene la métrica
)

opt = BayesSearchCV(
    xgb_model, param_space, n_iter=50, cv=3, scoring='f1', 
    n_jobs=-1, random_state=42
)

opt.fit(X_train_scaled, y_train)
print(f"Mejores parámetros encontrados: {opt.best_params_}")

# === 7. Reentrenar con mejores parámetros ===
best_xgb_model = opt.best_estimator_

# === 8. Predicciones (probabilidades) ===
y_pred_prob_best_xgb = best_xgb_model.predict_proba(X_test_scaled)[:, 1]

# === 9. Búsqueda del mejor umbral para F1 ===
threshold_range = [i * 0.05 for i in range(1, 21)]
best_threshold = 0.5
best_f1_score = 0

for threshold in threshold_range:
    y_pred_adjusted = (y_pred_prob_best_xgb >= threshold).astype(int)
    current_f1 = f1_score(y_test, y_pred_adjusted)
    if current_f1 > best_f1_score:
        best_f1_score = current_f1
        best_threshold = threshold

# === 10. Evaluar con el mejor umbral ===
y_pred_adjusted_best = (y_pred_prob_best_xgb >= best_threshold).astype(int)
accuracy_best = accuracy_score(y_test, y_pred_adjusted_best)
classification_report_best = classification_report(y_test, y_pred_adjusted_best)

print(f"\nPrecisión con umbral ajustado: {accuracy_best:.4f}")
print(f"Mejor umbral: {best_threshold}")
print("\nReporte de clasificación:\n", classification_report_best)

# === 11. Estudiantes en riesgo de deserción con umbral ajustado ===
students_in_risk = X_test[y_pred_adjusted_best == 1]
print(f"\nNúmero de estudiantes en riesgo de deserción: {len(students_in_risk)}")

#Grafica:
#Grafica las lineas de precision, recall, f1-score
thresholds = [i * 0.05 for i in range(1, 20)]
precisions, recalls, f1_scores = [], [], []

for t in thresholds:
    y_pred_temp = (y_pred_prob_best_xgb >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_temp))
    recalls.append(recall_score(y_test, y_pred_temp))
    f1_scores.append(f1_score(y_test, y_pred_temp))

#Grafica:
plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions, label='Precisión', marker='o')
plt.plot(thresholds, recalls, label='Recall', marker='o')
plt.plot(thresholds, f1_scores, label='F1-score', marker='o')

plt.axvline(x=best_threshold, color='red', linestyle='--', 
            label=f'Mejor umbral = {best_threshold:.2f}')
plt.xlabel("Umbral de decisión")
plt.ylabel("Métrica")
plt.title("Efecto del umbral en el desempeño del modelo")
plt.legend()
plt.grid(True)
plt.show()
