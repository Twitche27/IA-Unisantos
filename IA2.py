import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import collections

warnings.filterwarnings("ignore", message="The least populated class in y has only.*")

# =======================
# 1. Carregamento e limpeza
# =======================
df = pd.read_csv('PIB_taxa_desocup_gastos_publi_mort.csv', sep=';')

# Convertendo todos os valores para numéricos, tratando erros
for col in df.columns:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.').str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =======================
# 2. Features e targets
# =======================
X = df[['População Total Brasileira (Indivíduos)', 'PIB (R$)', 'Taxa de Desocupação (%)',
        'Gastos Públicos na Área da Saúde (R$) - Valor Pago',
        'Gastos Públicos na Área da Saúde (R$) - Valor Liquidado']].values

y_reg = df['Mortalidade - Óbitos (Individuos)'].values

# Classificação (baixo, médio, alto)
q1 = np.percentile(y_reg, 33)
q2 = np.percentile(y_reg, 66)

def classificar_mortalidade(valor):
    if valor <= q1:
        return 'baixo'
    elif valor <= q2:
        return 'medio'
    else:
        return 'alto'

y_clf = np.array([classificar_mortalidade(v) for v in y_reg])
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

# =======================
# 3. Normalização
# =======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =======================
# 4. Cross-validation - Classificação
# =======================
clf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
cv = StratifiedKFold(n_splits=10, shuffle=True)

scores = cross_val_score(clf_model, X_scaled, y_clf_encoded, cv=cv, scoring='accuracy')
formatted_scores = [f"{score*100:.3f}%" for score in scores]

print("✅ Acurácias nas 10 execuções (classificação):", formatted_scores)
print(f"🎯 Acurácia média: {np.mean(scores)*100:.4f}% | Desvio padrão: {np.std(scores):.4f}")

# =======================
# 5. Exibição de uma matriz de confusão final (última execução manual)
# =======================
# Pegando um dos splits para exibir a matriz de confusão
for train_idx, test_idx in cv.split(X_scaled, y_clf_encoded):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_clf_encoded[train_idx], y_clf_encoded[test_idx]
    break  # só o primeiro split para visualização

clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

# Confusion matrix
ordered_labels = ['baixo', 'medio', 'alto']
ordered_indices = le.transform(ordered_labels)
cm = confusion_matrix(y_test, y_pred, labels=ordered_indices)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_labels)
disp.plot(ax=ax, cmap='Blues')

plt.title("Matriz de Confusão - Classificação")
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")
plt.tight_layout()
plt.show()

# =======================
# 6. Regressão em único split (mesmo acima)
# =======================
reg_model = RandomForestRegressor(n_estimators=100)
reg_model.fit(X_train, y_reg[train_idx])
y_reg_pred = reg_model.predict(X_test)

mae = mean_absolute_error(y_reg[test_idx], y_reg_pred)
r2 = r2_score(y_reg[test_idx], y_reg_pred)

print(f'📉 Erro Médio Absoluto (MAE - regressão): {mae:.2f}')
print(f'📈 R² Score (regressão): {r2:.2f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_reg[test_idx], y_reg_pred, alpha=0.7)
plt.plot([min(y_reg[test_idx]), max(y_reg[test_idx])],
         [min(y_reg[test_idx]), max(y_reg[test_idx])], 'r--')
plt.xlabel("Valores Reais")
plt.ylabel("Preditos")
plt.title("Random Forest - Regressão (Óbitos)")
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# 7. Distribuição das classes
# =======================
print("\n📊 Distribuição das classes:")
print(collections.Counter(y_clf))

importances = clf_model.feature_importances_
for name, importance in zip(df.columns[1:6], importances):
    print(f'{name}: {importance:.3f}')
