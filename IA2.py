import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import collections

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
def classificar_mortalidade(v):
    if v <= 49903:
        return 'baixo'
    elif v <= 76881:
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
# 4. Divisão treino/teste
# =======================
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_scaled, y_reg, y_clf_encoded, test_size=0.2, random_state=42
)

# =======================
# 5. Modelos Random Forest
# =======================
class_weights = {0: 2, 1: 1, 2: 1}  # A classe 'baixo' recebe maior peso
clf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
clf_model.fit(X_train, y_clf_train)
y_clf_pred = clf_model.predict(X_test)

# Regressor
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

# =======================
# 6. Avaliação Classificação
# =======================
acc = accuracy_score(y_clf_test, y_clf_pred)
cm = confusion_matrix(y_clf_test, y_clf_pred)

# Exibindo a matriz de confusão
print(f'Matriz de Confusão:\n{cm}')

# Exibir as primeiras 20 predições e suas classes reais
print("\n📊 Exibindo as primeiras 20 predições:")
for i in range(20):
    print(f"Predição: {le.inverse_transform([y_clf_pred[i]])[0]} | Real: {le.inverse_transform([y_clf_test[i]])[0]}")

# Criando o gráfico com a matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))

# Ajuste manual dos ticks
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

# Forçar 3 posições de ticks
ax.set_xticks([0, 1, 2])  # Para 3 classes: baixo, médio, alto
ax.set_yticks([0, 1, 2])  # Para 3 classes: baixo, médio, alto

# Ajuste dos labels de acordo com as classes
ax.set_xticklabels(le.classes_)
ax.set_yticklabels(le.classes_)

# Plotando a matriz de confusão
disp.plot(ax=ax, cmap='Blues')

# Ajustando título e labels
plt.title("Matriz de Confusão - Classificação")
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")

# Exibir o gráfico
plt.tight_layout()
plt.show()

# =======================
# 7. Avaliação Regressão
# =======================
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f'📉 Erro Médio Absoluto (MAE - regressão): {mae:.2f}')
print(f'📈 R² Score (regressão): {r2:.2f}')

# =======================
# 8. Gráfico Regressão
# =======================
plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.7)
plt.plot([min(y_reg_test), max(y_reg_test)], [min(y_reg_test), max(y_reg_test)], 'r--')
plt.xlabel("Valores Reais")
plt.ylabel("Preditos")
plt.title("Random Forest - Regressão (Óbitos)")
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# 9. Distribuição das classes
# =======================
print("\n📊 Distribuição das classes:")
print(collections.Counter(y_clf))
