import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score
from sklearn.utils.multiclass import unique_labels
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
    if v <= 49903: #valores definidos através da análise da média e desvio padrão
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
    X_scaled, y_reg, y_clf_encoded, test_size=0.2
)

# =======================
# 5. Modelos Random Forest
# =======================
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_clf_train)
y_clf_pred = clf_model.predict(X_test)

# Regressor
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

# =======================
# 6. Avaliação Classificação
# =======================

# Mostra a acurácia do código em relação aos dados de teste
acc = accuracy_score(y_clf_test, y_clf_pred)
print(f"\n✅ Acurácia da classificação: {acc * 100:.2f}%")


# Força as 3 classes (mesmo que alguma não apareça no conjunto de teste)
all_labels = np.array([0, 1, 2])  # Classes codificadas: alto, baixo, medio
cm = confusion_matrix(y_clf_test, y_clf_pred, labels=all_labels)

# Mostrando no máximo 20 amostras, ou o total disponível
n_to_show = min(20, len(y_clf_pred))

print("\n📊 Exibindo as primeiras predições:")
for i in range(n_to_show):
    pred = le.inverse_transform([y_clf_pred[i]])[0]
    real = le.inverse_transform([y_clf_test[i]])[0]
    print(f"Predição: {pred} | Real: {real}")

# Criando o gráfico com a matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
fig.canvas.manager.set_window_title("Matriz de Confusão - Classificação")

# Exibe todas as classes no eixo em ordem (mesmo que tenham valor 0)
ordered_labels = ['baixo', 'medio', 'alto']
ordered_indices = le.transform(ordered_labels)
cm_ordered = cm[np.ix_(ordered_indices, ordered_indices)]

disp = ConfusionMatrixDisplay(confusion_matrix=cm_ordered, display_labels=ordered_labels)
disp.plot(ax=ax, cmap='Blues')

plt.title("Matriz de Confusão - Classificação")
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")
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