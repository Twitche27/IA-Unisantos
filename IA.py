import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import collections
import os

import tensorflow as tf
tf.config.run_functions_eagerly(True)

# Função para criar e compilar o modelo
def criar_modelo():
    entrada = Input(shape=(5,))
    x = Dense(32, activation='relu')(entrada)
    x = Dense(16, activation='relu')(x)

    saida_reg = Dense(1, name='regressao')(x)
    saida_clf = Dense(3, activation='softmax', name='classificacao')(x)

    modelo = Model(inputs=entrada, outputs=[saida_reg, saida_clf])

    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'regressao': MeanSquaredError(),
            'classificacao': 'categorical_crossentropy'
        },
        loss_weights={
            'regressao': 0.5,
            'classificacao': 1.0
        },
        metrics={
            'regressao': 'mae',
            'classificacao': 'accuracy'
        }
    )
    return modelo

# Carrega o CSV
df = pd.read_csv('PIB_taxa_desocup_gastos_publi_mort.csv', sep=';')

# Limpa as colunas numéricas que vieram como string
for col in df.columns:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.').str.strip()

# Converte todas as colunas para float
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define as features (entradas)
X = df[['População Total Brasileira (Indivíduos)', 'PIB (R$)', 'Taxa de Desocupação (%)',
        'Gastos Públicos na Área da Saúde (R$) - Valor Pago',
        'Gastos Públicos na Área da Saúde (R$) - Valor Liquidado']].values

# Target para regressão
y_regressao = df['Mortalidade - Óbitos (Individuos)'].values

# Classifica a mortalidade em "baixo", "médio" e "alto" usando tercis
q1 = np.percentile(y_regressao, 33)
q2 = np.percentile(y_regressao, 66)

def classificar_mortalidade(valor):
    if valor <= q1:
        return 'baixo'
    elif valor <= q2:
        return 'medio'
    else:
        return 'alto'

# Cria os labels classificativos
y_classificacao = np.array([classificar_mortalidade(v) for v in y_regressao])

# One-hot encoding das classes
le = LabelEncoder()
y_class_int = le.fit_transform(y_classificacao)  # Converte para inteiros 0,1,2
y_classificacao_oh = to_categorical(y_class_int, num_classes=3)

# Normaliza as features (z-score)
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# Divide os dados em treino e teste
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_normalizado, y_regressao, y_classificacao_oh, test_size=0.2, random_state=42
)

# Tenta carregar o modelo salvo, se existir
modelo = criar_modelo()
if os.path.exists('modelo_completo.weights.h5'):
    modelo.load_weights('modelo_completo.weights.h5')
    print("Pesos carregados no modelo.")
else:
    print("Treinando modelo do zero.")


# Callback para EarlyStopping (para evitar overfitting)
early_stop = EarlyStopping(
    monitor='val_classificacao_accuracy',
    patience=250,
    restore_best_weights=True,
    mode='max'  # Aqui ajustado para maximizar a acurácia
)

# Treinamento
history = modelo.fit(
    X_train,
    {'regressao': y_reg_train, 'classificacao': y_clf_train},
    epochs=500,
    batch_size=25,
    validation_split=0.1,
    callbacks=[early_stop]
)

# Salva o modelo após treino
modelo.save_weights("modelo_completo.weights.h5")

# Avaliação do modelo
resultados = modelo.evaluate(X_test, {'regressao': y_reg_test, 'classificacao': y_clf_test})
classe = history.history['classificacao_accuracy']
val_classe = history.history['val_classificacao_accuracy']
mae_treino = history.history['regressao_mae']

print(f'Média geral da acurácia no treino: {((sum(classe)/ len(classe))*100):.2f}%')  
print(f'Média geral da acurácia na validação: {((sum(val_classe)/ len(val_classe))*100):.2f}%')  
print(f'Média geral do Erro Médio Absoluto (MAE) no treino: {((sum(mae_treino)/ len(mae_treino))):.2f}%')  

# Contagem das classes para ver balanceamento
print("Distribuição das classes:")
print(collections.Counter(y_classificacao))

import matplotlib.pyplot as plt

plt.plot(history.history['classificacao_accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_classificacao_accuracy'], label='Acurácia Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
