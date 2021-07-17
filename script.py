"""
https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/modules/multiclass.html
https://scikit-learn.org/stable/modules/preprocessing.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html
https://archive.ics.uci.edu/ml/datasets/glass+identification
https://www.youtube.com/watch?v=Zj1CoJk2feE
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

# Lendo dados
dataset_df = pd.read_csv('glass.data')
dataset_df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

# Número de linhas e colunas
print(f'dataset_df LxC = {dataset_df.shape}')

# Limpando dados
dataset_df = dataset_df.dropna()

# Número de linhas e colunas
print(f'dataset_df LxC = {dataset_df.shape}')

# Embaralhando os dados
#dataset_df = dataset_df.sample(frac=1)
#print(dataset_df)

# -------------------------------------------------------------------------
# Separação dos conjuntos de dados
# Train = 60% | Temp Test = 40%
train_dataset, temp_test_dataset = train_test_split(dataset_df, test_size=0.4)
print(f'train_dataset LxC = {train_dataset.shape}')
print(f'temp_test_dataset LxC = {temp_test_dataset.shape}')
# Test = 50% | Valid  = 50%
test_dataset, valid_dataset = train_test_split(temp_test_dataset, test_size=0.5)
print(f'test_dataset LxC = {test_dataset.shape}')
print(f'valid_dataset LxC = {valid_dataset.shape}')

# -------------------------------------------------------------------------
# Gráfico de Relações
train_stats = train_dataset.describe()
train_stats.pop('k')
#sns.pairplot(train_stats[train_stats.columns], diag_kind='kde')
#plt.show()
train_stats = train_stats.transpose()
#print(train_stats)

# Definindo Labels
train_labels = train_dataset.pop('k')
test_labels = test_dataset.pop('k')
valid_labels = valid_dataset.pop('k')

# -------------------------------------------------------------------------
# Normalização dos dados
norm = lambda x: (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_valid_data = norm(valid_dataset)
#print(normed_train_data.head(10))
#print(normed_test_data.head(10))
#print(normed_valid_data.head(10))

# -------------------------------------------------------------------------
# Treinando Modelo

# Definindo o modo de funcionamento do modelo
model = svm.SVC(
    C=1, # Termo de regularização
    kernel='linear', # Kernels possíveis: linear, poly, rbf, sigmoid, precomputed
)

# Inserindo os dados de treinamento no modelo
model.fit(normed_train_data, train_labels)

# Predição do modelo com base nos dados de test
y_pred = model.predict(normed_test_data)
#print(y_pred)
example_batch = normed_test_data[:10]
example_result = model.predict(example_batch)
print(f"Predict values:\r\n{example_batch}")

# -------------------------------------------------------------------------
# Avaliando a precisão dos conjuntos aplicados ao modelo

# Precisão da Predição do modelo com base nos dados de train
y_pred = model.predict(normed_train_data)
acc_train = metrics.accuracy_score(train_labels, y_pred)
print(f"Accuracy Train: {acc_train}")

# Precisão da Predição do modelo com base nos dados de valid
y_pred = model.predict(normed_valid_data)
acc_valid = metrics.accuracy_score(valid_labels, y_pred)
print(f"Accuracy Valid: {acc_valid}")

# Precisão da Predição do modelo com base nos dados de test
y_pred = model.predict(normed_test_data)
acc_test = metrics.accuracy_score(test_labels, y_pred)
print(f"Accuracy Test: {acc_test}")

# Exibindo a matriz de confusão
ax = plt.subplot()
ax.set_title("Confusion Matrix")
predict_results = model.predict(normed_test_data)
cm = confusion_matrix(predict_results, predict_results)
sns.heatmap(cm, annot=True, ax=ax)
plt.show()
