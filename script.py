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
dataset_df.columns = ['ID', 'RI', 'NA', 'MG', 'AL', 'SI', 'K', 'CA', 'BA', 'FE', 'class']

# Número de linhas e colunas
print(f'dataset_df shape: {dataset_df.shape}')

# Limpando dados
dataset_df = dataset_df.dropna()

# Número de linhas e colunas
print(f'dataset_df shape: {dataset_df.shape}')

# Embaralhando os dados
#dataset_df = dataset_df.sample(frac=1)

# Imprimindo datataset
print(f'dataset_df:\r\n {dataset_df}')

# -------------------------------------------------------------------------
# Separação dos conjuntos de dados
# Train = 60% | Temp Test = 40%
train_dataset, temp_test_dataset = train_test_split(dataset_df, test_size=0.4)
print(f'train_dataset shape: {train_dataset.shape}')
print(f'temp_test_dataset shape: {temp_test_dataset.shape}')
# Test = 50% | Valid  = 50%
test_dataset, valid_dataset = train_test_split(temp_test_dataset, test_size=0.5)
print(f'test_dataset shape: {test_dataset.shape}')
print(f'valid_dataset shape: {valid_dataset.shape}')

# -------------------------------------------------------------------------
# Gráfico de Relações
train_stats = train_dataset.describe()
train_stats.pop('class')
sns.pairplot(train_stats[train_stats.columns], diag_kind='kde')
plt.show()
train_stats = train_stats.transpose()
print(f'train_stats:\r\n {train_stats}')

# Definindo Labels
train_labels = train_dataset.pop('class')
test_labels = test_dataset.pop('class')
valid_labels = valid_dataset.pop('class')

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

def smv_model_run_kernel(kernel, C=1):    
    # Definindo o modo de funcionamento do modelo
    model = svm.SVC(
        C=C, # Termo de regularização
        kernel=kernel, # Kernels possíveis: linear, poly, rbf, sigmoid, precomputed
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
    ax.set_title(f"Confusion Matrix - Kernel ({kernel})")
    predict_results = model.predict(normed_test_data)
    cm = confusion_matrix(predict_results, predict_results)
    sns.heatmap(cm, annot=True, ax=ax)
    plt.show()

smv_model_run_kernel('poly')
smv_model_run_kernel('rbf')
smv_model_run_kernel('sigmoid')
smv_model_run_kernel('linear')

# -------------------------------------------------------------------------
dt = dataset_df #pd.read_csv('glass.data')
dt = dt.drop('ID', 1)

print(dt['class'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dt['class'], color='g', bins=100, hist_kws={'alpha': 0.4})

dt.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

df_num_corr = dt.corr()['class'][:-1]
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Class:\n{}".format(len(golden_features_list), golden_features_list))

for i in range(0, len(dt.columns), 5):
    sns.pairplot(
        data=dt,
        x_vars=dt.columns[i:i+5],
        y_vars=['class']
    )

corr = dt.drop('class', axis=1).corr() # We already examined class correlations
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr[(corr >= 0.5) | (corr <= -0.4)], 
    cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
    annot=True, annot_kws={"size": 8}, square=True
)

dt.describe(percentiles=[0.5])

norm_min_max = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
dt = dt.copy()
dt.iloc[:,0:9] = dt.iloc[:,0:9].apply(norm_min_max, axis=0)
dt.describe(percentiles=[0.5])

# RepeatedStratifiedKFold

y = dt['class']
X = dt.drop('class', 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

for train_index, test_index in rskf.split(X_train, y_train):
  X_train_k, X_test_k = X_train.iloc[train_index], X_train.iloc[test_index]
  y_train_k, y_test_k = y_train.iloc[train_index], y_train.iloc[test_index]

  model_poly = svm.SVC(
    C=1, # Termo de regularização
    kernel='poly', # Kernels possíveis: linear, poly, rbf, sigmoid, precomputed
  )
  model_rbf = svm.SVC(
    C=1, # Termo de regularização
    kernel='rbf', # Kernels possíveis: linear, poly, rbf, sigmoid, precomputed
  )
  model_sigmoid = svm.SVC(
    C=1, # Termo de regularização
    kernel='sigmoid', # Kernels possíveis: linear, poly, rbf, sigmoid, precomputed
  )
  model_precomputed = svm.SVC(
    C=1, # Termo de regularização
    kernel='linear', # Kernels possíveis: linear, poly, rbf, sigmoid, precomputed
  )

  model_poly.fit(X_train_k, y_train_k)
  model_rbf.fit(X_train_k, y_train_k)
  model_sigmoid.fit(X_train_k, y_train_k)
  model_precomputed.fit(X_train_k, y_train_k)

  # Validação do modelo com base nos dados de t
  poly_val = model_poly.score(X_test_k, y_test_k)
  rbf_val = model_rbf.score(X_test_k, y_test_k)
  sigmoid_val = model_sigmoid.score(X_test_k, y_test_k)
  precomputed_val = model_precomputed.score(X_test_k, y_test_k)

poly_test = model_poly.score(X_test, y_test)
rbf_test = model_rbf.score(X_test, y_test)
sigmoid_test = model_sigmoid.score(X_test, y_test)
precomputed_test = model_precomputed.score(X_test, y_test)
