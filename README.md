# Fatores que Influenciam a Satisfação de Passageiros: Uma Abordagem com Árvore de Decisão

Este projeto pessoal utiliza aprendizado de máquina para prever a satisfação de clientes de uma companhia aérea com base em diversos fatores, como conforto do assento, entretenimento a bordo, pontualidade e outros. O objetivo é construir um modelo preditivo que possa auxiliar a companhia aérea a identificar áreas de melhoria e aumentar a satisfação do cliente.

## Conjunto de Dados

O projeto utiliza o conjunto de dados "Invistico_Airline.csv", que contém informações sobre as experiências de voo dos clientes e sua satisfação geral. O conjunto de dados inclui variáveis categóricas e numéricas.
[Dados](https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline)

![image](https://github.com/user-attachments/assets/dc4381f1-d0d2-4019-b829-30976e3a5b77)


## Pré-processamento de Dados

Os seguintes passos de pré-processamento foram realizados:

* **Transformação de Dados Categóricos:** Variáveis categóricas, como "satisfação", "gênero", "tipo de cliente", "tipo de viagem" e "classe", foram convertidas em representações numéricas usando mapeamento.

```python
dados['satisfaction'] = dados['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0}).astype('int64')
dados['Gender'] = dados['Gender'].map({'Female': 1, 'Male': 0}).astype('int64')
dados['Customer Type'] = dados['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0}).astype('int64')
dados['Type of Travel'] = dados['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0}).astype('int64')
dados['Class'] = dados['Class'].map({'Business': 1, 'Eco Plus': 2, 'Eco': 0}).astype('int64')
```


* **Tratamento de Valores Ausentes:** Valores ausentes na coluna "Atraso na Chegada em Minutos" foram preenchidos com 0.
* **Conversão de Tipos de Dados:** A coluna "Atraso na Chegada em Minutos" foi convertida para o tipo inteiro.
```python
dados['Arrival Delay in Minutes'].fillna(0, inplace=True)
dados['Arrival Delay in Minutes'] = dados['Arrival Delay in Minutes'].astype('int64')
```

## Modelagem

Diversos modelos de classificação e de regressão foram testados, incluindo:

* **Regressão Linear:** Um modelo de regressão linear foi usado como linha de base.
* **LinearSVC:**  Support Vector Classifier com kernel linear.
* **SGDClassifier:** Stochastic Gradient Descent Classifier.
* **DummyClassifier:** Classificador Dummy para comparação.
* **DecisionTreeClassifier:** Árvore de Decisão.
* **ExtraTreeClassifier:**  Árvore de Decisão Extra.
* **ExtraTreeRegressor:** Regressor de Árvore Extra.

A melhor performance foi obtida com o modelo `DecisionTreeClassifier` (e `ExtraTreeClassifier`), que apresentou uma acurácia de aproximadamente 85.50% (e 90.71%).  Uma árvore de decisão com `max_depth=3` foi escolhida para o modelo final por oferecer um bom equilíbrio entre acurácia e interpretabilidade.  A árvore de decisão com `max_depth=18` (ExtraTreeClassifier), embora mais precisa, resultou em uma árvore muito complexa, dificultando sua análise.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def f(x):
  return int(x)

f2 = np.vectorize(f)

x = dados.drop(['satisfaction', 'Customer Type'], axis=1)
y = dados['satisfaction']

modelo = DecisionTreeClassifier(max_depth=3)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, random_state=43)
modelo.fit(treino_x, treino_y)
prev = modelo.predict(teste_x)

print(f"A acurácia do modelo DecisionTreeClassifier é: {((accuracy_score(teste_y, f2(prev.round())))*100):.2f}%")
```

```python
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def f(x):
  return int(x)

f2 = np.vectorize(f)

x = dados.drop(['satisfaction', 'Customer Type'], axis=1)
y = dados['satisfaction']

modelo = ExtraTreeClassifier(max_depth=18)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, random_state=43)
modelo.fit(treino_x, treino_y)
prev = modelo.predict(teste_x)

print(f"A acurácia do modelo ExtraTreeClassifier é: {((accuracy_score(teste_y, f2(prev.round())))*100):.2f}%")
```

## Resultado da Árvore de Decisão

![image](https://github.com/user-attachments/assets/563e54d7-e175-4d06-89db-75e13fd62129)

O modelo final foi treinado utilizando um `Pipeline` com `StandardScaler` para padronizar os dados numéricos antes do treinamento do `DecisionTreeClassifier`.
A acurácia do modelo final, `DecisionTreeClassifier` com `max_depth=3` e `StandardScaler`, foi de 85.50%. O notebook Jupyter incluído neste repositório detalha o processo de treinamento, avaliação e visualização da árvore de decisão.
Apesar do StandardScaler ser útil para padronizar os dados e melhorar a eficácia do modelo, não o utilizei para essa árvore, até porque não faria sentido ter esses dados padronizados na minha árvore de decisão. 


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def f(x):
  return int(x)

f2 = np.vectorize(f)

x = dados.drop(['satisfaction', 'Customer Type'], axis=1)
y = dados['satisfaction']

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, random_state=43)

modelo = Pipeline([
    ('scaler', StandardScaler()),
    ('tree', DecisionTreeClassifier(max_depth=3))
])

modelo.fit(treino_x, treino_y)
prev = modelo.predict(teste_x)

print(f"A acurácia do modelo DecisionTreeClassifier é: {((accuracy_score(teste_y, f2(prev.round())))*100):.2f}%")
```
