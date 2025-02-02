# Fatores que Influenciam a Satisfação de Passageiros: Uma Abordagem com Árvore de Decisão

Este projeto pessoal utiliza aprendizado de máquina para prever a satisfação de clientes de uma companhia aérea com base em diversos fatores, como conforto do assento, entretenimento a bordo, pontualidade e outros. O objetivo é construir um modelo preditivo que possa auxiliar a companhia aérea a identificar áreas de melhoria e aumentar a satisfação do cliente.

## Conjunto de Dados

O projeto utiliza o conjunto de dados "Invistico_Airline.csv", que contém informações sobre as experiências de voo dos clientes e sua satisfação geral. O conjunto de dados inclui variáveis categóricas e numéricas.

![image](https://github.com/user-attachments/assets/dc4381f1-d0d2-4019-b829-30976e3a5b77)


## Pré-processamento de Dados

Os seguintes passos de pré-processamento foram realizados:

* **Transformação de Dados Categóricos:** Variáveis categóricas, como "satisfação", "gênero", "tipo de cliente", "tipo de viagem" e "classe", foram convertidas em representações numéricas usando mapeamento.

![image](https://github.com/user-attachments/assets/d6a7d72d-fac9-4ea3-89e7-cc34030899b2)


* **Tratamento de Valores Ausentes:** Valores ausentes na coluna "Atraso na Chegada em Minutos" foram preenchidos com 0.
* **Conversão de Tipos de Dados:** A coluna "Atraso na Chegada em Minutos" foi convertida para o tipo inteiro.

![image](https://github.com/user-attachments/assets/a17823e9-b953-4faa-b944-1201832d92fc)


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

![image](https://github.com/user-attachments/assets/0ae6d505-32dc-458e-9ec2-dfed05d33ba1)

![image](https://github.com/user-attachments/assets/87894992-5b1d-46a7-b921-7e2b108a4bd5)


O modelo final foi treinado utilizando um `Pipeline` com `StandardScaler` para padronizar os dados numéricos antes do treinamento do `DecisionTreeClassifier`.

## Resultado da Árvore de Decisão

![image](https://github.com/user-attachments/assets/563e54d7-e175-4d06-89db-75e13fd62129)


A acurácia do modelo final, `DecisionTreeClassifier` com `max_depth=3` e `StandardScaler`, foi de 85.50%. O notebook Jupyter incluído neste repositório detalha o processo de treinamento, avaliação e visualização da árvore de decisão.

![image](https://github.com/user-attachments/assets/12a3416e-8b66-4f40-906c-5e397aab782a)
