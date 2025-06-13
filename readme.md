## Classificação de Sobrevivência no Titanic com AdaBoost

Este projeto analisa os dados históricos do naufrágio do Titanic para prever a sobrevivência de passageiros, com base em características como sexo, idade, classe social, entre outros fatores. Foi utilizado o algoritmo **AdaBoost** com uma árvore de decisão como estimador base e todo o pipeline é construído com ferramentas de **machine learning supervisionado**, incluindo pré-processamento, engenharia de atributos e avaliação de desempenho.

---

### Sobre o Conjunto de Dados

Os dados são provenientes da [competição Titanic do Kaggle](https://www.kaggle.com/competitions/titanic):

* `train.csv`: dados rotulados (com a coluna `Survived`)
* `test.csv`: dados não rotulados, usados para previsão

**Atributos relevantes:**

* `Pclass`: Classe do passageiro (1ª, 2ª, 3ª)
* `Sex`: Sexo do passageiro
* `Age`: Idade
* `SibSp`: Número de irmãos/cônjuges a bordo
* `Parch`: Número de pais/filhos a bordo
* `Fare`: Valor da passagem
* `Embarked`: Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

---

### Etapas do Projeto

#### 1. Carregamento e Combinação dos Dados

* Os dados de treino e teste são combinados temporariamente para facilitar o pré-processamento.
* A variável alvo (`Survived`) é separada para ser usada apenas nos dados de treino.

#### 2. Tratamento de Valores Ausentes

* Preenchimento de:

  * `Age` e `Fare`: com a mediana.
  * `Embarked`: com o valor mais frequente ('S').

#### 3. Engenharia de Features

* Conversão de variáveis categóricas em numéricas com **OneHotEncoder** e **LabelEncoder**.
* Normalização de variáveis numéricas com **StandardScaler**.
* Features criadas ou reorganizadas:

  * `Title`: título extraído do nome (Mr, Mrs, Miss, etc.)
  * `FamilySize`: soma de `SibSp` e `Parch` + 1
  * `IsAlone`: identifica passageiros que viajaram sozinhos
  * Binagem de idade e tarifa em faixas

#### 4. Construção do Pipeline

* Utilização do `Pipeline` e `ColumnTransformer` para organizar o fluxo de pré-processamento e modelagem.
* Aplicação de um `AdaBoostClassifier` com `DecisionTreeClassifier` como estimador base.
* Ajuste de hiperparâmetros com `GridSearchCV`.

#### 5. Avaliação do Modelo

* Acurácia nos dados de treino/teste
* Relatório de classificação (`classification_report`)
* Matriz de confusão
* Curva ROC e cálculo da AUC

---

### Visualizações Chave

O projeto inclui várias visualizações para entender os dados e o desempenho do modelo:

* Gráficos de Barras: Contagem de sobreviventes por Sex e Pclass.

![alt text](/assets/image-2.png)
* Histogramas: Distribuição de `Age` e `Fare` em relação à sobrevivência.

![Distribuição de Idade por Status de Sobrevivência](/assets/image-3.png)
![Distribuição de Tarifa por Status de Sobrevivência](/assets/image-4.png)
* Mapa de Calor: Representação visual da Matriz de Confusão do modelo.

![alt text](/assets/image-5.png)
* Curva ROC: Avalia a capacidade discriminatória do modelo, com o valor AUC.

![alt text](/assets/image-6.png)
* Gráfico de Importância das Features: Destaca as features mais influentes para a previsão do modelo.

![alt text](/assets/image-7.png)

---

### Tecnologias Utilizadas

* Python 3
* Bibliotecas:

  * `pandas`, `numpy` – manipulação de dados
  * `matplotlib`, `seaborn` – visualização
  * `scikit-learn` – machine learning

---

### Ideias de Expansão

* Comparar AdaBoost com outros algoritmos (Random Forest, XGBoost)
* Utilizar `SHAP` ou `eli5` para explicar as decisões do modelo
