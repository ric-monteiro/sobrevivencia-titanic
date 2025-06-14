## Titanic Survival Classification with AdaBoost

This project analyzes historical data from the Titanic shipwreck to predict passenger survival based on characteristics such as gender, age, social class, among other factors. The **AdaBoost** algorithm was used with a decision tree as the base estimator, and the entire pipeline is built with **supervised machine learning** tools, including preprocessing, feature engineering, and performance evaluation.

-----

### About the Dataset

The data comes from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic):

  * `train.csv`: labeled data (with the `Survived` column)
  * `test.csv`: unlabeled data, used for prediction

**Relevant Attributes:**

  * `Pclass`: Passenger class (1st, 2nd, 3rd)
  * `Sex`: Passenger's gender
  * `Age`: Age
  * `SibSp`: Number of siblings/spouses aboard
  * `Parch`: Number of parents/children aboard
  * `Fare`: Ticket fare
  * `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

-----

### Project Steps

#### 1\. Data Loading and Combination

  * Training and test data are temporarily combined to facilitate preprocessing.
  * The target variable (`Survived`) is separated to be used only with the training data.

#### 2\. Handling Missing Values

  * Filling:
      * `Age` and `Fare`: with the median.
      * `Embarked`: with the most frequent value ('S').

#### 3\. Feature Engineering

  * Conversion of categorical variables to numerical with **OneHotEncoder** and **LabelEncoder**.
  * Normalization of numerical variables with **StandardScaler**.
  * Features created or reorganized:
      * `Title`: extracted from the name (Mr, Mrs, Miss, etc.)
      * `FamilySize`: sum of `SibSp` and `Parch` + 1
      * `IsAlone`: identifies passengers who traveled alone
      * Binning of age and fare into ranges

#### 4\. Pipeline Construction

  * Use of `Pipeline` and `ColumnTransformer` to organize the preprocessing and modeling workflow.
  * Application of an `AdaBoostClassifier` with `DecisionTreeClassifier` as the base estimator.
  * Hyperparameter tuning with `GridSearchCV`.

#### 5\. Model Evaluation

  * Accuracy on training/test data
  * Classification report (`classification_report`)
  * Confusion matrix
  * ROC curve and AUC calculation

-----

### Charts

The project includes some visualizations to help understand the data and the model's performance:

  * Bar Charts: Count of survivors by Sex and Pclass.
  ![alt text](/assets/image-2.png)

  * Histograms: Distribution of `Age` and `Fare` in relation to survival.
  ![Distribution of Age in Relation to Survival](/assets/image-3.png)
  ![Distribution of Fare in Relation to Survival](/assets/image-4.png)

  * Heatmap: Visual representation of the model's Confusion Matrix.
  ![alt text](/assets/image-5.png)
  
  * ROC Curve: Evaluates the model's discriminatory ability, with the AUC value.
  ![alt text](/assets/image-6.png)

  * Feature Importance Plot: Highlights the most influential features for the model's prediction.
  ![alt text](/assets/image-7.png)

-----

### Technologies Used

  * Python 3
  * Libraries:
      * `pandas`, `numpy` – data manipulation
      * `matplotlib`, `seaborn` – visualization
      * `scikit-learn` – machine learning

-----

### Expansion Ideas

  * Compare AdaBoost with other algorithms (Random Forest, XGBoost)
  * Use `SHAP` or `eli5` to explain the model's decisions
