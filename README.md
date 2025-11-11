# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries for data manipulation, model building, evaluation, and visualization.
2.Load the car price dataset and perform preprocessing:
3.Drop irrelevant columns.
4.Convert categorical variables into dummy/indicator variables.
5.Separate the dataset into features (X) and target variable (y).
6.Standardize the features and target using StandardScaler.
7.Split the dataset into training and testing sets using train_test_split.
8.Define Ridge, Lasso, and ElasticNet regression models with polynomial feature transformation using Pipeline.
9.Train each model on the training data.
10.Predict the car prices using the trained models on the testing data.
11.Evaluate the model performance using Mean Squared Error (MSE) and R² score.
12.Visualize the performance metrics for each model using bar plots.
## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
```
```
data = pd.read_csv("/content/encoded_car_data (1) (1).csv")
data.head()
```
```
data = pd.get_dummies(data, drop_first=True)
```

```
X = data.drop('price', axis=1)
y = data['price']

```

```

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

```
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}
```

```
results = {}

```
```
for name, model in models.items():
    # Create a pipeline with polynomial features and the model
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Store results
    results[name] = {'MSE': mse, 'R² Score': r2}

```

```
for model_name, metrics in results.items():
    print(f"{model_name} Mean Squared Error: {metrics['MSE']:.2f}, R² Score: {metrics['R² Score']:.2f}")

```

```

results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
```

```
plt.figure(figsize=(12,5))

```

```
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)

```

```
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

```

```

plt.tight_layout()
plt.show()

```
## Output:

LOAD THE DATASET

<img width="1255" height="315" alt="image" src="https://github.com/user-attachments/assets/2fa7104a-0bac-47ee-8f66-1c10f4d2b3e3" />


PRINT RESULTS


<img width="734" height="81" alt="image" src="https://github.com/user-attachments/assets/f59f725b-e5b7-4cf3-85e2-aa92641050fb" />


FIGURE SIZE



<img width="817" height="64" alt="image" src="https://github.com/user-attachments/assets/5b557e61-6605-4f6e-b64d-c5f77a518b51" />


Bar plot for MSE


<img width="1017" height="714" alt="image" src="https://github.com/user-attachments/assets/b64fcbaf-c7ec-4703-a8db-f3068560c150" />



Bar plot for R² Score


<img width="1276" height="746" alt="image" src="https://github.com/user-attachments/assets/38675380-110e-4d75-ae73-0711a71b655d" />



PLOTS


<img width="854" height="53" alt="image" src="https://github.com/user-attachments/assets/351ca17d-1a46-4047-bb4d-2d8899545bbf" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
