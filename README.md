# Predictive Analytics for Marketing Expansion: Sales Forecasting and Customer Segmentation

## Introduction

In today's competitive business landscape, leveraging data to make informed decisions is crucial for marketing success. This article outlines the process of using predictive analytics to enhance marketing strategies through sales forecasting and customer segmentation. By following these steps, businesses can optimize their marketing efforts, target the right customer segments, and ultimately drive higher profitability.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Predictive Modeling](#predictive-modeling)
6. [Model Evaluation](#model-evaluation)
7. [Actionable Marketing Insights](#actionable-marketing-insights)
8. [Implementation and Deployment](#implementation-and-deployment)
9. [Conclusion](#conclusion)

# Data Preparation

### Importing Necessary Libraries

First, import the necessary libraries for data analysis and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
```

## Load dataset using full path
Load the dataset and inspect the first few rows to understand its structure.

```python
data = pd.read_csv('sales_data.csv')
```

## Display the first few rows of the dataset

```python
data.head()
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/ba4ade09-2909-4b12-bf48-071857926230)

## Checking for missing values
Check for and handle missing values to ensure a clean dataset.

```python
data.isnull().sum()
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/1e65218a-948b-4971-9ece-66859591fe37)


## Drop rows with missing values (if any)

```python
data.dropna(inplace=True)
```

## Convert categorical variables to numerical

```python
data = pd.get_dummies(data, drop_first=True)
```
# Data Transformation

Convert categorical variables into numerical ones for machine learning models.

```python
data = pd.get_dummies(data, drop_first=True)
```

## Replace infinite values with NaN

```python
data.replace([np.inf, -np.inf], np.nan, inplace=True)
```

## Summary statistics

```python
print(data.describe())
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/e75cd3af-dc01-45e6-b7c5-53c132b39af4)

## Key Observations
- Customer Age: The average customer age is 36, with a spread from 17 to 87 years old. This wide range indicates a diverse customer base in terms of age.
- Order Quantity: On average, customers order about 15 units, with most orders ranging from 8 to 22 units.
- Unit Cost and Price: There is a significant difference between the mean unit cost (6.63) and unit price (16.41), indicating potential for profit.
- Profit: The average profit per transaction is 104.45, but there is high variability (std: 143.16), with some transactions even resulting in a loss (min: -5).
- Cost and Revenue: The average cost and revenue per transaction are 86.51 and 190.96, respectively, showing that revenue is typically more than twice the cost.

## Visualize the distribution of 'Profit'

```python
plt.figure(figsize=(10, 6))
sns.histplot(data['Profit'], kde=True)
plt.title('Distribution of Profit')
plt.show()
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/ab739967-53d6-4fa1-9d30-7a1589415e39)

- Skewed Distribution: The distribution of profit is highly right-skewed, meaning that most of the profit values are concentrated towards the lower end, with a long tail extending to the right.
- Majority of Profits: The majority of the profit values are low, with a high frequency around the lower profit range. This is evident from the peak on the left side of the plot.

## Implications for Marketing

- Focus on High-Value Customers: Since the majority of profits are low, but there are few high-profit instances, targeting high-value customers with tailored marketing strategies could be beneficial.
- Product Pricing and Promotion: Evaluate the products or services that contribute to higher profit margins and consider focusing promotional efforts on these.



## Calculate the correlation matrix

```python
corr_matrix = numerical_data.corr()
```

## Create a correlation heatmap for the subset of features

```python
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
```
![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/4df14bd7-22b0-447f-b5d7-c15843e6dae9)


## Key Observations from the Correlation Heatmap

1. Strong Positive Correlations:
- Unit Cost and Unit Price: There is a very strong positive correlation (0.99) between unit cost and unit price, indicating that as the cost of the unit increases, the price tends to increase proportionally.
- Profit and Revenue: A strong positive correlation (0.98) exists between profit and revenue, meaning that higher revenue generally leads to higher profit.
- Cost and Revenue: There is also a strong positive correlation (0.97) between cost and revenue, suggesting that higher costs are associated with higher revenues.
- Profit and Cost: A strong positive correlation (0.91) between profit and cost indicates that higher costs can lead to higher profits, likely because more expensive products are priced higher and thus generate more profit.
  
2. Moderate Positive Correlations:

- Order Quantity and Revenue: A moderate positive correlation (0.36) between order quantity and revenue suggests that larger orders generally lead to higher revenue.
- Order Quantity and Profit: The correlation between order quantity and profit is (0.32), indicating that larger orders contribute to higher profits.
- Customer Age and Profit: A slight positive correlation (0.05) between customer age and profit might indicate that older customers tend to generate more profit.

3. Negative Correlations:

- Order Quantity and Unit Cost/Unit Price: Negative correlations (-0.16 and -0.17) between order quantity and both unit cost and unit price suggest that larger orders might be associated with lower per-unit costs and prices, possibly due to bulk purchase discounts.

  ## Pairplot for detailed EDA (only a subset for clarity)
  
```python
sns.pairplot(data[['Customer_Age', 'Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Profit']])
plt.show()
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/039acb46-62cb-4753-8103-c10d984843cb)


## Analysis of Pairplot

1. Customer Age vs. Other Variables:
- Customer Age vs. Profit: There is a noticeable spread, indicating variability in profit across different ages. Some older customers seem to generate higher profits.
- Customer Age vs. Order Quantity: No strong pattern is observed, suggesting that order quantity is fairly distributed across different ages.

2. Order Quantity vs. Other Variables:
- Order Quantity vs. Profit: There is a positive relationship, indicating that higher order quantities tend to result in higher profits.
- Order Quantity vs. Unit Cost and Order Quantity vs. Unit Price: There are clustered patterns that might indicate certain price or cost brackets are more common for different order quantities.

3. Unit Cost vs. Other Variables:
- Unit Cost vs. Profit: Higher unit costs tend to correlate with higher profits, though the relationship is not linear.
- Unit Cost vs. Unit Price: There is a strong positive linear relationship, indicating that higher unit costs are associated with higher unit prices.

4. Unit Price vs. Other Variables:
- Unit Price vs. Profit: There is a positive relationship where higher unit prices generally lead to higher profits.
- Unit Price vs. Order Quantity: The relationship shows distinct clusters, suggesting that certain price ranges are more common for specific order quantities.

5. Profit vs. Other Variables:
- Profit vs. Customer Age: As mentioned, older customers tend to generate higher profits.
- Profit vs. Order Quantity: Higher order quantities are associated with higher profits.
- Profit vs. Unit Cost and Unit Price: Both show positive relationships with profit, indicating that more expensive items (both in cost and price) tend to yield higher profits.

## Feature Engineering: Create new features
Create new features to enhance the dataset for predictive modeling.

```python
data['Profit_Margin'] = data['Profit'] / data['Revenue']
data['Revenue_Per_Unit'] = data['Revenue'] / data['Order_Quantity']
data['Cost_Per_Unit'] = data['Cost'] / data['Order_Quantity']
```

# Predictive Modeling

Selecting Features and Target Variable
Choose the relevant features and target variable for modeling.

## Select relevant features for the model

```python
features = ['Customer_Age', 'Order_Quantity', 'Unit_Cost', 'Unit_Price',
            'Profit_Margin', 'Revenue_Per_Unit', 'Cost_Per_Unit']
target = 'Profit'
```
## Display the first few rows of the prepared dataset

```python
print(data[features + [target]].head())
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/135ce7c9-7a00-4ac6-8be8-033d538cc6d1)


## Split data into training and testing sets

```python
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Function to evaluate models

```python
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2
```

## Initialize models
Training Models
Train multiple regression models to predict profit.

```python
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}
```


## Evaluate models

```python
results = {}
for name, model in models.items():
    mae, mse, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
```


## Display results

```python
results_df = pd.DataFrame(results).T
print(results_df)
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/1064bae9-082b-4fd2-8a2c-8f4b1f5e1493)


# Interpretation of Model Results Linear Regression:

- MAE: 49.116 MSE: 6512.703 R²: 0.658 Interpretation: Linear Regression has a relatively higher error and lower R² compared to the other models, indicating it might not be capturing the complexity of the data well. Random Forest:

- MAE: 0.319 MSE: 7.474 R²: 0.9996 Interpretation: Random Forest shows excellent performance with very low error values and an R² close to 1, indicating it fits the data extremely well. Gradient Boosting:

- MAE: 5.119 MSE: 76.435 R²: 0.996 Interpretation: Gradient Boosting also performs well but with slightly higher errors compared to Random Forest. Its R² value is also very high, indicating a good fit. Given the superior performance of the Random Forest model, we will proceed with this model for further analysis and deployment.


# Let's take a closer look at the feature importance for the Random Forest model to understand which features are contributing the most to the predictions.

## Feature importance for Random Forest

```python
rf_model = models['Random Forest']
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
```

## Plot feature importance

```python
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices])
plt.title('Feature Importances (Random Forest)')
plt.show()
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/446d293f-2819-4d3a-86ab-32425d104a5e)

Interpretation of Feature Importance

Based on the feature importance plot from the Random Forest model, we can see which features are contributing the most to the model's predictions of profitability.

**Revenue_Per_Unit:**

This is the most important feature in the model.
It indicates the revenue generated per unit sold, which makes sense as a key driver of profit.

**Order_Quantity:**

This is the second most important feature.
The number of units ordered directly impacts the total sales and, consequently, the profit.

**Other Features:**

Unit_Cost, Profit_Margin, Cost_Per_Unit, Unit_Price, Customer_Age have less impact on the model compared to the top two features.

These features still contribute to the model but to a lesser extent.
Recommendations

**Given the importance of these features, here are some strategic recommendations:**

- Focus on Increasing Revenue Per Unit:
  

Implement pricing strategies to maximize revenue per unit sold.
Consider upselling and cross-selling tactics to increase the average revenue per sale.

- Optimize Order Quantities:

Analyze and optimize inventory and sales strategies to increase order quantities.

- Implement promotions and discounts strategically to encourage bulk purchases.
Monitor and Control Costs:

Keep an eye on unit costs and cost per unit to ensure profitability is maintained.

Negotiate better terms with suppliers to reduce costs without compromising quality.

- Customer Segmentation and Targeting:

Although customer age is not a major factor, understanding the demographics of high-value customers can help tailor marketing strategies.

Use insights from profit margin and unit price to segment customers and target them with personalized offers.


# Saving the Model for Deployment

This part of the code is for saving the trained Random Forest model to a file. This file can later be loaded for making predictions without having to retrain the model.

import joblib

## Save the best performing model (Random Forest) for deployment

```python
joblib.dump(rf_model, 'market_expansion_model.pkl')
```

## Loading the Model for Predictions
This part of the code demonstrates how to load the saved model and use it to make predictions. This is what you would do in a production environment where you need to make predictions on new data.

```python
rf_model = joblib.load('market_expansion_model.pkl')
```


## Make predictions on the test set

```python
y_pred = rf_model.predict(X_test)
```

## Evaluate the predictions

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

```python
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
```
![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/ade8df3f-7d31-4fb9-91fa-5e6dad6e664e)

## Plot actual vs. predicted values

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()
```

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/4dee355b-6eec-4284-b691-ee4a7b51e894)


# Implment the prediction on New Data we Can choose and write down

## Example new data for prediction
new_data = pd.DataFrame({
    'Customer_Age': [35, 45, 50],
    'Order_Quantity': [10, 20, 30],
    'Unit_Cost': [50, 40, 60],
    'Unit_Price': [120, 150, 200],
    'Profit_Margin': [0.2, 0.3, 0.25],
    'Revenue_Per_Unit': [150, 180, 210],
    'Cost_Per_Unit': [40, 35, 45]
})


## Predict the profit for the new data
predicted_profit = rf_model.predict(new_data)
new_data['Predicted_Profit'] = predicted_profit

print(new_data)

![image](https://github.com/Hagar-zakaria/Predictive-Analytics-for-Marketing-Expansion-Sales-Forecasting-and-Customer-Segmentation/assets/93611934/cedb4537-319d-4dd3-803a-9c5d350ac699)


# Given the prediction output, we can derive several actionable insights for marketing purposes:

1. Customer Demographics and Targeting
Insight: Understanding the age groups that are generating the most profit.

Action: Tailor marketing campaigns to target age groups with higher profitability, such as age groups 45 and 50 in this example. Create personalized advertisements, promotions, and content that resonate with these demographics.
2. Order Quantity Analysis
Insight: Higher order quantities correlate with higher predicted profits.

Action: Encourage bulk purchases by offering discounts or incentives for larger orders. Implement marketing strategies like "Buy More, Save More" campaigns to increase order quantities.
3. Product Pricing and Cost Management
Insight: Analyzing the relationship between unit cost, unit price, and profit margins.

Action: Adjust pricing strategies to optimize profit margins. For example, if higher unit prices yield higher profit margins, consider repositioning the product as a premium offering. Alternatively, find ways to reduce unit costs without compromising quality to enhance profit margins.
4. Revenue Per Unit
Insight: Revenue per unit varies across different orders.

Action: Focus marketing efforts on promoting products or bundles that offer higher revenue per unit. Highlight the value proposition of these high-revenue products in marketing materials.
5. Predicted Profit Maximization
Insight: Predicted profit values indicate potential high-gain customers or orders.

Action: Prioritize marketing resources on customers or orders predicted to yield the highest profit. Implement loyalty programs or exclusive deals for these high-value customers to retain their business.
Example Marketing Strategies
1. Targeted Email Campaigns
Target Group: Customers aged 45 and 50.
Content: Personalized product recommendations, exclusive discounts, and loyalty rewards.
Goal: Increase order quantities and repeat purchases.
2. Bulk Purchase Incentives
Strategy: Offer tiered discounts based on order quantity (e.g., 10% off for orders over 10 units, 15% off for orders over 20 units).
Goal: Encourage larger purchases to boost overall sales volume and profitability.
3. Premium Product Positioning
Strategy: Rebrand and market products with high unit prices and profit margins as premium options.
Content: High-quality visuals, testimonials, and highlighting unique product features.
Goal: Justify higher prices and attract customers willing to pay a premium.
4. Revenue Optimization
Focus: Promote products with the highest revenue per unit.

**Action**: Create special campaigns, bundle offers, and highlight these products on the website and in promotional materials.

**Goal**: Maximize revenue from each sale.
Implementing the Strategies

**Data-Driven Decision Making**: Use the predicted profit data to inform which customer segments to target and what products to promote.

**Marketing Campaigns**: Design and execute campaigns based on the insights derived from the predictive model output.

**Monitoring and Adjustment**: Continuously monitor the performance of these campaigns and adjust strategies based on real-time data and feedback.

By leveraging the predictive model outputs effectively, marketing teams can make informed decisions that enhance customer targeting, optimize pricing strategies, and ultimately drive higher profitability.
