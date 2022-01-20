# %%
# Import numpy and pandas
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('stomach_cancer_new_cases_per_100000_men.csv')

# Create arrays for features and target variable
#y = df['life'].values
#X = df['fertility'].values

# Print the dimensions of y and X before reshaping
#print("Dimensions of y before reshaping: ", y.shape)
#print("Dimensions of X before reshaping: ", X.shape)

# Reshape X and y
#y_reshaped = y.reshape(-1, 1)
#X_reshaped = X.reshape(-1, 1)

# Print the dimensions of y_reshaped and X_reshaped
#print("Dimensions of y after reshaping: ", y_reshaped.shape)
#print("Dimensions of X after reshaping: ", X_reshaped.shape)

# %%
df.head()
# %%
df.fillna(0, inplace=True)
# %%

df
# %%

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(
    min(X_fertility), max(X_fertility)).reshape(-1, 1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

# %%
# Import necessary modules

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# %%%
# Cross validation

reg = LinearRegression()

cv_results = cross_val_score(reg, X, y, cv=5)


# %%

# Import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
