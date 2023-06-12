# import the linear regression model from Scikit-learn

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Full path to the CSV file
file_path= 'D:\\Y1S2\\ML\\ML Lab 1\\House Pricing.csv'

# Load the dataset into a pandas dataframe
df = pd.read_csv(file_path)

# # set display options to show all columns
# pd.set_option('display.max_columns',None)

# Print the first 5 rows of the dataset to verify it was loaded correctly
print(df.head())
print()

#Data Analysis
print("Data Analysis: ")
print("Shape of data:")
print(df.shape)
print()
print("Info of data:")
print(df.info())
print()
print("description of data:")
pd.set_option('display.max_columns',None)
print(df.describe())

# Check for missing values
print('Checking for missing values:')
print(df.isnull().sum())
print()

#Pair Plot
sns.pairplot(df)
plt.show()

#SubPlot
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.violinplot(x = 'mainroad', y = 'price', data = df)
plt.subplot(2,3,2)
sns.violinplot(x = 'guestroom', y = 'price', data = df)
plt.subplot(2,3,3)
sns.violinplot(x = 'basement', y = 'price', data = df)
plt.subplot(2,3,4)
sns.violinplot(x = 'hotwaterheating', y = 'price', data = df)
plt.subplot(2,3,5)
sns.violinplot(x = 'airconditioning', y = 'price', data = df)
plt.subplot(2,3,6)
sns.violinplot(x = 'furnishingstatus', y = 'price', data = df)
plt.show()

# Encoding categorical variables
print('Data Preprocessing:')
# create a LabelEncoder object
le = LabelEncoder()

# apply the LabelEncoder to each categorical column in the dataset
df['mainroad'] = le.fit_transform(df['mainroad'])
df['guestroom'] = le.fit_transform(df['guestroom'])
df['basement'] = le.fit_transform(df['basement'])
df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
df['airconditioning'] = le.fit_transform(df['airconditioning'])
df['prefarea'] = le.fit_transform(df['prefarea'])
df['furnishingstatus'] = df['furnishingstatus'].replace({'unfurnished':0,'semi-furnished':1,'furnished':2})

print(df.head())
print()

# Scaling and normalising features
print('Scaling and normalising features')

# Min-max scaling:
print('Min-max scaling:')

# create the scaler object
scaler = MinMaxScaler()

# fit and transform the data
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
# print scaled data
print(scaled_df.head())
print()

# spliting dataset to train set and test set
print('Spliting dataset to training set and testing set')

x = df.drop('price',axis=1) # Features
y = df['price'] # Target variable

# Split the dataset into 80% training data and 20% testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# print the shapes of the training and testing sets
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# x_train has a shape of (436, 12), which means that it contains 436 samples (rows) and 12 features (columns) for the training data.
# x_test has a shape of (109, 12), which means that it contains 109 samples (rows) and 12 features (columns) for the testing data.
# y_train has a shape of (436,), which means that it contains 436 target values (prices) for the training data.
# y_test has a shape of (109,), which means that it contains 109 target values (prices) for the testing data.


# Corellation Heatmap
#corr_matrix = df.corr()
plt.figure(figsize = (16,10))
sns.heatmap(scaled_df.corr(),annot=True, cmap='coolwarm')
plt.show()


# Regression Model Development
print("Regression Model Development:")
print("-------------------------------")
# create a linear regression object
print("Using Linear Regression: ")
reg = LinearRegression()

#fit the model to the training data
reg.fit(x_train, y_train)

#make predictions on the testing data
y_pred_lr = reg.predict(x_test)

#evaluate the model using mean squared error,Root mean squared error and R-squared
mse_lr = mean_squared_error(y_test, y_pred_lr)
print("MSE: ", round(mse_lr,2))
rmse_lr = np.sqrt(mse_lr)
print("RMSE: ", round(rmse_lr,2))
r2_scores_lr = reg.score(x_test,y_test)
print("R-squared: ", round(r2_scores_lr,2))
print("-------------------------------")

# Using Random Forest Regression
print("Using Random Forest Regression:")

# create a Random Forest Regressor Object
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

#Train the model using the training data sets
regressor.fit(x_train, y_train)

# Predict the house prices using the testing set
y_pred_rf = regressor.predict(x_test)

# Calculation of RMSE, MSE, and R-squared
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_scores_rf = r2_score(y_test, y_pred_rf)
print("MSE: ", round(mse_rf,2))
print("RMSE: ", round(rmse_rf, 2))
print("R-squared: ", round(r2_scores_rf, 2))
print("-------------------------------")

# using Decision Tree
print("Using Decision Tree Regressor: ")

# create a decision tree regressor object
dtr = DecisionTreeRegressor()

# Train the model using the training sets
dtr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred_dt = dtr.predict(x_test)

# Calculate evaluation matrics(MSE,RMSE,R-squared)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_scores_dt = r2_score(y_test, y_pred_dt)
print("MSE: ", round(mse_dt, 2))
print("RMSE: ", round(rmse_dt, 2))
print("R-squared: ", round(r2_scores_dt, 2))
print("-------------------------------")


print("Conclusion:")
print("Based on the metrics, it seems that the Linear Regression model performs slightly better than the Random Forest Regression model and Decision Tree Regressor. ")
print("The Linear Regression model has a lower Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) than the Random Forest Regression model and Decision Tree Regressor, ")
print("indicating that its predictions are closer to the actual values.")
print("Additionally, the Linear Regression model has a slightly higher R-squared value, ")
print("indicating that it explains a slightly higher proportion of the variance in the target variable.")


#Linear Regression Scatter Plot
fig1 = plt.figure()
plt.scatter(y_test, y_pred_lr, color='blue')
plt.title('Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
#plt.show()


# Random Forest Regression Scatter Plot
fig2 = plt.figure()
plt.scatter(y_test, y_pred_rf, color='green')
plt.title('Random Forest Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
#plt.show()

# Decision Tree Regressor
fig3 = plt.figure()
plt.scatter(y_test, y_pred_dt, color='red')
plt.title('Decision Tree Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()


