#!/usr/bin/env python
# coding: utf-8

# # question 01
To address the task described, you can follow these steps:

Automated Feature Selection:

You can use various automated feature selection techniques to identify important features. One common approach is to use a tree-based algorithm like a Random Forest or Gradient Boosting Machine and examine the feature importance scores. Higher importance scores indicate more significant features.
Handling Missing Values:

For missing values, you can use techniques like mean imputation, median imputation, mode imputation, or more advanced methods like K-Nearest Neighbors imputation or regression imputation. The choice of imputation method depends on the nature of the data and the specific domain.
Dealing with Correlated Features:

Highly correlated features can lead to multicollinearity, which can affect the performance of some models. You can use techniques like Principal Component Analysis (PCA) or feature selection methods like Recursive Feature Elimination (RFE) to reduce the dimensionality and address multicollinearity.
Building a Pipeline:

You can use the scikit-learn library in Python to build a pipeline that automates these steps. The pipeline will handle data preprocessing, feature selection, and missing value imputation.
# In[1]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

# Define the pipeline steps
steps = [
    ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with mean
    ('feature_selection', SelectFromModel(RandomForestRegressor())),  # Automated feature selection
    ('regressor', RandomForestRegressor())  # Regression model
]

# Create the pipeline
pipeline = Pipeline(steps)

Training and Evaluation:

Once the pipeline is set up, you can split your data into training and testing sets, and then use the pipeline to train and evaluate the model.
# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming X is your feature matrix and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

Hyperparameter Tuning:

You may further tune hyperparameters to optimize the model's performance. Techniques like cross-validation and grid search can be used for this purpose.
By following these steps, you can automate the feature engineering process, handle missing values, and build a machine learning model that incorporates important features from your dataset.Certainly! To create a numerical pipeline, you can use scikit-learn's Pipeline class to sequentially apply a series of transformations to the numerical features in your dataset. Below are some common steps you might include in a numerical pipeline:

Standardization or Scaling:

Standardize or scale the numerical features to have zero mean and unit variance. This is important for models like Support Vector Machines and K-Nearest Neighbors.
Handling Missing Values:

Apply an imputation strategy to handle any missing values in the numerical features.
Feature Selection:

Optionally, perform automated feature selection to choose the most relevant numerical features.
Model:

Add a regression or classification model to make predictions based on the processed numerical features.
Here is an example of how you can create a numerical pipeline using scikit-learn:
# In[3]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

# Define the steps for the numerical pipeline
numerical_steps = [
    ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with mean
    ('scaler', StandardScaler()),  # Standardize the numerical features
    ('feature_selection', SelectFromModel(RandomForestRegressor())),  # Automated feature selection
    ('regressor', RandomForestRegressor())  # Regression model
]

# Create the numerical pipeline
numerical_pipeline = Pipeline(numerical_steps)

In this example, the numerical pipeline consists of the following steps:

Imputation:

Missing values in the numerical features are replaced with the mean value.
Standardization:

The numerical features are standardized to have zero mean and unit variance.
Feature Selection:

Automated feature selection is performed using a Random Forest Regressor to choose the most relevant features.
Regression Model:

A Random Forest Regressor is used as the regression model.
You can customize the steps in the numerical pipeline based on your specific needs and the characteristics of your dataset.
# In[ ]:


from sklearn.impute import SimpleImputer

# Assuming X_num is your numerical feature matrix
# Create a SimpleImputer instance
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the data and transform it
X_num_imputed = imputer.fit_transform(X_num)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Define the steps for the numerical pipeline
numerical_steps = [
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler()),  # Standardize the numerical features
    # Add other steps as needed
]

# Create the numerical pipeline
numerical_pipeline = Pipeline(numerical_steps)

# Assuming X_num is your numerical feature matrix
X_num_processed = numerical_pipeline.fit_transform(X_num)


# In[ ]:


from sklearn.preprocessing import StandardScaler

# Assuming X_num_imputed is your numerical feature matrix with missing values imputed
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the data and transform it
X_num_scaled = scaler.fit_transform(X_num_imputed)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Define the steps for the numerical pipeline
numerical_steps = [
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler()),  # Standardize the numerical features
    # Add other steps as needed
]

# Create the numerical pipeline
numerical_pipeline = Pipeline(numerical_steps)

# Assuming X_num is your numerical feature matrix
X_num_processed = numerical_pipeline.fit_transform(X_num)

Certainly! To create a categorical pipeline, you can use scikit-learn's Pipeline class to sequentially apply a series of transformations to the categorical features in your dataset. Below are some common steps you might include in a categorical pipeline:

One-Hot Encoding:

Convert categorical variables into a binary (0 or 1) format using one-hot encoding.
Handling Missing Values:

Apply an imputation strategy to handle any missing values in the categorical features.
Feature Selection:

Optionally, perform automated feature selection to choose the most relevant categorical features.
Model:

Add a regression or classification model to make predictions based on the processed categorical features.
Here is an example of how you can create a categorical pipeline using scikit-learn:
# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

# Define the steps for the categorical pipeline
categorical_steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with mode
    ('encoder', OneHotEncoder(handle_unknown='ignore')),  # One-hot encode categorical features
    ('feature_selection', SelectFromModel(RandomForestRegressor())),  # Automated feature selection
    ('regressor', RandomForestRegressor())  # Regression model
]

# Create the categorical pipeline
categorical_pipeline = Pipeline(categorical_steps)

In this example, the categorical pipeline consists of the following steps:

Imputation:

Missing values in the categorical features are replaced with the mode (most frequent value).
One-Hot Encoding:

Categorical variables are converted into a binary format using one-hot encoding.
Feature Selection:

Automated feature selection is performed using a Random Forest Regressor to choose the most relevant categorical features.
Regression Model:

A Random Forest Regressor is used as the regression model.
# In[ ]:


from sklearn.impute import SimpleImputer

# Assuming X_cat is your categorical feature matrix
# Create a SimpleImputer instance
imputer = SimpleImputer(strategy='most_frequent')

# Fit the imputer on the data and transform it
X_cat_imputed = imputer.fit_transform(X_cat)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Define the steps for the categorical pipeline
categorical_steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
    ('encoder', OneHotEncoder(handle_unknown='ignore')),  # One-hot encode categorical features
    # Add other steps as needed
]

# Create the categorical pipeline
categorical_pipeline = Pipeline(categorical_steps)

# Assuming X_cat is your categorical feature matrix
X_cat_processed = categorical_pipeline.fit_transform(X_cat)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

# Assuming X_cat_imputed is your categorical feature matrix with missing values imputed
# Create a OneHotEncoder instance
encoder = OneHotEncoder(handle_unknown='ignore')  # 'ignore' handles new categories in test data

# Fit the encoder on the data and transform it
X_cat_encoded = encoder.fit_transform(X_cat_imputed)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Define the steps for the categorical pipeline
categorical_steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
    ('encoder', OneHotEncoder(handle_unknown='ignore')),  # One-hot encode categorical features
    # Add other steps as needed
]

# Create the categorical pipeline
categorical_pipeline = Pipeline(categorical_steps)

# Assuming X_cat is your categorical feature matrix
X_cat_processed = categorical_pipeline.fit_transform(X_cat)


# In[ ]:


from sklearn.compose import ColumnTransformer

# Assuming X_num_scaled is your scaled numerical feature matrix
# and X_cat_encoded is your one-hot encoded categorical feature matrix

# Define the transformations for numerical and categorical columns
numerical_features = ['numerical_feature1', 'numerical_feature2', ...]  # Replace with actual feature names
categorical_features = ['categorical_feature1', 'categorical_feature2', ...]  # Replace with actual feature names

# Define the transformations for numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Numerical pipeline
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Categorical pipeline
    ])

# Apply the transformations to the data
X_processed = preprocessor.fit_transform(X)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.compose import ColumnTransformer

# Define the transformations for numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])


# In[ ]:


from sklearn.pipeline import Pipeline

# Define the steps for the final pipeline
final_steps = [
    ('preprocessor', preprocessor),  # Apply the combined preprocessing steps
    ('classifier', RandomForestClassifier())  # Random Forest Classifier as the final model
]

# Create the final pipeline
final_pipeline = Pipeline(final_steps)


# In[ ]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the final pipeline
final_pipeline.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = final_pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = final_pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")


# # question 02

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming you have a dataset 'X' with features and 'y' with labels

# Define the numerical and categorical features
numerical_features = ['numerical_feature1', 'numerical_feature2', ...]  # Replace with actual numerical feature names
categorical_features = ['categorical_feature1', 'categorical_feature2', ...]  # Replace with actual categorical feature names

# Define the pipelines for numerical and categorical features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Define the final pipeline with the classifiers and the voting classifier
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),  # Random Forest Classifier
        ('lr', LogisticRegression())  # Logistic Regression Classifier
    ], voting='hard'))  # 'hard' voting combines the predictions by majority rule
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the final pipeline
final_pipeline.fit(X_train, y_train)

# Evaluate the accuracy on the test set
y_pred = final_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

