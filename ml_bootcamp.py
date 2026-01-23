"""Machine Learning Bootcamp: Data Preparation for ML Models

This module provides a comprehensive introduction to data preprocessing for
machine learning, including:
- Data type conversions and categorical variable handling
- Feature scaling and normalization techniques
- One-hot encoding for categorical variables
- Train-test-tune splitting with stratification
- Baseline/prevalence calculation for classification problems

Designed for students new to machine learning and data preparation.
"""

# %% [markdown]
# # Machine Learning Bootcamp

# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data


# %% [markdown]
# ## Phase I
#
# ### Working to develop a model that can predict cereal quality rating...
#
# Key Questions to Consider:
# - What is the target variable?
# - Assuming we are able to optimize and make recommendations,
#   how does this translate into a business context?
# - Prediction problem: Classification or Regression?
# - Independent Business Metric: Assuming that higher ratings result in
#   higher sales, can we predict which new cereals that enter the market
#   over the next year will perform the best?

# %% [markdown]
# ## Phase II

# %% [markdown]
# ### Scale/Center/Normalizing/Variable Classes

# %%
# Read in the cereal dataset from the class repository
# You can use this URL or download it locally
# Dataset source: https://github.com/UVADS/DS-3001
cereal_url = ("https://raw.githubusercontent.com/UVADS/DS-3001/"
              "main/data/cereal.csv")
cereal = pd.read_csv(cereal_url)

# Let's check the structure of the dataset and see if we have any issues
# with variable classes (data types)
cereal.info()
# Usually we need to convert string/object columns to category type


# %%
# Convert categorical columns to the 'category' data type
# This is important for:
# 1. Memory efficiency - categories use less memory
# 2. Proper statistical analysis
# 3. Correct handling in ML algorithms

cols = ["type", "mfr", "vitamins", "shelf"]
cereal[cols] = cereal[cols].astype('category')
# astype() changes the data type of specified columns
# 'category' is a special pandas dtype for categorical variables

# Alternative approach for a single column (commented out):
# cereal.type = cereal.type.astype('category')

# Check the data types to confirm changes
cereal.dtypes  # Simpler than .info(), shows just the data types

# %%
# Let's take a closer look at the manufacturer (mfr) variable
# value_counts() displays the frequency of each unique value
print(cereal.mfr.value_counts())

# %%
# Usually we don't want more than 5 groups in a categorical variable
# Too many categories can lead to:
# 1. Sparse data (too few samples per category)
# 2. Overfitting in ML models
# 3. Computational inefficiency
#
# Strategy: Keep the large groups (G=General Mills, K=Kellogg's)
# and group all smaller categories as "Other"

top_manufacturers = ['K', 'G']
# Lambda function explanation:
# lambda x: x if x in top else "Other"
# - Takes input x (each manufacturer code)
# - Returns x if it's in our top list
# - Otherwise returns "Other"
# Lambda functions are small anonymous functions useful for simple operations
cereal.mfr = (cereal.mfr.apply(lambda x: x if x in top_manufacturers
                               else "Other")).astype('category')

# Verify the grouping worked
cereal.mfr.value_counts()  # This is much better - only 3 groups now!

# %%
# Check other categorical variables
cereal.type.value_counts()  # Looks good - just 2 types (Cold/Hot)

# %%
cereal.vitamins.value_counts()  # Also good

# %%
# Weight is numeric, not categorical, so value counts aren't meaningful here
cereal.weight.value_counts()

# %% [markdown]
# ### Scale/Center
#
# **Why scale data?**
# Many ML algorithms (like SVM, neural networks, k-NN) are sensitive to the
# scale of features. Features with larger ranges can dominate the model.
# Scaling puts all features on the same scale for fair comparison.

# %%
# Standardization (Z-score normalization)
# Formula: (x - mean) / standard_deviation
# Result: Mean = 0, Standard Deviation = 1
# Use when: Data is normally distributed or algorithm assumes
# normal distribution

sodium_standardized = StandardScaler().fit_transform(cereal[['sodium']])
# fit_transform() learns the mean and std, then applies the transformation
# Double brackets [['sodium']] create a DataFrame (required by sklearn)

# Display first 10 standardized values
print(sodium_standardized[:10])  # These are z-scores

# %% [markdown]
# ### Normalizing the Numeric Values
#
# **Min-Max Normalization**
# Formula: (x - min) / (max - min)
# Result: All values between 0 and 1
# Use when: You want bounded values or don't assume normal distribution

# %%
# Min-Max scaling example with sodium
sodium_normalized = MinMaxScaler().fit_transform(cereal[['sodium']])
print(sodium_normalized[:10])  # Values now between 0 and 1

# %%
# Let's verify that scaling preserves the distribution shape
# Plot the original distribution
cereal.sodium.plot.density()

# %%
# Plot the normalized distribution
pd.DataFrame(sodium_normalized).plot.density()
# The shape is identical - only the scale changed!

# %%
# Now normalize ALL numeric columns in the dataset
# Step 1: Select all numeric columns
numeric_cols = list(cereal.select_dtypes('number'))
# select_dtypes('number') finds all int and float columns

# Step 2: Apply Min-Max scaling to all numeric columns
cereal[numeric_cols] = MinMaxScaler().fit_transform(cereal[numeric_cols])
# Now all numeric features are on the same scale (0 to 1)
# This is crucial for distance-based algorithms!

# %% [markdown]
# ### One-Hot Encoding
#
# **What is One-Hot Encoding?**
# ML algorithms work with numbers, not categories. One-hot encoding converts
# categorical variables into binary (0/1) indicator columns.
#
# Example: If 'type' has values ['C', 'H'], it becomes:
# - type_C: 1 if Cold, 0 otherwise
# - type_H: 1 if Hot, 0 otherwise
#
# **Why use it?**
# - Prevents the algorithm from assuming ordinal relationships
# - Works with any ML algorithm
# - No arbitrary numeric encoding

# %%
# Get list of all categorical columns
category_list = list(cereal.select_dtypes('category'))

# Apply one-hot encoding
cereal_encoded = pd.get_dummies(cereal, columns=category_list)
# get_dummies() creates new binary columns for each category level
# Original categorical columns are removed, replaced by indicator columns

# Check the result
cereal_encoded.info()
# Notice: Each category now has its own column with 1s and 0s!

# %% [markdown]
# ### Baseline/Prevalence
#
# **What is Baseline/Prevalence?**
# For classification problems, the baseline is the accuracy you'd get by
# always predicting the most common class. Your ML model should beat this!
#
# **Why calculate it?**
# - Sets a minimum performance target
# - Helps evaluate if your model is actually learning
# - Important for imbalanced datasets

# %%
# We'll convert the continuous 'rating' into a binary classification problem
# by identifying high-quality cereals (top quartile)

# Visualize the rating distribution
print(cereal_encoded.boxplot(column='rating', vert=False, grid=False))
# Display summary statistics
print(cereal_encoded.rating.describe())
# Note: The upper quartile (75th percentile) is at 0.43

# %%
# Create a binary target variable: rating_f (rating_flag)
# 1 = High quality (rating > 0.43), 0 = Lower quality (rating <= 0.43)
cereal_encoded['rating_f'] = pd.cut(cereal_encoded.rating,
                                    bins=[-1, 0.43, 1],
                                    labels=[0, 1])
# pd.cut() bins continuous values into discrete categories
# bins=[-1, 0.43, 1] creates two bins: (-1, 0.43] and (0.43, 1]
# labels=[0, 1] assigns 0 to first bin, 1 to second bin

# Verify the new column
cereal_encoded.info()  # See the new rating_f column

# %%
# Calculate the prevalence (percentage of high-quality cereals)
prevalence = (cereal_encoded.rating_f.value_counts()[1] /
              len(cereal_encoded.rating_f))
# value_counts()[1] gets count of '1' values (high quality)
# Divide by total count to get proportion

print(f"Baseline/Prevalence: {prevalence:.2%}")
# This is our baseline - any model should beat this accuracy!

# %%
# Double-check our calculation
print(cereal_encoded.rating_f.value_counts())
print(f"Manual calculation: 21/(21+56) = {21/(21+56):.4f}")  # Matches!

# %% [markdown]
# ### Dropping Variables and Partitioning
#
# **Data Partitioning Strategy**
# We split data into three sets:
# 1. **Training (55 samples)**: Used to train the model
# 2. **Tuning (11 samples)**: Used to tune hyperparameters
# 3. **Test (11 samples)**: Used for final evaluation only
#
# **Why three sets?**
# - Training: Model learns patterns
# - Tuning: Select best model configuration without biasing test results
# - Test: Unbiased evaluation of final model performance

# %%
# Clean up the dataset before splitting
# Remove columns we can't use as features:
# - 'name': Unique identifier, not a predictive feature
# - 'rating': Our target variable (we're using rating_f instead)

cereal_clean = cereal_encoded.drop(['name', 'rating'], axis=1)
# axis=1 means drop columns (axis=0 would drop rows)
print(cereal_clean)

# %%
# First split: Separate training data from the rest
train, test = train_test_split(
    cereal_clean,
    train_size=55,
    stratify=cereal_clean.rating_f
)
# stratify=cereal_clean.rating_f ensures class proportions are preserved
# This reduces sampling error and gives more reliable results
# PEP 8: lowercase variable names

# %%
# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# %%
# Second split: Split remaining data into tuning and test sets (50/50)
tune, test = train_test_split(
    test,
    train_size=.5,
    stratify=test.rating_f
)

# %%
# Verify prevalence in training set
# (Should match overall prevalence due to stratification)
print("Training set class distribution:")
print(train.rating_f.value_counts())
print(f"Training prevalence: {15/(40+15):.2%}")

# %%
# Verify prevalence in tuning set
print("\nTuning set class distribution:")
print(tune.rating_f.value_counts())
print(f"Tuning prevalence: {3/(8+3):.2%}")

# %%
# Verify prevalence in test set
print("\nTest set class distribution:")
print(test.rating_f.value_counts())
print(f"Test prevalence: {3/(8+3):.2%}")
# All three sets have similar prevalence - stratification worked!

# %% [markdown]
# # Now You Try!
#
# Practice what you've learned with a different dataset!
# Apply the same preprocessing steps to the job placement dataset.

# %%
# Load the job placement dataset
job_url = ("https://raw.githubusercontent.com/DG1606/CMS-R-2020/"
           "master/Placement_Data_Full_Class.csv")
job = pd.read_csv(job_url)
print(job.head())

# %%
# Explore the structure of the dataset
job.info()
# Check for data types and identify which columns need type conversion

# %%
# Load another dataset for practice
data_world_url = "https://query.data.world/s/ttvvwduzk3hwuahxgxe54jgfyjaiul"
response = requests.get(data_world_url).text
dataset = pd.read_csv(StringIO(response))
print(dataset.head())
print(response)

# %%
# Re-examine job dataset structure
job.info()
# TODO: Summarize the missing values

# %%
# Summarize missing values in the job dataset
# isna() returns True for missing values, sum() counts them
job.isna().sum()
# Alternative: job.notna().sum() counts non-missing values

# %%
# Remove rows with any missing values
# notna() returns True for non-missing values
# all(axis='columns') checks if all values in a row are non-missing
job_clean = job.loc[job.notna().all(axis='columns')]

# %%
