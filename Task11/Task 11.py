import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

path= kagglehub.dataset_download("uciml/iris")
print(path)

sns.set_theme(style="whitegrid")

csv_file = os.path.join(path, "iris.csv")  
df = pd.read_csv(csv_file)


print(df.shape)

#the first 5 rows of the dataset
print("\n--------First 5 rows of the dataset-------")
time.sleep(1)
print(df.head())

#data info
print("\n--------Data Info-------")
time.sleep(1)
print(df.info())

print("\n--------Data Types-------")
time.sleep(1)
print(df.dtypes)

#DATA CLEANING OPERATIONS

#missing values for the dataset
missing_values = df.isnull().sum()
print("\n--------Missing Values-------")
time.sleep(1)
print(missing_values)

#remove duplicates for the dataset
duplicates = df.duplicated().sum()#duplicate count
print("\n--------Duplicates-------")
time.sleep(1)
print(duplicates)

df.drop_duplicates(inplace=True)
time.sleep(1)
print("\n--------Duplicates Removed-------")
print(df)

#SUMMARY STATISTICS FOR THE DATASET
print("\n--------Summary Statistics-------")
time.sleep(1)

print("\n--------Data Description-------")
time.sleep(1)
print(df.describe())
print(df.describe(include="object"))  # text columns: count, unique, top, freq

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(exclude="number").columns

#median of numeric columns
print("\n--------Median of Numeric Columns-------")
print(df[num_cols].median())
time.sleep(1)

#mode of numeric columns
print("\n--------Mode of Numeric Columns-------")
print(df[num_cols].mode().iloc[0])
time.sleep(1)

#variance, skewness, and kurtosis of numeric columns
print("\n--------Variance, Skewness, and Kurtosis of Numeric Columns-------")

print("------Variance------")
print(df[num_cols].var())
print("\n------Skewness------")
print(df[num_cols].skew())    # asymmetry of distribution
print("\n------Kurtosis------")
print(df[num_cols].kurt())    # tail heaviness

#descriptive statistics for numeric columns
print("\n--------Descriptive Statistics for Numeric Columns-------")
time.sleep(1)
print(df[num_cols].describe())      # stats only on numeric columns

print("\n--------Correlation Matrix for Numeric Columns-------")
print(df[num_cols].corr())         # correlation matrix only on numeric columns


# --- 5. VISUALIZE ---
# ==========================================
# DATA VISUALIZATIONS FOR IRIS DATASET
# ==========================================

# 1. HISTOGRAMS (Distribution of each feature with KDE)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution of Numeric Features', fontsize=16)

feature_cols = [col for col in num_cols if col.lower() != 'id']  # Exclude 'Id' if present

for ax, col in zip(axes.flatten(), feature_cols):
    sns.histplot(data=df, x=col, hue='Species', kde=True, ax=ax, palette='Set2')
    ax.set_title(f'Histogram & KDE of {col}')

plt.tight_layout()
plt.show()


# 2. BOX PLOTS (Detecting Outliers & Class Distribution)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Boxplots across Iris Species', fontsize=16)

for ax, col in zip(axes.flatten(), feature_cols):
    sns.boxplot(data=df, x='Species', y=col, ax=ax, palette='Set2')
    ax.set_title(f'{col} by Species')

plt.tight_layout()
plt.show()


# 3. SCATTER PLOT & PAIRPLOT (Feature Relationships)
# Single scatter plot: Sepal Length vs Sepal Width
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x=feature_cols[0], y=feature_cols[1], hue='Species', style='Species', s=100, palette='Set1')
plt.title(f'Scatter Plot: {feature_cols[0]} vs {feature_cols[1]}')
plt.show()

# Pairplot across all numerical features (Very common for Iris EDA)
sns.pairplot(df[feature_cols + ['Species']], hue='Species', corner=True, palette='Set1')
plt.suptitle('Pairplot of Iris Features', y=1.02)
plt.show()


# 4. CORRELATION HEATMAP
plt.figure(figsize=(8, 6))
sns.heatmap(df[feature_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# 5. COUNT PLOT (Categorical Distribution)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Species', palette='viridis')
plt.title('Sample Count per Iris Species')
plt.show()

