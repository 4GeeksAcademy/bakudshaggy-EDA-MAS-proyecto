# diabetes_eda.py
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

# Configuration
sns.set(style="whitegrid", palette="pastel")
plt.rcParams.update({'figure.figsize': (12,6), 'font.size': 12})

# Load data
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Initial exploration
print("=== INITIAL DATA EXPLORATION ===")
print(f"Dataset shape: {df.shape}")
print("\nData types:\n", df.dtypes.value_counts())
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False))

# Target analysis
plt.figure()
ax = sns.countplot(x='Diabetes_binary', data=df)
for p in ax.patches:
    ax.annotate(f"{p.get_height()/len(df):.1%}", 
               (p.get_x()+0.3, p.get_height()+500))
plt.title("Diabetes Distribution")
plt.show()

# Binary feature analysis
def binary_analysis(col):
    contingency = pd.crosstab(df[col], df['Diabetes_binary'])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    plt.figure()
    sns.countplot(x=col, hue='Diabetes_binary', data=df)
    plt.title(f"{col} (χ² p={p:.4f})")
    plt.show()

for col in ['HighBP', 'HighChol', 'Smoker', 'HvyAlcoholConsump']:
    binary_analysis(col)

# Numerical analysis
def numerical_analysis(col):
    group0 = df[df['Diabetes_binary'] == 0][col]
    group1 = df[df['Diabetes_binary'] == 1][col]
    t_stat, p_val = stats.ttest_ind(group0, group1)
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    sns.histplot(data=df, x=col, hue='Diabetes_binary', kde=True, ax=ax[0])
    sns.boxplot(x='Diabetes_binary', y=col, data=df, ax=ax[1])
    plt.suptitle(f"{col} (t-test p={p_val:.4f})")
    plt.show()

for col in ['BMI', 'Age', 'MentHlth', 'PhysHlth']:
    numerical_analysis(col)

# Correlation matrix
plt.figure(figsize=(18,12))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Diabetes_binary', axis=1),
    df['Diabetes_binary'],
    test_size=0.2,
    stratify=df['Diabetes_binary'],
    random_state=42
)

# Save data
X_train.to_csv('data/processed/train_features.csv', index=False)
X_test.to_csv('data/processed/test_features.csv', index=False)
y_train.to_csv('data/processed/train_labels.csv', index=False)
y_test.to_csv('data/processed/test_labels.csv', index=False)

print("=== EDA COMPLETE ===")