import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fraud_oracle.csv')

# print(df.head())
#python3 -u "/Users/george/PredictHousePrices/PredictHousePrices/index.py"
# Summary statistics
# print(df.describe(), "\n")

# # Check for missing values
# print(df.isnull().sum())

#print(df.dtypes)

#Checking for duplicate values
#print(df.duplicated().sum())

# Histogram of the 'Age' column to see its distribution
# df['Age'].hist()
# plt.title('Distribution of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

df = df.rename(columns={'FraudFound_P': 'FraudFound'})

# Check how many records are fraud and plot pie chart
labels = df.FraudFound.value_counts().index
labels = ["No" if i==0 else "Yes" for i in labels]
sizes = df.FraudFound.value_counts().values


plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,explode=[0.05,0])
plt.title('Claims detected as Fraud')
plt.axis('equal') 
plt.show()