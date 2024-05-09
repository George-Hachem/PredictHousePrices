import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from category_encoders.ordinal import OrdinalEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


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
def checkHowManyFraudClaims():

    # df = df.rename(columns={'FraudFound_P': 'FraudFound'})

    # Check how many records are fraud and plot pie chart
    labels = df.FraudFound.value_counts().index
    labels = ["No" if i==0 else "Yes" for i in labels]
    sizes = df.FraudFound.value_counts().values


    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,explode=[0.05,0])
    plt.title('Claims detected as Fraud')
    plt.axis('equal') 
    plt.show()


def carFraudCounts():
    df_fraud = df[df['FraudFound'] == 1]  #filter to get rows where there is a fraud claim
    # print(df_fraud) 

    # Count the number claims in each vehicle category
    fraud_counts = df_fraud['VehicleCategory'].value_counts() 
    print(fraud_counts)
    fraud_counts.plot(kind='line')
    plt.show()

def ageHistorgram():
    #Analyzes the number of fraud per age
    plt.figure(figsize=(10, 5))
 
    plt.subplot(1, 2, 1)
    plt.hist(df['Age'], bins=10, edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Histogram')


    mapping_dict = {
        '16 to 17': '16-17',
        '18 to 20': '18-20',
        '21 to 25': '21-25',
        '26 to 30': '26-30',
        '31 to 35': '31-35',
        '36 to 40': '36-40',
        '41 to 50': '41-50',
        '51 to 65': '51-65',
        'over 65': '65+'
    }

    df['AgeOfPolicyHolder'] = df['AgeOfPolicyHolder'].replace(mapping_dict)

    # Frauds Per Policy Holder Age Group histogram
    plt.subplot(1, 2, 2)  
    policyAge = df.groupby('AgeOfPolicyHolder')['FraudFound'].sum()
    bars = plt.bar(policyAge.index, policyAge.values, edgecolor='black')
    plt.title("Frauds Per Policy Holder Age Group")
    plt.xlabel("Policy Holder Age Group")
    plt.ylabel("Number Of Fraud Claims")
    plt.xticks(rotation=45, ha='right')


    plt.tight_layout()
    plt.show()

    