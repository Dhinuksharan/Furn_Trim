#Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler

#Loading the dataset
customer_survey_df=pd.read_csv("C:\\data\\work\\data\\customer_survey_new.csv")

#Checking the dataset
customer_survey_df

#Checking the columns
column_list=(list(customer_survey_df))
print(column_list)

#Checking for missing values
print(customer_survey_df.isnull().sum())

#Checking for duplicates
print("Product Catalog Duplicates:", customer_survey_df.duplicated().sum())


#Standardizing categorical data
categorical_cols = ["Customer Type", "Budget", "Preferred Style", "Purchase Frequency"]
for col in categorical_cols:
    if col in customer_survey_df.columns:
        customer_survey_df[col] = customer_survey_df[col].str.lower().str.strip()

    
# Converting data types
convert_types = {
    "Recommendation Score": int,  # Keep as integer without scaling
    "Customer Reviews": float
}

for col, dtype in convert_types.items():
    if col in customer_survey_df.columns:
        customer_survey_df[col] = customer_survey_df[col].astype(dtype)

# Normalizing using StandardScaler
scaler = StandardScaler()
scaled_cols = ["Customer Reviews"]  
customer_survey_df[scaled_cols] = scaler.fit_transform(customer_survey_df[scaled_cols])

# Visualizing Outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=customer_survey_df["Recommendation Score"])
plt.title("Recommendation Score Distribution After Removing Outliers")
plt.show()

processed_file_path = "Processed_Customer_survey.csv"

print("Reached the CSV saving step. Attempting to save...")
customer_survey_df.to_csv(processed_file_path, index=False)
print("Processed file saved successfully at:", processed_file_path)

