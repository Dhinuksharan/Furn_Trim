import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler

#Loading dataset
product_catalog_df = pd.read_csv("C:\\data\\Work\\data\\product_catalog_new.csv")

#Checking whether the dataset has been loaded properly
product_catalog_df

#Checking the headings
product_catalog_df.head()

#Describing the dataset
product_catalog_df.describe

#Checking the column names
column_list=(list(product_catalog_df.columns))
print(column_list)

#Checking the number of products in the dataset
print(product_catalog_df["Product Name"].nunique())

#Observing the different types of data

print("=============================================")
print("Data types of the columns in the data frame:", product_catalog_df.dtypes)
print("=============================================")
print("Shape of the data frame:", product_catalog_df.shape)
print("=============================================")
print("Information about the data frame:", product_catalog_df.info())
product_catalog_df.head()

#Checking missing values
print(product_catalog_df.isnull().sum())

#Checking for duplicates
print("Product Catalog Duplicates:", product_catalog_df.duplicated().sum())


#Checking outliers
print("\n--- Checking Outliers (Boxplot) ---")
plt.figure(figsize=(8,5))
sns.boxplot(x=product_catalog_df["Price"])
plt.title("Price Distribution Before Handling Outliers")
plt.show()

#Standardizing categorical data
categorical_cols = [ "Payment Type"]
for col in categorical_cols:
    if col in product_catalog_df.columns:
        product_catalog_df[col] = product_catalog_df[col].str.lower().str.strip()

#map sellable online to 0 and 1
if "Sellable Online" in product_catalog_df.columns:
    product_catalog_df["Sellable Online"] = product_catalog_df["Sellable Online"].replace({"Yes": 1, "No": 0}).astype(int)

# Scaling only selected numerical columns
scaled_cols = ["Price", "Return Rate", "Storage Cost"]
if "Seasonality Score" in product_catalog_df.columns:  # Checking if it exists
    scaled_cols.append("Seasonality Score")

scaler = StandardScaler()
product_catalog_df[scaled_cols] = scaler.fit_transform(product_catalog_df[scaled_cols])


 #Convert data types
convert_types = {
    "Price": float,
    "Sales Volume": int, 
    "Return Rate": float,
    "Storage Cost": float
}

for col, dtype in convert_types.items():
    if col in product_catalog_df.columns:
        product_catalog_df[col] = pd.to_numeric(product_catalog_df[col], errors='coerce').astype(dtype)


#Bar plot for products vs sales
plt.figure(figsize=(10,5))
sns.barplot(x=product_catalog_df["Category"], y=product_catalog_df["Sales Volume"], ci=None)
plt.xticks(rotation=45)
plt.title("Sales Volume by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales Volume")
plt.show()

# Select only numerical columns for correlation
numeric_data = product_catalog_df.select_dtypes(include=["number"])

# Generate heatmap
plt.figure(figsize=(8,5))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

#return rate
plt.figure(figsize=(8,5))
sns.histplot(product_catalog_df["Price"], bins=20, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

#product distribution
plt.figure(figsize=(8,5))
product_catalog_df["Category"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, cmap="viridis")
plt.title("Product Distribution by Category")
plt.ylabel("") 

#scatter plot
plt.figure(figsize=(8,5))
sns.scatterplot(x=product_catalog_df["Price"], y=product_catalog_df["Sales Volume"])
plt.title("Price vs Sales Volume")
plt.xlabel("Price")
plt.ylabel("Sales Volume")
plt.show()


processed_file_path = "Processed_product_catalog.csv"
print(" Processed file saved successfully at:", processed_file_path)
product_catalog_df.to_csv(processed_file_path, index=False)

