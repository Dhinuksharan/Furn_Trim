#In this phase additional attributes were derived to enhance product insights. The "Sales-to-Return Ratio" was introduced to measure efficiency in handling returns, while "Revenue Per Product" quantified total revenue generated per item. 
#A calculated "Online Sellability Score" assessed the likelihood of online sales success, and "Storage Efficiency Score" measured how effectively revenue was generated relative to storage costs. 
#A "High Demand Indicator" was assigned to products in the top 25% of sales, while "Implicit Feedback Score" translated customer sentiment into numerical values.
# Additionally, an "Expert Judgment Score" was introduced, combining demand, ratings, and revenue performance to highlight high-impact products.
# The engineered features were validated with visualizations, including correlation heatmaps and box plots, before the final dataset was saved for further analysis.

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Loading the dataset
file_path = "C:/data/Work/Data_processing/Processed_product_catalog.csv"
product_catalog_df = pd.read_csv(file_path)

# Check if it loaded successfully
print(product_catalog_df.head())

# Checking for missing values 
print("Missing values before processing:")
print(product_catalog_df.isnull().sum())

# Creating Sales_to_Return_Ratio
product_catalog_df["Sales_to_Return_Ratio"] = product_catalog_df["Sales Volume"] / (product_catalog_df["Return Rate"] + 1)  

# Creating Revenue per product
product_catalog_df["Revenue_Per_Product"] = product_catalog_df["Sales Volume"] * product_catalog_df["Price"]


#calculating the online sellability score
product_catalog_df["Online_Sellability_Score"] = product_catalog_df["Sellable Online"] * product_catalog_df["Sales Volume"] / (product_catalog_df["Sales Volume"].max() + 1)

# Check if it was calculated correctly
print("\n Unique values in 'Online Sellability Score':")
print(product_catalog_df["Online_Sellability_Score"].unique())

# Creating Storage Efficiency Score
product_catalog_df["Storage_Efficiency_Score"] = product_catalog_df["Revenue_Per_Product"] / (product_catalog_df["Storage Cost"] + 1)

# High Demand Indicator (Top 25% best-selling products)
threshold_sales = product_catalog_df["Sales Volume"].quantile(0.75)
product_catalog_df["High_Demand_Indicator"] = np.where(product_catalog_df["Sales Volume"] >= threshold_sales, 1, 0)

# Convert Implicit Feedback to numeric values
feedback_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
product_catalog_df["Implicit_Feedback_Score"] = product_catalog_df["Implicit Feedback"].map(feedback_mapping)

# Define Top Rated based on positive implicit feedback
product_catalog_df["Top_Rated"] = np.where(product_catalog_df["Implicit_Feedback_Score"] > 0, 1, 0)

# Likelihood of Purchase (How likely a product is to be purchased)
total_sales = product_catalog_df["Sales Volume"].sum()
product_catalog_df["Likelihood_of_Purchase"] = product_catalog_df["Sales Volume"] / (total_sales + 1)

# Implicit Feedback Score (Using numerical mapping for customer engagement)
product_catalog_df["Customer_Interest_Score"] = product_catalog_df["Implicit_Feedback_Score"]

# Expert Judgment Score (A weighted score for business impact)
product_catalog_df["Expert_Judgment_Score"] = (
    (product_catalog_df["High_Demand_Indicator"] * 1) +
    (product_catalog_df["Top_Rated"] * 1) +
    (product_catalog_df["Revenue_Per_Product"] > product_catalog_df["Revenue_Per_Product"].median()) * 1
)

# Ensure all numerical columns are correctly formatted
numerical_features = ["Sales_to_Return_Ratio", "Revenue_Per_Product", "Storage_Efficiency_Score",
                      "Online_Sellability_Score", "Likelihood_of_Purchase", "Customer_Interest_Score"]

# Convert columns to float for accurate visualizations
for col in numerical_features:
    product_catalog_df[col] = pd.to_numeric(product_catalog_df[col], errors='coerce')


# Generate heatmap with updated numeric columns
numeric_product_catalog_df = product_catalog_df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_product_catalog_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


#Feature Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(product_catalog_df[numerical_features].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(product_catalog_df["Sales_to_Return_Ratio"], bins=20, kde=True)
plt.title("Distribution of Sales-to-Return Ratio")
plt.xlabel("Sales-to-Return Ratio")
plt.ylabel("Frequency")
plt.show()

# Box Plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=product_catalog_df["Revenue_Per_Product"])
plt.title("Boxplot of Revenue Per Product")
plt.xlabel("Revenue Per Product")
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=product_catalog_df["Revenue_Per_Product"], y=product_catalog_df["High_Demand_Indicator"], alpha=0.6)
plt.title("High Demand Indicator vs. Revenue Per Product")
plt.xlabel("Revenue Per Product")
plt.ylabel("High Demand Indicator (1=High, 0=Low)")
plt.show()

# Bar chart for top-rated products
plt.figure(figsize=(8, 5))
sns.countplot(x=product_catalog_df["Top_Rated"], palette="coolwarm")
plt.title("Count of Top Rated Products")
plt.xlabel("Top Rated (1 = Yes, 0 = No)")
plt.ylabel("Number of Products")
plt.show()

# Save processed dataset
processed_file_path = "Feature_product_catalog.csv"
print("Processed file saved successfully at:", processed_file_path)
product_catalog_df.to_csv(processed_file_path, index=False)
