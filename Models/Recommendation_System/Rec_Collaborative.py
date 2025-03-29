# Import required libraries
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

#Collaborative Filtering
#Renaming "Suggested Products" to "Product Name" in the customer survey data

#Load the product catalog
product_catalog_df =pd.read_csv ("C:/data/Work/Data_processing/Feature_Engineering/Feature_product_catalog.csv")

# Load the customer survey dataset
customer_survey_df = pd.read_csv("C:/data/Work/Data_processing/Feature_Engineering/Feature_customer_survey.csv")


if "Suggested Products" in customer_survey_df.columns:
    customer_survey_df.rename(columns={"Suggested Products": "Product Name"}, inplace=True)

# Convert product lists stored as strings to actual lists
customer_survey_df["Product Name"] = customer_survey_df["Product Name"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [x]
)

# Expand product list so each product gets its own row
customer_survey_df = customer_survey_df.explode("Product Name")

# Clean product names (lowercase and remove unwanted spaces)
customer_survey_df["Product Name"] = customer_survey_df["Product Name"].str.lower().str.strip()
product_catalog_df["Product Name"] = product_catalog_df["Product Name"].str.lower().str.strip()

# Remove unmatched products
customer_survey_df = customer_survey_df[customer_survey_df["Product Name"].isin(product_catalog_df["Product Name"])]

# Keep only needed columns
survey_cols = ["Customer ID", "Product Name", "Recommendation Score", "Purchase Frequency", "Review_Sentiment", "Engagement_Sentiment"]
customer_survey_df = customer_survey_df[survey_cols]

catalog_cols = ["Product Name", "Implicit_Feedback_Score"]
product_catalog_df = product_catalog_df[catalog_cols]

# Merge both datasets
merged_df = pd.merge(customer_survey_df, product_catalog_df, on="Product Name", how="inner")

# Fill missing values
merged_df.fillna(0, inplace=True)


# Create Final Score (Combining Recommendation Score & Implicit Feedback Score)
merged_df["Final_Score"] = merged_df["Recommendation Score"].combine_first(merged_df["Implicit_Feedback_Score"])

# Create User-Product Interaction Matrix
interaction_matrix = merged_df.pivot_table(index="Customer ID",  # Rows = Customers
                                           columns="Product Name",  # Columns = Products
                                           values="Final_Score",  # Values = Scores
                                           aggfunc="mean")  # Average in case of duplicates

# Fill missing values 
interaction_matrix.fillna(0, inplace=True)

#  Display final interaction matrix
print("User-Product Interaction Matrix:")
print(interaction_matrix.head())

# Creating a binary version of the interaction matrix (1 = interacted, 0 = not interacted)
binary_matrix = (interaction_matrix > 0).astype(int)

# Filter out users who haven’t interacted with any products
binary_matrix = binary_matrix[binary_matrix.sum(axis=1) > 0]

# Ouput
print("Binary User-Product Matrix Sample:")
print(binary_matrix.head(10))

# Transpose interaction matrix to get Products as rows and Customers as columns
item_user_matrix = interaction_matrix.T

# Calculate cosine similarity between products
item_similarity = cosine_similarity(item_user_matrix)

# Store it in a DataFrame for easier lookup
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user_matrix.index,
    columns=item_user_matrix.index
)

# Output
print("Item-Item Similarity Matrix (Sample):")
print(item_similarity_df.iloc[:5, :5])


def recommend_similar_items(product_name, top_n=5):
    if product_name not in item_similarity_df.index:
        print(f"Product '{product_name}' not found in the data.")
        return []

    # Sort the similarity scores and skip the product itself
    similar_scores = item_similarity_df[product_name].sort_values(ascending=False)
    top_matches = similar_scores.iloc[1:top_n + 1]

    print(f"\nProducts similar to '{product_name}':")
    for item, score in top_matches.items():
        print(f"- {item} (Score: {score:.2f})")

    return top_matches.index.tolist()

# Example 
recommend_similar_items("rustic bookshelf")

