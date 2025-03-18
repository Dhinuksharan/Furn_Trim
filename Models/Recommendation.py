# Importing necessary libraries
import pandas as pd
import ast
import numpy as np
import random
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
#content based filtering
#Loading dataset
product_catalog_df =pd.read_csv ("C:/data/Work/Data_processing/Feature_Engineering/Feature_product_catalog.csv")

# Display basic information about the dataset
product_info = {
    "Shape": product_catalog_df.shape,
    "Columns": list(product_catalog_df.columns),
    "Missing Values": product_catalog_df.isnull().sum().to_dict(),
    "Sample Data": product_catalog_df.head()}
print(product_info)

# Select relevant columns for recommendations
selected_features = [
    "Product Name", "Category", "Material", "Size", "Special_features", 
    "Sales Volume", "Likelihood_of_Purchase", "Revenue_Per_Product", "Expert_Judgment_Score"
]

# Drop missing values
product_catalog_df = product_catalog_df.dropna(subset=selected_features)

# Normalize business-related numerical values using MinMaxScaler
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(product_catalog_df[["Sales Volume", "Likelihood_of_Purchase", "Revenue_Per_Product", "Expert_Judgment_Score"]])

# Add normalized values back to dataframe
product_catalog_df[["Sales_Score", "Purchase_Score", "Revenue_Score", "Expert_Score"]] = scaled_values

# Function to create feature strings with structured business metrics
def create_feature_string(row):
    return (
        f"{row['Category']} " * 4 + 
        f"{row['Material']} " * 3 +  
        f"{row['Special_features']} " * 2 +  
        f"{row['Size']} " * 1 +
        f"sales_{round(row['Sales_Score'], 2)} " + 
        f"purchase_{round(row['Purchase_Score'], 2)} " + 
        f"revenue_{round(row['Revenue_Score'], 2)} " + 
        f"expert_{round(row['Expert_Score'], 2)}"
    ).strip()

# Apply the function
product_catalog_df["Combined_Features"] = product_catalog_df.apply(create_feature_string, axis=1)

# Display sample feature combinations
print(product_catalog_df[["Product Name", "Combined_Features"]].head(10))

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1,2))

# Convert Combined_Features column into a TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(product_catalog_df["Combined_Features"])

# Display matrix shape
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# Compute cosine similarity between product TF-IDF vectors
cosine_sim = cosine_similarity(tfidf_matrix)

# Display shape of similarity matrix
print("Cosine Similarity Matrix Shape:", cosine_sim.shape)

# Show a sample of the similarity matrix
print("Sample Similarity Scores:\n", np.round(cosine_sim[:5, :5], 2))

def recommend_products(product_name, product_catalog_df, cosine_sim, top_n=5, min_similarity=0.1):
    print(f"\nSearching for recommendations for: {product_name}")
    
    # Find all indices of the selected product name
    product_indices = product_catalog_df[product_catalog_df["Product Name"] == product_name].index

    if product_indices.empty:
        print("Product not found in the dataset.")
        return [f"Product '{product_name}' not found in the catalog."]

    product_index = product_indices[0]
    similarity_scores = list(enumerate(cosine_sim[product_index]))

    # Sort by similarity and exclude itself
    sorted_products = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]
    
    recommendations = []
    seen_products = set()

    for idx, score in sorted_products:
        rec_product = product_catalog_df.iloc[idx]

        # Print similarity scores for debugging
        print(f"Checking Product: {rec_product['Product Name']} - Score: {score:.2f}")

        # Only add unique product names, avoiding duplicates
        if rec_product["Product Name"] not in seen_products and score >= min_similarity:
            recommendations.append(rec_product["Product Name"])
            seen_products.add(rec_product["Product Name"])

        if len(recommendations) == top_n:
            break

    if not recommendations:
        print("No strong matches found.")   
    
    return recommendations if recommendations else ["No strong matches found. Try another product."]


# Use Case 
# Find similar products for a given product
product_to_search = "Rustic Bookshelf"  # Change this to a real product name from your dataset
recommendations = recommend_products(product_to_search, product_catalog_df, cosine_sim)

# Display recommendations
print(f"\nRecommended Products for '{product_to_search}':")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")


