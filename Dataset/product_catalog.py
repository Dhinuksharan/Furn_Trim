import pandas as pd
import random

# Define category-wise product names
category_mapping = {
    "Sofas & Seating": ["2 Seater sofa", "Chaise Sofas", "Upholstered armchair", "Minimalist Sectional Sofa", "Rattan Armchair", "Luxury Bean Bag"],
    "Tables": ["Bistro Tables", "Glass Dining Table", "Wooden Side Table", "Industrial Console Table", "Smart Office Desk", "Adjustable Standing Desk"],
    "Storage & Organization": ["Rustic Bookshelf", "Modern TV Cabinet", "Sliding Door Wardrobe", "Compact Shoe Rack", "Ottoman Storage Bench"],
    "Bedroom Furniture": ["King Size Bed Frame", "Queen Bed with Storage", "Classic Nightstand", "Large Wooden Dresser", "Open Closet Organizer"],
    "Kitchen & Dining": ["Adjustable Bar Stool", "Padded Dining Chair", "Kitchen Pantry Shelf", "Expandable Kitchen Island"],
    "Outdoor Furniture": ["Outdoor Patio Chair", "Garden Lounge Bench", "6-Piece Outdoor Dining Set", "Hanging Hammock", "Foldable Sun Lounger"]
}

# Define other attribute options
materials = {"2 Seater sofa": "Leather", "Chaise Sofas": "Fabric", "Upholstered armchair": "Plastic",
             "Minimalist Sectional Sofa": "Fabric", "Rattan Armchair": "Velvet", "Luxury Bean Bag": "Foam",
             "Bisto Tables": "Wood", "Glass Dining Table": "Glass", "Wooden Side Table": "Wood",
             "Industrial Console Table": "Metal", "Smart Office Desk": "Wood", "Adjustable Standing Desk": "Metal",
             "Rustic Bookshelf": "Wood", "Modern TV Cabinet": "Wood", "Sliding Door Wardrobe": "Wood",
             "Compact Shoe Rack": "Plastic", "Ottoman Storage Bench": "Fabric", "King Size Bed Frame": "Wood",
             "Queen Bed with Storage": "Wood", "Classic Nightstand": "Wood", "Large Wooden Dresser": "Wood",
             "Open Closet Organizer": "Plastic", "Adjustable Bar Stool": "Metal", "Padded Dining Chair": "Fabric",
             "Kitchen Pantry Shelf": "Wood", "Expandable Kitchen Island": "Wood", "Outdoor Patio Chair": "Metal",
             "Garden Lounge Bench": "Wood", "6-Piece Outdoor Dining Set": "Wood", "Hanging Hammock": "Fabric",
             "Foldable Sun Lounger": "Plastic"}

sizes = ["Small", "Medium", "Large"]
special_features= ["Foldable", "Ergonomic", "Minimalist", "Smart", "Durable", "Waterproof"]
customer_types = ["Normal", "Member"]
payment_types = ["Credit Card", "Debit Card","Cash on Delivery"]
sellable_online_options = ["Yes", "No"]

# Generate dataset with 100 entries
num_entries = 100
data = []
for product_id in range(1, num_entries + 1):
    category = random.choice(list(category_mapping.keys()))
    product_name = random.choice(category_mapping[category])
    material = materials.get(product_name, "Unknown")  # Ensure consistent material for each product
    data.append([
        product_id,  # Numeric Product ID
        product_name,
        category,
        material,
        random.choice(sizes),
        random.choice(special_features),
        random.choice(customer_types),
        round(random.uniform(50, 5000), 2),  # Price between $50 and $5000
        random.choice(payment_types),
        random.choice(sellable_online_options),
        random.randint(10, 1000),  # Sales volume
        round(random.uniform(0, 20), 2),  # Return Rate 
        round(random.uniform(5, 200), 2),  # Storage Cost 
        random.randint(1, 10),  # Seasonality Score (1-10)
        random.choice(["Positive", "Neutral", "Negative"])  # Implicit Feedback
    ])

# Create DataFrame
columns = [
    "Product ID", "Product Name", "Category", "Material", "Size", "Special_features", 
    "Customer Type", "Price", "Payment Type", "Sellable Online", 
    "Sales Volume", "Return Rate", "Storage Cost", "Seasonality Score", "Implicit Feedback"
]
product_catalog_df = pd.DataFrame(data, columns=columns)

# Save to CSV
product_catalog_df.to_csv("product_catalog_new.csv", index=False)

print("Product Catalog dataset generated and saved ")
