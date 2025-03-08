import pandas as pd
import random
import uuid

# Define attribute options
customer_types = ["Normal", "Member"]
personalities = ["Minimalist", "Trendy", "Practical"]
house_sizes = ["Small Apartment", "Medium House", "Large House"]
key_desires = ["Comfort", "Aesthetic", "Durability"]
future_plans = ["Relax at home", "Work from home", "Move", "Travel"]
main_activities = ["Gaming", "Watching TV", "Reading", "Exercising"]
budgets = ["Low", "Medium", "High"]
preferred_styles = ["Cozy", "Minimalist", "Luxurious", "Modern"]
purchase_frequencies = ["Monthly", "Occasionally", "Rarely"]

# Define suggested products based on future plans
suggested_products_mapping = {
    "Relax at home":["2 Seater sofa", "Chaise Sofas", "Upholstered armchair", "Minimalist Sectional Sofa", "Rattan Armchair", "Luxury Bean Bag",  "Queen Bed with Storage"],
    "Work from home":["Smart Office Desk", "Adjustable Standing Desk", "Rustic Bookshelf", "Padded Dining Chair"],
    "Move":["Sliding Door Wardrobe", "Compact Shoe Rack",  "Ottoman Storage Bench", "6-Piece Outdoor Dining Set"],
    "Travel":["Travel Bags", "Portable Furniture", "Storage Organizers",  "Outdoor Patio Chair", "Foldable Sun Lounger"]
}


# Generate dataset with 100 entries
num_entries = 100
data = []
for customer_id in range(1, num_entries + 1):
    future_plan = random.choice(future_plans)
    suggested_products = random.sample(suggested_products_mapping[future_plan], min(2, len(suggested_products_mapping[future_plan])))
    suggested_products_str = ", ".join(suggested_products)

    data.append([
        customer_id,  # Numeric Customer ID
        random.choice(customer_types),
        random.choice(personalities),
        random.choice(house_sizes),
        random.choice(key_desires),
        future_plan,
        random.choice(main_activities),
        random.choice(budgets),
        random.choice(preferred_styles),
        random.choice(purchase_frequencies),
        random.randint(0, 100),  # Recommendation Score (0-100%)
        round(random.uniform(-1, 1), 2),  # Customer Review Sentiment (-1 to 1)
        suggested_products  # Suggested Products based on future plan
    ])

# Create DataFrame
columns = [
    "Customer ID", "Customer Type", "Personality", "House Size", "Key Desires", 
    "Future Plan", "Main Activity", "Budget", "Preferred Style", "Purchase Frequency", 
    "Recommendation Score", "Customer Reviews", "Suggested Products"
]
customer_survey_df = pd.DataFrame(data, columns=columns)

# Save to CSV
customer_survey_df.to_csv("customer_survey_new.csv", index=False)

print("Customer Survey dataset generated")
