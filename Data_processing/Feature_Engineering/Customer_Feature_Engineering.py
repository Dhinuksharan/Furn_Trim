#The feature engineering script builds on the processed dataset to create meaningful new attributes for customer behavior analysis. 
#I mapped categorical variables like Future Plan, Budget, and Purchase Frequency into numerical scores. 
# Additional features were introduced, including Premium_Buyer (indicating high-budget customers with a preference for luxury),
#  Customer_Engagement_Score (combining purchase tendency and recommendation score),and Preference_Score (integrating budget, future plans, and activity preferences).
# To explore relationships, I generated visualizations such as heatmaps, scatter plots, and histograms.
#  The final dataset, enriched with insights, was saved for further business analysis and modeling.

#Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler

#Loading the dataset
file_path = "C:/data/Work/Data_processing/Processed_Customer_survey.csv"
customer_survey_df = pd.read_csv(file_path)

#Check if the loaded successfully
print(customer_survey_df.head())

#Converting 'Future Plan' into numerical categories 
future_plan_mapping = {"Relax at home": 1, "Work from home": 2, "Move": 3, "Travel": 4}
customer_survey_df["Future_Plan_Score"] = customer_survey_df["Future Plan"].map(future_plan_mapping)

#Creating a budget score
budget_mapping = {"low": 0, "medium": 1, "high": 2}
customer_survey_df["Budget_Score"] = customer_survey_df["Budget"].map(budget_mapping)

# Fill NaN values with the median Budget Score
customer_survey_df["Budget_Score"].fillna(customer_survey_df["Budget_Score"].median(), inplace=True)

# Converting Customer Reviews sentiment scores into positive/negative classes
customer_survey_df["Review_Sentiment"] = np.where(customer_survey_df["Customer Reviews"] >= 0, 1, 0)

if "Purchase Frequency" in customer_survey_df.columns:
    purchase_freq_mapping = {"rarely": 0, "occasionally": 1, "monthly": 2}
    customer_survey_df["Purchase_Tendency"] = customer_survey_df["Purchase Frequency"].map(purchase_freq_mapping)

   
    customer_survey_df["Purchase_Tendency"] = customer_survey_df["Purchase_Tendency"].fillna(0)
else:
    print("Purchase Frequency column is missing Setting Purchase_Tendency to 0 for all rows.")
    customer_survey_df["Purchase_Tendency"] = 0  

# Creating a feature indicating whether customers are likely to buy premium products
customer_survey_df["Premium_Buyer"] = (customer_survey_df["Budget_Score"] == 2) & (customer_survey_df["Preferred Style"] == "Luxurious")
customer_survey_df["Premium_Buyer"] = customer_survey_df["Premium_Buyer"].astype(int)

# Encode Main Activity into numerical categories
main_activity_mapping = {"Gaming": 1, "Watching TV": 2, "Reading": 3, "Exercising": 4}
customer_survey_df["Main_Activity_Score"] = customer_survey_df["Main Activity"].map(main_activity_mapping)

#  Extracting keyword-based interest score (to align with recommendation system)
customer_survey_df["Interest_Keywords"] = customer_survey_df["Key Desires"].str.lower()

#  Converting Customer Review text into numerical sentiment score 
customer_survey_df["Review_Text_Processed"] = customer_survey_df["Customer Reviews"]

# Preference-based Recommendation Score 
customer_survey_df["Preference_Score"] = (
    customer_survey_df["Budget_Score"] * 0.4 +
    customer_survey_df["Future_Plan_Score"] * 0.3 +
    customer_survey_df["Main_Activity_Score"] * 0.3
)

# Check if Preference_Score is entirely NaN
if customer_survey_df["Preference_Score"].dropna().empty:
    default_median = 0  
else:
    default_median = customer_survey_df["Preference_Score"].median()

# Fill missing values safely
customer_survey_df["Preference_Score"] = customer_survey_df["Preference_Score"].fillna(default_median)

# Customer Engagement Score 
customer_survey_df["Customer_Engagement_Score"] = (
    customer_survey_df["Purchase_Tendency"].fillna(0) * 0.5 +
    customer_survey_df["Recommendation Score"].fillna(customer_survey_df["Recommendation Score"].median()) * 0.5
)

# Low Interest Indicator-If Purchase Frequency is low & Review Sentiment is negative)
customer_survey_df["Low_Interest_Indicator"] = np.where(
    (customer_survey_df["Purchase_Tendency"] == 0) & (customer_survey_df["Review_Sentiment"] == 0), 1, 0
)

# Sentiment from Recommendation Score
customer_survey_df["Sentiment_Score"] = np.where(
    customer_survey_df["Recommendation Score"] > 50, 1, 0  
)

# Engagement-Based Sentiment 
if not customer_survey_df["Customer_Engagement_Score"].dropna().empty:
    median_engagement = customer_survey_df["Customer_Engagement_Score"].median()
else:
    median_engagement = 0  

customer_survey_df["Engagement_Sentiment"] = np.where(
    customer_survey_df["Customer_Engagement_Score"].fillna(0) > median_engagement, 1, 0
)

# Distribution of Budget Scores
plt.figure(figsize=(8, 5))
sns.countplot(x="Budget_Score", data=customer_survey_df, palette="viridis")
plt.title("Distribution of Customer Budget Preferences")
plt.xlabel("Budget Score (0 = Low, 1 = Medium, 2 = High)")
plt.ylabel("Count")
plt.show()

# Dropping customer ID for the heatmap
valid_numerical_features = customer_survey_df.select_dtypes(include=[np.number]).drop(columns=["Customer ID"]).nunique()
valid_numerical_features = valid_numerical_features[valid_numerical_features > 1].index
corr_matrix = customer_survey_df[valid_numerical_features].corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap (Without Customer ID)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(x="Future_Plan_Score", y="Budget_Score", data=customer_survey_df, palette="coolwarm")
plt.title("Future Plans vs. Budget Score")
plt.xlabel("Future Plan (1 = Relax, 2 = Work, 3 = Move, 4 = Travel)")
plt.ylabel("Budget Score")
plt.show()

#histogram
plt.figure(figsize=(8,5))
sns.histplot(customer_survey_df["Recommendation Score"], bins=20, kde=True, color="blue")
plt.title("Distribution of Recommendation Scores")
plt.xlabel("Recommendation Score")
plt.ylabel("Frequency")
plt.show()

#scatterplot
plt.figure(figsize=(8,5))
sns.scatterplot(x="Purchase_Tendency", y="Customer_Engagement_Score", data=customer_survey_df, hue="Sentiment_Score", palette="coolwarm")
plt.title("Purchase Tendency vs. Customer Engagement")
plt.xlabel("Purchase Tendency (0 = rarely, 1 = occasionally, 2 = monthly)")
plt.ylabel("Customer Engagement Score")
plt.show()


plt.figure(figsize=(8,5))
sns.regplot(x="Future_Plan_Score", y="Preference_Score", data=customer_survey_df, scatter_kws={"alpha":0.6}, line_kws={"color":"red"})
plt.title("Future Plan Score vs. Preference Score")
plt.xlabel("Future Plan Score")
plt.ylabel("Preference Score")
plt.show()


processed_file_path = "Feature_customer_survey.csv"
print("Processed file saved successfully at:", processed_file_path)
customer_survey_df.to_csv(processed_file_path, index=False)

