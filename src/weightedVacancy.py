import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load data
leases_df = pd.read_csv("../../data/Leases.csv")
occupancy_df = pd.read_csv("../../data/Major_Market_Occupancy_Data-revised.csv")

# Prep for merge
leases_df["year_quarter"] = leases_df["year"].astype(str) + " " + leases_df["quarter"]
occupancy_df["year_quarter"] = occupancy_df["year"].astype(str) + " " + occupancy_df["quarter"]

# Merge
merged_df = pd.merge(
    leases_df,
    occupancy_df[["market", "year_quarter", "avg_occupancy_proportion"]],
    on=["market", "year_quarter"],
    how="inner"
)

# Normalize rent
scaler = MinMaxScaler()
merged_df["normalized_rent"] = scaler.fit_transform(merged_df[["overall_rent"]])

# Weighted vacancy score (70% vacancy, 30% normalized rent)
merged_df["weighted_vacancy_score"] = (
    (1 - merged_df["avg_occupancy_proportion"]) + merged_df["normalized_rent"] 
)

# Aggregate for heatmap
heatmap_data = (
    merged_df.groupby(['market', 'year_quarter'])['weighted_vacancy_score']
    .mean()
    .reset_index()
    .pivot(index='market', columns='year_quarter', values='weighted_vacancy_score')
)

# Plot
plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_data, cmap='OrRd', linewidths=0.5, linecolor='gray')
plt.title("Weighted Vacancy Score by Market and Quarter", fontsize=16)
plt.xlabel("Year - Quarter")
plt.ylabel("Market")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save plot
plt.savefig("../../plots/normalized_vacancy_score_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()