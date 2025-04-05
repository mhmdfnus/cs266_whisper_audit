import pandas as pd
import matplotlib.pyplot as plt

# Load summary CSV
data_summary_file = "age_group_summary.csv"
df = pd.read_csv(data_summary_file)

# Sort age groups
age_order = ["teens", "twenties", "thirties", "fifties", "sixties"]
df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_order, ordered=True)
df = df.sort_values("Age Group")

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(df["Age Group"], df["WER"], color="skyblue", edgecolor="black")

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.2%}", ha='center', va='bottom', fontsize=10)

# Chart styling
plt.title("Misheard Word Rate by Age Group", fontsize=14)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Word Error Rate", fontsize=12)
plt.ylim(0, max(df["WER"]) * 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("misheard_rate_by_age.png")
plt.show()
