import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_paths = {
    "CiteULike": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens100K": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens1M": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens10M": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "Globo": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'"
}

zero_distributions = {}

for dataset_name, file_path in file_paths.items():
    df = pd.read_csv(file_path)
    pivot_table = df.pivot_table(index='user_id', columns='algorithm', values='score', fill_value=0)

    zero_counts = (pivot_table == 0).sum(axis=1)

    zero_distribution = zero_counts.value_counts().sort_index()

    zero_distributions[dataset_name] = zero_distribution

distribution_table = pd.DataFrame(zero_distributions).fillna(0).astype(int)
distribution_table.index.name = "Number of Algorithms with Zero Score"

distribution_table["Total"] = distribution_table.sum(axis=1)

distribution_table.to_csv("path to save the CSV file")
distribution_table.to_excel(
    "path to save the Excel file")

plt.figure(figsize=(12, 6))
colors = sns.color_palette("colorblind", len(distribution_table.index))
labels = [f"Zero in {int(index)} algorithms" for index in distribution_table.index]
sizes = distribution_table["Total"].values

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.tight_layout()
plt.savefig("path to save the figure",
            bbox_inches='tight',
            dpi=300)
plt.show()
