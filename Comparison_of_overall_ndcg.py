import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

file_paths = {
    "CiteULike": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens100K": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens1M": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens10M": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "Globo": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'"
}

consolidated_data = []

for dataset_name, path in file_paths.items():
    df = pd.read_csv(path)
    if 'ndcg@10' in df.columns and 'algorithm' in df.columns:
        for algorithm in df['algorithm'].unique():
            overall_ndcg = df[df['algorithm'] == algorithm]['ndcg@10'].iloc[0]
            consolidated_data.append({'Dataset': dataset_name, 'Algorithm': algorithm, 'NDCG@10': overall_ndcg})

consolidated_df = pd.DataFrame(consolidated_data)
print(consolidated_df)

plt.figure(figsize=(12, 6))

for algorithm in consolidated_df['Algorithm'].unique():
    subset = consolidated_df[consolidated_df['Algorithm'] == algorithm]
    sns.lineplot(x=subset['Dataset'], y=subset['NDCG@10'], marker='o', linewidth=3, markersize=8, label=algorithm)

plt.xlabel("Datasets", fontsize=12)
plt.ylabel("NDCG@10", fontsize=12)

plt.legend(title="Algorithms", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("path to save the figure",
            bbox_inches='tight',
            dpi=300)
plt.show()
