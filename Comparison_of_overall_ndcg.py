import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

file_paths = {
    "CiteULike": "C:/Users/aftha/OneDrive/Desktop/data/all/CiteULike_combined_algorithms_sorted_ranking_data_all.csv",
    "MovieLens100K": "C:/Users/aftha/OneDrive/Desktop/data/all/MovieLens100K_combined_algorithms_sorted_ranking_data_all.csv",
    "MovieLens1M": "C:/Users/aftha/OneDrive/Desktop/data/all/MovieLens1M_combined_algorithms_sorted_ranking_data_all.csv",
    "MovieLens10M": "C:/Users/aftha/OneDrive/Desktop/data/all/MovieLens10M_combined_algorithms_sorted_ranking_data_all.csv",
    "Globo": "C:/Users/aftha/OneDrive/Desktop/data/all/Globo_combined_algorithms_sorted_ranking_data_all.csv"
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
plt.savefig("C:/Users/aftha/OneDrive/Desktop/Final Graphs/Performance Consistency of Algorithms Across Datasets.jpg",
            bbox_inches='tight',
            dpi=300)
plt.show()
