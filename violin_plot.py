import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_paths = {
    "CiteULike": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/CiteULike/Final_CiteULike_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens100K": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens100K/Final_MovieLens100K_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens1M": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens1M/Final_MovieLens1M_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens10M": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens10M/Final_MovieLens10M_combined_algorithms_sorted_with_ndcg_all.csv",
    "Globo": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/Globo/Final_Globo_combined_algorithms_sorted_with_ndcg_all.csv"
}


dataframes = []
for dataset_name, file_path in file_paths.items():
    df = pd.read_csv(file_path)
    df['Dataset'] = dataset_name
    dataframes.append(df)


combined_df = pd.concat(dataframes, ignore_index=True)


combined_df.head()

plt.figure(figsize=(12, 6))

overall_ndcg = combined_df.groupby(['algorithm', 'Dataset'])['ndcg@10'].mean().reset_index()

algorithm_order = combined_df['algorithm'].unique()
overall_ndcg['algorithm'] = pd.Categorical(overall_ndcg['algorithm'], categories=algorithm_order, ordered=True)
sorted_overall_ndcg = overall_ndcg.sort_values('algorithm')
print(sorted_overall_ndcg)
print(overall_ndcg)


print(overall_ndcg)
sns.violinplot(
    data=combined_df,
    x="algorithm",
    y="score",
    hue="Dataset",
    inner="quart",
    density_norm='width',
    alpha=0.7,
    split=True,
)

for dataset in overall_ndcg['Dataset'].unique():
    dataset_data = sorted_overall_ndcg[sorted_overall_ndcg['Dataset'] == dataset]
    plt.plot(
        dataset_data['algorithm'],
        dataset_data['ndcg@10'],
        marker='o',
        label=f"Overall NDCG ({dataset})"
    )


plt.xlabel("Algorithm", fontsize=12)
plt.ylabel("NDCG Score", fontsize=12)
plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig(
    'C:/Users/aftha/OneDrive/Desktop/Final Graphs/violin_plot_ndcg_scores_overall.jpg',
    bbox_inches='tight',
    dpi=300
)

plt.show()

