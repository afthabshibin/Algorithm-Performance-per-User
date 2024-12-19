import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_paths = {
    "CiteULike": "C:/Users/aftha/OneDrive/Desktop/Final "
                 "Graphs/CiteULike/Final_CiteULike_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens100K": "C:/Users/aftha/OneDrive/Desktop/Final "
                     "Graphs/MovieLens100K/Final_MovieLens100K_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens1M": "C:/Users/aftha/OneDrive/Desktop/Final "
                   "Graphs/MovieLens1M/Final_MovieLens1M_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens10M": "C:/Users/aftha/OneDrive/Desktop/Final "
                    "Graphs/MovieLens10M/Final_MovieLens10M_combined_algorithms_sorted_with_ndcg_all.csv",
    "Globo": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/Globo/Final_Globo_combined_algorithms_sorted_with_ndcg_all"
             ".csv"
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

distribution_table.to_csv("C:/Users/aftha/OneDrive/Desktop/Final Graphs/Aggregated_Zero_Score_Distribution_Table.csv")
distribution_table.to_excel(
    "C:/Users/aftha/OneDrive/Desktop/Final Graphs/Aggregated_Zero_Score_Distribution_Table.xlsx")

plt.figure(figsize=(12, 6))
colors = sns.color_palette("colorblind", len(distribution_table.index))
labels = [f"Zero in {int(index)} algorithms" for index in distribution_table.index]
sizes = distribution_table["Total"].values

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.tight_layout()
plt.savefig("C:/Users/aftha/OneDrive/Desktop/Final Graphs/Aggregated_Zero_Score_Distribution.jpg",
            bbox_inches='tight',
            dpi=300)
plt.show()
