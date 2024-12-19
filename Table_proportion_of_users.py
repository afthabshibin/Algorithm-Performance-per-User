import pandas as pd

file_paths = {
    "CiteULike": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/CiteULike/Final_CiteULike_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens100K": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens100K/Final_MovieLens100K_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens1M": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens1M/Final_MovieLens1M_combined_algorithms_sorted_with_ndcg_all.csv",
    "MovieLens10M": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens10M/Final_MovieLens10M_combined_algorithms_sorted_with_ndcg_all.csv",
    "Globo": "C:/Users/aftha/OneDrive/Desktop/Final Graphs/Globo/Final_Globo_combined_algorithms_sorted_with_ndcg_all.csv"
}

summary_data = []
threshold = 0.1

for dataset, path in file_paths.items():
    df = pd.read_csv(path)
    grouped = df.groupby("algorithm")
    for algo, group in grouped:
        num_users_score_1 = (group['score'] == 1).sum()
        num_users_score_0 = (group['score'] == 0).sum()
        total_users = len(group)

        overall_ndcg = group['ndcg@10'].iloc[0]

        close_to_Overall_ndcg = ((group['score'] - overall_ndcg).abs() <= threshold).sum()

        summary_data.append({
            "Dataset": dataset,
            "Algorithm": algo,
            "Users with Score = 0": num_users_score_0,
            "Proportion of Users with Score = 0": num_users_score_0 / total_users,
            "Users with Score = 1": num_users_score_1,
            "Proportion of Users with Score = 1": num_users_score_1 / total_users,
            "Users with score close to Overall ndcg": close_to_Overall_ndcg,
            "Proportion Close to Overall NDCG@10": close_to_Overall_ndcg / total_users,
        })


summary_df = pd.DataFrame(summary_data)

summary_output_path = "C:/Users/aftha/OneDrive/Desktop/Final Graphs/Final_summary_results_proportional.csv"
summary_df.to_csv(summary_output_path, index=False)
summary_df.to_excel("C:/Users/aftha/OneDrive/Desktop/Final Graphs/Final_summary_results_proportional.xlsx", index=False)
