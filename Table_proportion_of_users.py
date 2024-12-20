import pandas as pd

file_paths = {
    "CiteULike": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens100K": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens1M": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "MovieLens10M": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'",
    "Globo": "path to the saved CSV file of corresponding dataset obtained from 'combining_ndcg.py'"
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

summary_output_path = "path to save the CSV file"
summary_df.to_csv(summary_output_path, index=False)
summary_df.to_excel("path to save the Excel file", index=False)
