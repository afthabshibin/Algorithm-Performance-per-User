import pandas as pd


combined_data_path = 'path to the saved CSV file from "sorting ranking data.py"'
ndcg_data_path = "path to the overall ndcg data of the corresponding algorithm"
output_path = 'path to save the CSV file.'


combined_df = pd.read_csv(combined_data_path)
ndcg_df = pd.read_excel(ndcg_data_path)


ndcg_dict = dict(zip(ndcg_df['algorithm'], ndcg_df['ndcg@10']))


combined_df['ndcg@10'] = combined_df['algorithm'].map(ndcg_dict)


combined_df.to_csv(output_path, index=False)

