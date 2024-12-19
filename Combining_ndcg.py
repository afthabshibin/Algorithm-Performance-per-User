import pandas as pd


combined_data_path = 'C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens10M/MovieLens10M_combined_algorithms_sorted_ranking_data.csv'
ndcg_data_path = "C:/Users/aftha/OneDrive/Desktop/ndcg_values_MovieLens10M.xlsx"
output_path = 'C:/Users/aftha/OneDrive/Desktop/Final Graphs/MovieLens10M/Final_MovieLens10M_combined_algorithms_sorted_with_ndcg_all.csv'


combined_df = pd.read_csv(combined_data_path)
ndcg_df = pd.read_excel(ndcg_data_path)


ndcg_dict = dict(zip(ndcg_df['algorithm'], ndcg_df['ndcg@10']))


combined_df['ndcg@10'] = combined_df['algorithm'].map(ndcg_dict)


combined_df.to_csv(output_path, index=False)

