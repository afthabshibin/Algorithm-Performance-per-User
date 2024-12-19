import pandas as pd


def plotting_values(algorithm_names):

    combined_df = pd.DataFrame()


    for algorithm_name in algorithm_names:

        df = pd.read_excel(
            f'path to current file'
        )


        df_sorted = df.sort_values(by='score', ascending=False)


        df_sorted['rank'] = range(1, len(df_sorted) + 1)


        df_sorted['algorithm'] = algorithm_name


        combined_df = pd.concat([combined_df, df_sorted], ignore_index=True)


    combined_df.to_csv(
        'C:/Users/aftha/OneDrive/Desktop/Final '
        'Graphs/MovieLens100K/MovieLens100K_combined_algorithms_sorted_ranking_data.csv',
        index=False
    )


if __name__ == '__main__':

    algorithms = ['ItemKNN', 'KUNN', 'NMF', 'NMFItemToItem', 'SVDItemToItem', 'SVD', 'SLIM']


    plotting_values(algorithms)
