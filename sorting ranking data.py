import pandas as pd


def combining(algorithm_names):
    combined_df = pd.DataFrame()
    """
    Ensure that pipeline.py names the Excel file name with corresponding 
    algorithm names 
    """
    for algorithm_name in algorithm_names:

        excel_file = f'{algorithm_name}.xlsx'

        df = pd.read_excel(excel_file)

        df_sorted = df.sort_values(by='score', ascending=False)

        df_sorted['rank'] = range(1, len(df_sorted) + 1)

        df_sorted['algorithm'] = algorithm_name

        combined_df = pd.concat([combined_df, df_sorted], ignore_index=True)

    combined_df.to_csv(
        'path to save the CSV file',
        index=False
    )


if __name__ == '__main__':
    algorithms = ['ItemKNN', 'KUNN', 'NMF', 'NMFItemToItem', 'SVDItemToItem', 'SVD', 'SLIM']

    combining(algorithms)
