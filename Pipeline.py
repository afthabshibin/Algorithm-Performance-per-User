# Import necessary modules from RecPack and other libraries
from recpack.datasets import CiteULike  #Specify the dataset to use
from recpack.scenarios import WeakGeneralization
from recpack.pipelines import PipelineBuilder, Pipeline
from recpack.metrics.dcg import NDCGK
from recpack.algorithms import KUNN, ItemKNN, SLIM, NMFItemToItem, NMF, SVDItemToItem, SVD
import random
import numpy as np
import torch

# Set random seeds for reproducibility across random, numpy, and torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True  # Ensure deterministic results
torch.backends.cudnn.benchmark = False  # Disable benchmarking to prevent randomness in CUDA operations


# Define the function to build and run the recommendation pipeline
def run_pipeline(algorithm_name, algorithm_params, ndcg_k, file_prefix):
    """
    Configures and runs a pipeline for a specified recommendation algorithm.

    Args:
        algorithm_name (str): Name of the algorithm to use.
        algorithm_params (dict): Parameter grid for the algorithm.
        ndcg_k (int): Cut-off for NDCG metric.
        file_prefix (str): Prefix for saving results to files.
    """

    builder = PipelineBuilder()   # Initialize the pipeline builder
    builder.set_data_from_scenario(scenario)   # To use training,validation and test dataset

    # Add algorithm and its parameter grid
    builder.add_algorithm(algorithm_name, grid=algorithm_params)

    # Set optimization metric for parameters and additional metric for evaluating the algorithm
    builder.set_optimisation_metric('NDCGK', K=ndcg_k)
    builder.add_metric('NDCGK', K=[3, 5, 10, 20])

    # Build and run the pipeline
    pipeline: Pipeline = builder.build()
    pipeline.run()

    # Retrieve and print calculated metrics
    metrics = pipeline.get_metrics()
    print(metrics)

    # Calculate NDCG metric specifically for K=10
    ndcg_metric = NDCGK(K=10)

    # Instantiate the algorithm class and fit it to training and validation training data
    algorithm_class = globals()[algorithm_name]
    algorithm = algorithm_class()
    algorithm.fit(scenario.validation_training_data.binary_values)
    algorithm.fit(scenario.full_training_data.binary_values)

    # Predict and calculate per-user NDCG score for the algorithm
    y_pred = pipeline._predict_and_postprocess(algorithm, scenario.test_data_in)
    ndcg_metric.calculate(scenario.test_data_out.binary_values, y_pred)

    # Save per-user NDCG scores to an Excel file
    per_user_ndcg_scores = ndcg_metric.results
    per_user_ndcg_scores.to_excel(
        f'path to save the excel file', index=False)

    # Print per-user NDCG scores
    print(per_user_ndcg_scores)


# Load and preprocess the dataset
dataset = CiteULike(path='path to the dataset')  # Specify dataset and the path
dataset.fetch_dataset()  # check if a file containing the dataset is present and download it if it is not
interaction_matrix = dataset.load()  # To preprocess data

# Define a weak generalization scenario and split the dataset
scenario = WeakGeneralization(0.75, validation=True, seed=42)
scenario.split(interaction_matrix)

# Run the pipeline for different recommendation algorithms

run_pipeline(
    algorithm_name='ItemKNN',
    algorithm_params={
        'K': [100, 200, 500],  # Number of neighbors
        'similarity': ['cosine', 'conditional_probability'],  # Similarity measures
        'pop_discount': [None, 0.5, 0.7],  # Power applied to the comparing item in the denominator
        'normalize_X': [True, False],  # Normalize rows in the interaction matrix
        'normalize_sim': [True, False]  # Normalize scores per row in the similarity matrix
    },
    ndcg_k=10,
    file_prefix='ItemKNN_CiteULike'
)

run_pipeline(
    algorithm_name='SLIM',
    algorithm_params={
        'l1_reg': [.0003, .0005, .0007],  # L1 regularization coefficient
        'l2_reg': [.00003, .00005, .00007],  # L2 regularization coefficient
        'fit_intercept': [True],  # Whether the intercept should be estimated or not during gradient descent
        'ignore_neg_weights': [True]  # Remove negative weights
    },
    ndcg_k=10,
    file_prefix='SLIM_CiteULike'
)


run_pipeline(
    algorithm_name='NMFItemToItem',
    algorithm_params={
        'num_components': [10, 50, 100, 200]  # The size of the latent dimension
    },
    ndcg_k=10,
    file_prefix='NMFItemToItem_CiteULike'
)


run_pipeline(
    algorithm_name='SVDItemToItem',
    algorithm_params={
        'num_components': [10, 50, 100, 200]  # The size of the latent dimension
    },
    ndcg_k=10,
    file_prefix='SVDItemToItem_CiteULike'
)


run_pipeline(
    algorithm_name='NMF',
    algorithm_params={
        'num_components': [50, 100, 200],  # The size of the latent dimension
        'alpha': [.0001, .001, .1],  # Regularization parameter
        'l1_ratio': [0, .5, 1]  # Defines how much L1 normalisation is used, compared to L2 normalisation
    },
    ndcg_k=10,
    file_prefix='NMF_CiteULike'
)

run_pipeline(
    algorithm_name='SVD',
    algorithm_params={
        'num_components': [50, 100, 200],  # The size of the embeddings
    },
    ndcg_k=10,
    file_prefix='SVD_CiteULike'
)


run_pipeline(
    algorithm_name='KUNN',
    algorithm_params={
        'Ku': [50, 100, 200],  # Number of neighbors to keep in the user similarity matrix
        'Ki': [50, 100, 200]  # Number of neighbours in the item similarity matrix
    },
    ndcg_k=10,
    file_prefix='KUNN_CiteULike'
)
