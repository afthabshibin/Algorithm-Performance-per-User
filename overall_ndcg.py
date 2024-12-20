# Import necessary modules from RecPack and other libraries
from recpack.datasets import MovieLens100K   #Specify the dataset to use
from recpack.scenarios import WeakGeneralization
from recpack.pipelines import PipelineBuilder
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
torch.backends.cudnn.deterministic = True   # Ensure deterministic results
torch.backends.cudnn.benchmark = False   # Disable benchmarking to prevent randomness in CUDA operations

# Load and preprocess the dataset
dataset = MovieLens100K(path="path")   # Specify dataset and the path
dataset.fetch_dataset()   # check if a file containing the dataset is present and download it if it is not
interaction_matrix = dataset.load()   # To preprocess data

# Define a weak generalization scenario and split the dataset
scenario = WeakGeneralization(0.75, validation=True)
scenario.split(interaction_matrix)

builder = PipelineBuilder()   # Initialize the pipeline builder
builder.set_data_from_scenario(scenario)   # To use training,validation and test dataset

# Add algorithm and its parameter grid
builder.add_algorithm('ItemKNN', grid={
    'K': [100, 200, 500],   # Number of neighbors
    'similarity': ['cosine', 'conditional_probability'],   # Similarity measures
    'pop_discount': [None, 0.5, 0.7],   # Power applied to the comparing item in the denominator
    'normalize_X': [True, False],   # Normalize rows in the interaction matrix
    'normalize_sim': [True, False]   # Normalize scores per row in the similarity matrix
})

builder.add_algorithm('SLIM', grid={
    'l1_reg': [.0003, .0005, .0007],   # L1 regularization coefficient
    'l2_reg': [.00003, .00005, .00007],   # L2 regularization coefficient
    'fit_intercept': [True],   # Whether the intercept should be estimated or not during gradient descent
    'ignore_neg_weights': [True]   # Remove negative weights
})

builder.add_algorithm('NMFItemToItem', grid={
    'num_components': [10, 50, 100, 200]   # The size of the latent dimension
})

builder.add_algorithm('SVDItemToItem', grid={
    'num_components': [10, 50, 100, 200]   # The size of the latent dimension
})

builder.add_algorithm('NMF', grid={
    'num_components': [50, 100, 200],   # The size of the latent dimension
    'alpha': [.0001, .001, .1],   # Regularization parameter
    'l1_ratio': [0, .5, 1]   # Defines how much L1 normalisation is used, compared to L2 normalisation
})

builder.add_algorithm('SVD', grid={
    'num_components': [50, 100, 200],   # The size of the embeddings
})

builder.add_algorithm('KUNN', grid={
    'Ku': [50, 100, 200],   # Number of neighbors to keep in the user similarity matrix
    'Ki': [50, 100, 200]   # Number of neighbours in the item similarity matrix
})

# Set optimization metric for parameters and additional metric for evaluating the algorithm
builder.set_optimisation_metric('NDCGK', K=10)
builder.add_metric('NDCGK', K=[3, 5, 10, 20])

# Build and run the pipeline
pipeline = builder.build()
pipeline.run()

# Retrieve and print calculated metrics
metrics = pipeline.get_metrics()
print(pipeline.get_num_users())

# Reset the index of the metrics DataFrame for saving with algorithm names and parameter values
metrics.reset_index(inplace=True)


# Save per-user NDCG scores to an Excel file and csv file
metrics.to_csv(f"path to save csv file", index=False)
metrics.to_excel(f"path to save excel file", index=False)
print(metrics)

