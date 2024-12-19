# Algorithm-Performance-per-User
This project aims to evaluate the performance of different recommendation algorithms at an user level across multiple datasets. By calculating the Normalized Discounted Cumulative Gain (NDCG) for each user across all algorithm-dataset pair, the project investigate whether selecting algorithm based on user can enhance recommendation quality and user satisfaction. RecPack library was used for calculating per-user NDCG values.

## Algorithms
- **Item Similarity Algorithms**: SLIM, ItemKNN, NMFItemToItem, and SVDItemToItem.
- **Hybrid Similarity Algorithms**: KUNN.
-	**Factorization Algorithms**: NMF and SVD.

## Datsets
- **MovieLens Data**: MovieLens100K, MovieLens1M, MovieLens10M datasets including movie ratings.
- **Globo Dataset**: large dataset collected from a news portal, including user interaction logs, specifically tracking page views.
- **CiteULike Dataset**: This dataset includes user-created collections of articles.

## Pipelines 

### Overall NDCG
'overall_ndcg.py' python script is used to compute the overall NDCG of the selected algorithms across the datasets. Using the recpack.datasets module, datasets can be imported to achieve overall NDCG of different algorithms. Globo dataset must be downloaded seperately, and should provide the path to the downloaded zip file.

### Per-User Ndcg

