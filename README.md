# Ocean Cluster Validation
A methodological approach to validate clusterings and ensure that they represent the actual data strucutres at the example of ocean physics and biogeochemistry.


## Scripts
umap_experiments.ipynb  -  Try different hyperparameters on UMAP and decide for a combination.
clustering_experiments.ipynb  -  Compute scores for different clusterings and hyperparameters.
clustering_post_processing.ipynb  - Dealing with many small clusters (remove small clusters or perform hierarchical clustering to merge small clusters)
score_visualization.ipynb  -  Visualize the previously computed scores
clustering_visualization.ipynb  -  Visualize the previsouly computed clusterings in TS-, UMAP- and geographic space
uncertainy_experiments.ipynb  -  Rerun UMAP-DBSCAN combination several times to determine uncertainties of the methodd