# Ocean Cluster Validation
A methodological approach to validate clusterings and ensure that they represent the actual data strucutres at the example of ocean physics and biogeochemistry.


## Scripts
data.ipynb - Data description.
umap_experiments.ipynb  -  Try different hyperparameters on UMAP and decide for a combination.
clustering_experiments.ipynb  -  Compute scores for different clusterings and hyperparameters.
score_visualization.ipynb  -  Visualize the previously computed scores.
model_training.ipynb - Training final clustering models based on hyperparameters determined in clustering_experiments.ipynb.
clustering_post_processing.ipynb  - Dealing with many small clusters (remove small clusters or perform hierarchical clustering to merge small clusters).
uncertainy_experiments.ipynb  -  Rerun UMAP-DBSCAN combinations several times to determine uncertainties of the method.
region_analysis.ipynb - Visualise clustering and 3 specific areas (Deep Atlantic, Mediterranean, Labardor Sea).
emus.ipynb - Comparison to Ecological Marine Units (EMUs) by Sayre et al. ().
longhurst.ipynb - Comparison to provinces by Longhurst ().
water_masses.ipynb - Comparison to various water mass defintions ().
