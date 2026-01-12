# Flood-Cascading-Effects-Resilience
Code for cascading disaster network construction based on complex networks and rainstorm–flood resilience assessment using K-means clustering
# Flood Cascade Resilience Analysis

This repository contains the Python code used in the manuscript on cascading
rainstorm–flood resilience assessment in mountainous valley cities.

## Code Description

All scripts correspond directly to the figures, tables, and appendices in the manuscript.

- **Calculation_of_confidence_intervals.py**  
  Calculates confidence intervals for the Silhouette Coefficient (Table 5).

- **Calculation_of_correlations_among_network_indicators.py**  
  Analyzes correlations among four complex network centrality indicators and
  evaluates their interdependence and robustness (Appendix B).

- **Comparison_of_kernel_density_bandwidths.py**  
  Performs robustness testing of kernel density bandwidth parameters
  (Appendix A).

- **Complex_network_vulnerability_calculation.py**  
  Calculates complex network vulnerability metrics (Figure 8).

- **Construction_of_complex_networks.py**  
  Constructs cascading disaster networks based on historical rainstorm–flood events
  (Figure 4).

- **Rainfall_Resilience_Classification_tSNE_KMeans.py**  
  Classifies rainstorm–flood resilience patterns using t-SNE and K-means clustering
  (Figure 10).

- **Stability_analysis_of_clustering.py**  
  Evaluates the stability and uncertainty of clustering results
  (Appendix C).

## Notes on Data Availability

Due to data availability and policy restrictions, the raw disaster event records
used in this study are not publicly shared. However, all algorithms, analytical
procedures, and evaluation workflows are fully provided to ensure transparency
and reproducibility of the proposed methods.
