<p align="center"><img width="561" height="701" alt="CodeMan" src="https://github.com/user-attachments/assets/8ba6a5c4-5030-4d9c-b40d-1b29bd4c1042" /></p>


# CMPIF2120: Cliff Notes

These are my personal auto-generated cliff notes, produced in part with my **CMPIF2120 Intelligent Transcriber**. I'm sharing them with the rest of the cohort. I'll do my best to keep this repo updated as they are mostly for my own use.

## Cliff Notes

The notes are organized by module:

- `Cliff Jumper Notes/` (each module's notes woven into one connected lesson)
- `Cliff Notes Module 01/` (the applied predictive modeling roadmap: supervised vs unsupervised learning, classification vs regression, the kNN method for classification and for regression, train/test splits and MAE/RMSE, plus the Chapter 2 reading in three parts: data mining tasks, the CRISP-DM process, and the neighboring analytics techniques)
- `Cliff Notes Module 02/` (data preprocessing and what it fixes: the Pitt Smart Living bus data, attribute and data set types, the text pipeline, missing data and outliers. Summary statistics and histograms: mean, median, mode, quartiles, variance, equal-width and equal-height buckets. Numerical transforms: standardization, min-max, and robust scaling. Categorical encoding: label, ordinal, and one-hot, with the Titanic notebooks. Then the five-video PCA arc: the linear algebra primer, vectors, dot products, and projection. Covariance matrices, the data ellipse, and eigenvectors. What PCA is and why it pays. The three-step recipe in scikit-learn. Choosing components with the scree plot, reading loadings, and Simpson's paradox)
- `Cliff Notes Module 03/` (clustering, the unsupervised side of the course: what a cluster is and why members should look more like each other than like anyone else. Hierarchical clustering, agglomerative and divisive, and the three questions it has to answer: how to represent a cluster, how to measure nearness, when to stop. Then k-means step by step: pick k, seed the centroids, assign every point to its nearest center, move each center to the mean, repeat until nothing changes. The three ways to seed the centroids, expired, wired, and upgraded (k-means++). Three worked runs on one ten point set, two seeds for k = 3 and a k = 2 run that welds three natural groups into two. Then judging the result: the silhouette index worked by hand, within cluster sum of squares, and the elbow method for picking k)

## Programs



Bare metal calculators, one per method, that run in a terminal and print every step of the math. No packages, just Python. Run one with no flags and it prompts for the inputs, or pass flags, or edit the data list at the top of the file. They are for checking homework math one step at a time, not for real data. (For real data there is scikit-learn, and it does not show its work.)

- `Programs/Module 01/` (kNN for regression and kNN for classification: distances, the k nearest, the average or the vote, then MAE and RMSE, or accuracy and a confusion matrix, on masked rows)
- `Programs/Module 02/` (PCA from a data set or straight from a covariance matrix: scaling, the covariance matrix, eigenvectors and eigenvalues, the scree table with the elbow and a variance target, loadings, and the projection. Plus the vector helpers from the linear algebra primer)
- `Programs/Module 03/` (k-means with every initial centroid rule the lecture named, WCSS, silhouette, and the elbow sweep. Hierarchical clustering bottom up with a dendrogram and top down by splitting the widest cluster)

The Module 01 and Module 03 files share helpers with their folder mates, so keep each folder together.

## Contributors

- **DatJavaClass (Victor S)**, author and director. Conceived these notes, established their format and structure, directed their creation, and fact-checked, edited, and quality-controlled every one, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2120 Faculty and Assistants.

As Always! You're awesome, Stay awesome! and I wish everyone the best of grades!

Can you find the passive aggressive parentheses?

## License

(c) 2026 Victor S (DatJavaClass). Everything in this repo is released under the [MIT License](LICENSE). Use it, share it, fork it, remix it. Keep the copyright notice, and keep the credit: these notes were created by **Victor S (DatJavaClass)** for **CMPINF-2120** at the **University of Pittsburgh**.
