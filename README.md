<p align="center"><img width="561" height="701" alt="CodeMan" src="https://github.com/user-attachments/assets/8ba6a5c4-5030-4d9c-b40d-1b29bd4c1042" /></p>


# CMPIF2120: Cliff Notes

These are my personal auto-generated cliff notes, produced in part with my **CMPIF2120 Intelligent Transcriber**. I'm sharing them with the rest of the cohort. I'll do my best to keep this repo updated as they are mostly for my own use.

## Cliff Notes

The notes are organized by module:

- `Cliff Jumper Notes/` (each module's notes woven into one connected lesson)
- `Cliff Notes Module 01/` (the course roadmap: supervised vs unsupervised learning, classification vs regression, kNN for both, train/test splits with MAE and RMSE, plus the Chapter 2 reading on data mining tasks and CRISP-DM)
- `Cliff Notes Module 02/` (data preprocessing: data and attribute types, missing data and outliers, summary statistics and histograms, scaling and encoding, then the PCA arc from vectors and covariance to components, scree plots and loadings)
- `Cliff Notes Module 03/` (clustering: hierarchical, agglomerative and divisive, k-means step by step with three ways to seed the centroids, then judging the result with the silhouette index, within cluster sum of squares and the elbow method)
- `Cliff Notes Module 04/` (association rule mining: market basket data, support and frequent itemsets, the subset property, rules scored by support and confidence, and the a priori algorithm with its join and prune steps)

## Programs



Bare metal calculators, one per method, that run in a terminal and print every step of the math. No packages, just Python. Run one with no flags and it prompts for the inputs, or pass flags, or edit the data list at the top of the file. They are for checking homework math one step at a time, not for real data. (For real data there is scikit-learn, and it does not show its work.)

- `Programs/Module 01/` (kNN regression and classification: distances, the k nearest, the average or the vote, then MAE and RMSE or accuracy and a confusion matrix on masked rows)
- `Programs/Module 02/` (PCA from a data set or a covariance matrix: scaling, covariance, eigenvectors and eigenvalues, the scree table, loadings and projection, plus the vector helpers)
- `Programs/Module 03/` (k-means with every seeding rule, WCSS, silhouette and the elbow sweep. DBSCAN with core, border and noise points and the k-distance table. Hierarchical clustering bottom up and top down)
- `Programs/Module 04/` (itemset support and the subset property, association rules with support and confidence for every split, and the a priori join, prune, count loop with the candidate check and the n choose k count)

Every calculator also has a `_GenUse` twin in the same folder. Same step by step printout, no data list to edit: point it at a csv with `-d` (or `-` for stdin, or rows typed inline), name the columns you want, and it runs on whatever you give it, any number of rows or features. A few knobs rode along, weighted kNN, leave one out, k-means restarts, ward linkage, a height cut on the dendrogram, and a `--brief` switch for when the data is too big to watch every step. Still shows its work.

The Module 01, Module 03 and Module 04 files share helpers with their folder mates, so keep each folder together.

## PyPrime Environment

The class Python setup in a box. Download one file, let Docker Desktop do the rest, and JupyterLab opens in your browser with pandas, scikit learn, XGBoost and friends already installed, the same versions for everyone.

- `PyPrime Environment/` (the start file for Windows and Mac, the Docker recipe, and a README that walks through setup as if you have never heard of Docker)

## Contributors

- **DatJavaClass (Victor S)**, author and director. Conceived these notes, established their format and structure, directed their creation, and fact-checked, edited, and quality-controlled every one, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2120 Faculty and Assistants.

As Always! You're awesome, Stay awesome! and I wish everyone the best of grades!

Can you find the passive aggressive parentheses?

## License

(c) 2026 Victor S (DatJavaClass). Everything in this repo is released under the [MIT License](LICENSE). Use it, share it, fork it, remix it. Keep the copyright notice, and keep the credit: these notes were created by **Victor S (DatJavaClass)** for **CMPINF-2120** at the **University of Pittsburgh**.
