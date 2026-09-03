# Cliff Jumper Notes

## What these are

A Cliff Jumper Note takes all the separate Cliff Notes from one module and
weaves them into a single, connected lesson. Instead of six short notes that
repeat the same ideas, you get one walkthrough that introduces each idea once
and builds on it. Where it helps, a small amount of extra explanation or
example is added, but only when it is correct and it is labeled as added.

One thing specific to this course: the notes define every term of art where
it appears, since the lectures tend to use terms without stopping to define
them. Definitions that go beyond what the lecture or the reading said are
marked "(added definition)".

## How they are checked: "Who Watches the Watchmen"

These notes were drafted with AI help, so every one was put through a checking
system to catch mistakes before it went in here. It works in a diamond shape:

1. The note is written.
2. Two separate checkers read it independently and hunt for errors. One checks
   that every claim matches the source Cliff Notes, the lecture transcripts,
   and the textbook pages. The other checks the technical side and actually
   runs every formula and every number to confirm it is right.
3. A third checker reviews the first two. This is the "who watches the
   watchmen" step. It catches anything the first two missed and throws out any
   complaint they got wrong.

Anything the checkers flagged was fixed, and the fix was checked again. All
formulas and all numbers in these notes were run, not guessed.

## Confidence of Accuracy

| Module | Confidence of Accuracy | Basis |
|---|---|---|
| 1 | High (~95%) | Every number and formula run (the kNN 415 average, the k = 1 to 5 majority-vote table, Euclidean distance against numpy and scipy in 2 to 50 dimensions, MAE and RMSE against scikit-learn, the 22.5% vs 15% churn ratio); every figure value, citation year, and chapter reference traced to the textbook pages; the execution auditor caught that the lecture's "k not a multiple of the class count" rule only guarantees no ties for two classes (kept as stated for quiz purposes, caveat added); the fidelity auditor caught an invented bridge between the book's profiling-based anomaly detection and the lecture's Challenger example, a dropped hedge on how Victor's own CRISP-DM loop numbering was read, and an overbroad "all covered later" claim; the meta-auditor confirmed a wrong "five headings" count against the book's "six groups" and caught the shared blind spot both auditors inherited from the source note, that the book itself (not the notes) names the explanatory versus predictive modeling debate; six minimal edits applied and re-verified |
| 2 | High (~95%) | Nine notes, each through its own Diamond, then the lesson through a Double Diamond. Every number and code cell run under pandas 3.0.5 and scikit-learn 1.9.0: the weighted-mean, median, quartile, and histogram walks, the StandardScaler, MinMaxScaler, and RobustScaler scores on one salary column, the Titanic and education notebooks, the lemonade-stand covariance matrix with its eigenvalues checked against a 361-angle variance sweep, the penguin PCA reproduced from the Palmer data (342 rows, 68.8 / 19.3 / 9.1 / 2.7 percent, loading signs), and the wine and three-department examples with fixed seeds. The fidelity auditors caught a dropped three-way-tie mode case (a quiz hazard), eight unmarked bridges and glosses, notebook text credited to the professor, and two terms used before they were defined; the execution auditors caught a backwards sample-versus-population explanation and a forward reference to the wrong section; the meta-auditors overturned false alarms on figure descriptions that were the professor's own words and on a supposedly lifted example the professor never worked. All fixes re-run and re-checked |

## A note of caution

This is still a study aid, not a textbook. Use it to understand and review, but
check anything important against the actual course material, especially before
a quiz.

## Contributors

- **DatJavaClass (Victor S)**, author and director. Conceived these notes, established their format and structure, directed their creation, and fact-checked, edited, and quality-controlled every one, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2120 Faculty and Assistants.
