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
| 2 | High (~95%) | Every number and code cell run. The equal-width 13/25/14 and equal-height 13/10/15/14 bucket walks against the 52-item price table, the median and three-dataset examples, sqrt(n/(n-1)) = 1.118 and 1.054 behind the professor's "not exactly 1" standard deviation, the StandardScaler, MinMaxScaler, and RobustScaler properties on a 10-row age/income set, and both Titanic and education notebooks re-executed under pandas 3.0.5 and scikit-learn 1.9.0 (neither ships saved outputs). Every quote, name, and figure traced to the four transcripts. The execution auditor caught that notebook 4.2 also imports LabelEncoder unused. The fidelity auditor caught the "Label Encoding (for ordinal data)" heading sitting over education_order rather than education_map (inherited from the cliff note, fixed in both) and an invented bridge claiming encoding leans on the summary statistics. The meta-auditor overturned a false alarm on the Port Authority's human-error explanation (the lecturer drops his own hedge two sentences later), ruled the skew-ordering and sklearn quartile-interpolation caveats unfit for a quiz-keyed document, and caught a lecture-defined term (outlier) mislabeled as added. Four minimal edits applied and re-verified |

## A note of caution

This is still a study aid, not a textbook. Use it to understand and review, but
check anything important against the actual course material, especially before
a quiz.

## Contributors

- **DatJavaClass (Victor S)**, author and director. Conceived these notes, established their format and structure, directed their creation, and fact-checked, edited, and quality-controlled every one, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2120 Faculty and Assistants.
