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
| 3 | High (~95%) | Three notes through DocTrio (writer, editor, fact-checker, each head a fresh agent), then the lesson through a full Diamond. Every number run under scikit-learn 1.9.0 and numpy 2.5.3: the eight address and ten address k-means runs traced by hand and matched to KMeans (n_init=10, random_state=0), every quoted distance and centroid, the six address silhouette by hand matched to silhouette_samples to the digit (0.610, then 0.697 with the misfit moved), the WCSS hand check (3.75 + 5.5 = 9.25) against inertia_, the k = 1 to 10 elbow table with its drops, and both code blocks run verbatim; the execution auditor went further and enumerated every partition of the ten addresses to confirm each converged answer is the global optimum at every k. The fidelity auditor traced about 190 lecture claims to the transcripts and caught the one outright error, a closing sentence that hardened the professor's "typically decrease" into "only ever falls"; the execution auditor caught the same sentence from the other side (k-means inertia can rise with k on a bad seed); the meta-auditor confirmed it, overturned seven auditor findings (six "unlabeled" bridges that were editorial voice the Module 2 notes also leave unmarked, and a rounding change that would have desynced the Jumper from its cliff note), and caught what both missed: a scope creep on the numerical data claim and one placeholder phrase ("the 9.25 style number") that a non mathematical reader could not follow. Seven minimal edits applied and re-verified, both code blocks re-run |
| 4 | High (~95%) | Three notes through DocTrio (writer, editor, fact checker, each head a fresh agent), then the lesson through a full Diamond. Every number was precomputed from one ten receipt basket before any writer started and run three ways, by the Module 04 calculators, by an independent brute force enumeration, and by each auditor's own recount: every support count and receipt list, the eleven frequent sets at a minimum support of 3, the six rule confidences from the frequent triple and the ten pair rules at 70%, the five strong rules in order, the a priori walk with its four joins and three vetoes, the n choose k figures, and the pair storage sizes all matched. The fidelity auditor traced every claim to the transcripts and the three notes and caught the two outright errors, a loyalty card sentence that turned specials once available without a card into specials that used to be free, and an added note that called the ignoring of item order a factor of two when it is not; the execution auditor found nothing to refute and flagged a storage comparison whose 36, 15 and 10 read as descending while the triples method costs 30 numbers on this basket; the meta auditor confirmed both errors, overturned four findings (a threshold read as the lecture's when the sentence scopes it to the notes' own basket, a paraphrase of the lecture's containment rule, an unmarked editorial gloss, and a table format that matches its cliff note), caught what both missed (a preamble that under described the notes' own marks, plus an AI tell sweep and a sibling format check neither auditor had run, both clean), and returned nine minimal edits, all applied and re-verified |

## A note of caution

This is still a study aid, not a textbook. Use it to understand and review, but
check anything important against the actual course material, especially before
a quiz.

## Contributors

- **DatJavaClass (Victor S)**, author and director. Conceived these notes, established their format and structure, directed their creation, and fact-checked, edited, and quality-controlled every one, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2120 Faculty and Assistants.
