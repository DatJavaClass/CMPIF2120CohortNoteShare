# Module 1: Cliff Jumper Notes

*One continuous lesson stitched from the Module 1 cliff notes of Applied
Predictive Modeling (CMPINF-2120): what data looks like for each kind of
task, the lecture's tour of techniques, the book's wider map of data
mining tasks, the CRISP-DM process that organizes all of it, kNN as the
first algorithm you can run by hand, how a model gets scored, and where
predictive modeling sits among its neighboring disciplines.*

---

## 1. The roadmap: everything starts with data

The opening lecture maps the course: task types, the data each needs,
example techniques, and how to choose among them. The idea underneath is
that **everything starts with data**, and the shape of the data decides
which kind of task you are doing.

Picture a table with one row per record. The columns recorded for each
record are the **predictor measurements** (also called the "initial
data"): for a house, square feet, year built, number of bedrooms. If the
table carries one extra column holding the thing you want to predict,
that column is the **response measurement**, and it comes in two flavors:
a **label** (a category) or a **value** (a number).

Whether that response column exists is the whole distinction between the
two families of learning:

- **Unsupervised learning**: predictor measurements only, no response
  column. Data + algorithm (+ optional parameters) yields interesting
  observations or discovered patterns.
- **Supervised learning**: predictors plus response measurements. Data +
  labels or values + algorithm (+ optional parameters) yields a
  **model**; then new data + model yields predicted labels or values.

Two words in those flows get hand-waved, so the notes add definitions.
**Parameters** (added) are settings supplied to an algorithm that shape
how it runs, such as the number of groups to find in clustering. A
**model** (added) is the learned mapping from predictors to response,
reused on future data. Hold onto both: "parameters" returns as the k in
kNN, and "model" is what every later section builds, scores, or deploys.

---

## 2. Unsupervised examples: patterns with no answer key

With no response column, an algorithm can only report structure it finds
in the predictors. The lecture gives three examples.

**Clustering** is discovering groups of records that are in some way
similar: members of a cluster should be more alike each other than
members of other clusters. The lecture calls it the simplest,
easiest-to-understand unsupervised technique. Its example is points on a
2D plane split into three groups (yellow, blue, pink), with the caveat
that asking for a different number of groups (a parameter) can give
different clusters.

**Anomaly detection** is identifying interesting records that either need
further investigation because they look like **outliers** (added: a
record that lies far from the bulk of the data) or may indicate data
errors. It is unsupervised and usually paired with visualization. The
lecture's example is the 1986 Challenger space shuttle accident. The
investigation concluded that o-rings in part of the fuel supply failed at
launch, destroying the shuttle. Plot launch temperature against o-ring
damage and the pattern is roughly "lower temperature, higher damage." No
prior launch had been below 50 degrees, and even at 50 to 52 the damage
index was very high. Challenger launched below 30 degrees Fahrenheit, as
stated in the lecture, a clear outlier relative to all prior data. (The
notes add that historically the launch was about 36 F and the o-rings
sealed the solid rocket booster joints.) The lecture's own caveat is the
lesson: this outlier was found after the fact, during the investigation,
and such outliers are much harder to spot among millions of data points.

**Association rule learning** is finding relationships between variables,
such as which products customers frequently buy together: people who buy
beer often buy chips too. The goal is novel, interesting patterns a store
can use for marketing and promotions. Applied to shopping data this is
**market basket analysis**, covered in detail later in the course.

---

## 3. Supervised examples: the housing table and the technique tour

Now add the response column back. The lecture's running example is a
**housing table**: one row per house, with square feet, year built, and
number of bedrooms as predictors. What sits in the extra column decides
the task:

- **Classification**: the response column holds labels, here yes/no for
  "does this house have a pool?" Given a new house (square feet, year,
  bedrooms), predict the unknown label.
- **Regression**: the response column holds values, here the price the
  house sold for. Given a new row, predict the unknown value. The
  lecture's worked new row: 1700 sq ft, built in 2000, 3 bedrooms.

Labels give classification; values give regression. That sentence is the
hinge of the module, the same hinge the book uses in Section 4 and that
kNN splits on in Section 6.

The lecture then tours four supervised techniques, most of them
covered properly later in the course:

- **Recommender systems.** The data is user ratings of items (movies,
  books, products); the system predicts an item's rating for a user who
  presumably has not rated it yet. **Collaborative filtering** predicts a
  user's rating from other users' ratings and that user's similarity to
  everybody else.
- **Decision trees** (added definition: a sequence of rule tests on
  predictors leading to a predicted label or value). Usable for both
  classification and regression, and very intuitive to implement. The
  lecture's rule for the housing data: if the house has over 2000 square
  feet, it has a pool; otherwise it does not.
- **Regression as a technique.** Find a function that models the data
  with the least amount of error. **Linear regression** is the simpler
  form (added: it fits a straight-line function of the predictors).
- **Naive Bayes.** A probabilistic classification algorithm: very simple,
  fast, and scalable. A **probabilistic classifier** (added) assigns the
  class with the highest computed probability given the predictors.
  Applications: text classification (spam filtering, sentiment analysis),
  document categorization, medical analysis.

---

## 4. The book's wider map: nine tasks, and mining versus using

The lecture's tour is a sampler. Provost and Fawcett's Chapter 2 supplies
the catalog and the principle behind it: **data mining** (added: the
automated extraction of patterns and models from data, treated as a
staged process rather than one algorithm run) is a **process** with
well-understood stages. Some are IT work (automated pattern discovery and
evaluation); others need an analyst's creativity, business knowledge, and
common sense. Knowing the process makes projects systematic instead of
"heroic endeavors driven by chance and individual acumen."

The book's unit is the **individual**: an entity we have data about,
usually a customer, but possibly inanimate such as a business. Every
business problem is unique, but common tasks underlie them, so data
scientists and stakeholders **decompose** the problem into subtasks and
**recompose** the solutions. The running example is MegaTelCo **churn**
(added: customers leaving for a competitor or cancelling service); one
subtask appears in any churn problem: estimate from historical data the
probability that a customer terminates her contract shortly after it
expires. Matching pieces to known tasks avoids reinventing the wheel and
focuses human creativity on what is not automated.

Despite countless algorithms, only a handful of task types exist. The
book lists nine:

1. **Classification and class probability estimation.** Predict which of
   a small set of (usually mutually exclusive) **classes** an individual
   belongs to: which customers will respond to an offer? Its close
   relative **scoring** outputs a **score** (a probability or other
   likelihood) per class instead of a hard label; a model that does one
   can usually be modified to do the other.
2. **Regression** ("value estimation"). Estimate a numerical value per
   individual: how much will a customer use the service? Classification
   predicts *whether*; regression predicts *how much*.
3. **Similarity matching.** Identify individuals similar to a given one.
   IBM finds companies like its best customers using **firmographic
   data** (added: industry, size, location, revenue). It underlies a
   popular recommendation method and solutions to classification,
   regression, and clustering.
4. **Clustering.** Group individuals by similarity, not driven by any
   specific purpose: do our customers form natural segments?
5. **Co-occurrence grouping.** Also called frequent itemset mining,
   association rule discovery, and market-basket analysis (the lecture's
   beer and chips). Find associations from the transactions entities
   appear in together: ground meat with hot sauce far more often than
   expected. Clustering uses similarity of attributes; co-occurrence uses
   appearing together in transactions. An **association rule** (added) is
   a pattern "if X is in the transaction, Y tends to be too," with
   frequency and surprise statistics.
6. **Profiling** (behavior description). Characterize the typical
   behavior of an individual, group, or population. Profiling often sets
   the norms for **anomaly detection** (added: flagging items whose
   behavior deviates sharply from an established profile): a new credit
   card charge is checked against the profile, the mismatch becomes a
   suspicion score, and an alarm fires if it is too high. Compare the
   lecture's Challenger example, which found its anomaly visually
   rather than against a profile.
7. **Link prediction.** Predict connections between data items, possibly
   with strength: "you and Karen share 10 friends, maybe you'd like to be
   Karen's friend?" Movie recommendations are customer-to-movie links
   that do not exist yet but are predicted to be strong.
8. **Data reduction.** Replace a large dataset with a smaller one that
   keeps much of the important information, e.g. viewing data reduced to
   **latent** (added: not directly observed but inferable) taste
   preferences. It loses information; the trade-off for insight matters.
9. **Causal modeling.** Understand which actions actually influence
   others: did the ads cause purchases, or did the model pick people who
   would have bought anyway? Techniques are **randomized controlled
   experiments** ("A/B tests"; added: random assignment to treatment or
   control so outcome differences can be attributed to the treatment) and
   methods for causal conclusions from **observational data** (added:
   collected without the analyst controlling who got the treatment). Both
   are **counterfactual** analysis (added: comparing against the outcome
   under the alternative that did not happen). Always state the
   assumptions a causal conclusion needs; the placebo effect is the
   notorious overlooked one.

### Which tasks are supervised

The book traces the vocabulary to machine learning: a teacher
"supervises" the learner by supplying **targets** with the examples;
unsupervised learning uses the same examples with no targets. The
**target variable** is the specific quantity the model is built to
predict, and the target value recorded for one individual is its
**label**, a word reflecting that one often has to spend money to label
data.

Two similar-sounding questions show the line. "Do our customers naturally
fall into different groups?" has no target: unsupervised. "Can we find
groups of customers with particularly high likelihoods of canceling soon
after their contracts expire?" defines a target and the **segmentation**
(added: a segment is a subgroup sharing some characteristic or behavior)
exists to act on it: supervised. If a specific target can be provided,
phrase the problem as supervised, because supervised results are often
much more useful; clustering groups by similarity with no guarantee the
groups mean anything for any purpose. Supervision has a second condition:
there must be **data** on the target. If records are retained only two
months, a six-month retention label cannot be produced. Acquiring target
data is often a key investment.

The mapping, per the book:

- Generally supervised: classification, regression, causal modeling.
- Could be either: similarity matching, link prediction, data reduction.
- Generally unsupervised: clustering, co-occurrence grouping, profiling.

Within supervised work, target type decides the task, exactly as the
housing table did: a **numeric variable** (added: a measurable number)
gives regression; a **categorical variable** (added: one of a fixed set
of discrete values) gives classification, often with a **binary target**
(added: exactly two values). The book's three questions: "Will this
customer purchase S1 if given incentive I?" is binary classification;
"Which package (S1, S2, or none)?" is three-valued classification; "How
much will this customer use the service?" is regression. The subtlety:
business often wants a numerical prediction over a categorical target,
such as the probability a customer continues. That is still
classification, called **class probability estimation** where clarity
matters. The two vital early decisions are therefore (i) supervised or
unsupervised, and (ii) if supervised, a precise definition of the target
variable for which values can be obtained for some example data.

### Mining the data versus using the results

The book's second distinction separates **mining** the data to build
models from **using** the results. Students and managers confuse them;
intended use should inform the mining, but they stay distinct. **Figure
2-1** draws it: on top, a historical data table with columns x, y, z,
class (rows 14 / True / Red / accepted; 6 / True / Blue / rejected;
50.3 / False / Red / accepted), captioned "Training data have all values
specified," passed through a data mining step to produce a model drawn
as a decision tree, which is deployed. On the bottom, a new item,
30 / false / Red / ?, is fed to the deployed model and comes back
"Class: accepted, Probability: 0.88." That is Section 1's supervised flow
in the book's hand. The notes add the vocabulary it relies on: **training
data** are historical examples with targets specified; **deployment** is
putting the finished model into operation on new cases; **attrition** is
a synonym for churn; a **decision tree** predicts by walking attribute
tests from a root node down to a leaf holding the prediction.

---

## 5. CRISP-DM: the process that organizes all of it

Data mining is a craft, heavy on science and technology but still
involving art, and like other mature crafts it has a process that buys
consistency, repeatability, and objectivity. That process is
**CRISP-DM**, the Cross Industry Standard Process for Data Mining (the
book cites Shearer, 2000). Victor's notes add the history the book does
not: it dates from 1999 and was tool- and industry-agnostic by design,
which is why it survived into an era it was never built for. His
one-line summary: do not jump straight into building a model before you
understand the problem.

Figure 2-2 draws a clockwise outer loop around six boxes with a "Data"
cylinder at the center. Arrows: Business Understanding and Data
Understanding two-way; Data Understanding to Data Preparation; Data
Preparation and Modeling two-way; Modeling to Evaluation; Evaluation to
Deployment; a "shortcut" from Evaluation back to Business Understanding;
and the outer ring from Deployment back to Business Understanding.
Iteration is the rule: the first pass is often data exploration and the
second far better informed. Victor's notes name the two most common loops
as 1<->2 and 3<->4; read under the standard 1-6 numbering (his "phases 1
and 6" remark implies it), that is Business/Data Understanding and Data
Preparation/Modeling, exactly the two two-way arrows.

**Business Understanding.** Understand the problem to be solved. Business
projects seldom arrive as clean data mining problems; recasting them is
iterative ("cycles within a cycle") and the first formulation is rarely
optimal. This is where analyst creativity matters most: the key to great
success is often a creative problem formulation that casts the business
problem as one or more of Section 4's standard tasks. Think about the
problem and the use scenario together, and expect to loop back. Framing
in terms of **expected value** (added: the probability-weighted average
of a decision's outcomes) lets you decompose systematically. Victor's
example: "predict which customers are likely to cancel so we can
intervene."

**Data Understanding.** Data are the raw material; know their strengths
and limits. Historical data were usually collected for unrelated purposes
or none, and data are an asset with an acquisition cost, so weigh cost
against benefit per source; even acquired data need collating. The book's
fraud example shows how this stage can redirect a project. Credit card
fraud appears on the statement and fraudster and customer are different
people with opposite goals, so labels are reliable: supervised. Medicare
fraud perpetrators are also legitimate users of the billing system and
nobody disinterested declares the correct charges, so there is no
reliable target: use the unsupervised tools (profiling, clustering,
anomaly detection, co-occurrence grouping). "Both are fraud detection"
is a superficial, misleading similarity. Victor's summary: determine what
the data contain, whether they are relevant, and what problems they have.

**Data Preparation.** Tools impose requirements on data form, so
conversion is usually needed, often alongside data understanding: convert
to tabular format, remove or infer missing values, convert types, and
apply **normalization / scaling** (added: rescaling numeric variables to
a common range so no variable dominates by its units). Much early time
goes to defining the variables used later, a main entry point for
creativity and business knowledge; Victor's "number of support calls in
the last 90 days" is exactly that. This is also where **leakage** lives.
A **leak** is a variable in the historical data carrying information
about the target that would not actually be available at decision time:
"total pages visited in the session" predicts whether a visitor will
leave but is known only after the session ends; tax paid predicts "big
spender" but is unknown at decision time. Leaks are easy to build in
unnoticed precisely because prep is done after the fact from historical
data. (The notes add that the 90-day support-call variable is known at
decision time, so it is not a leak.)

**Modeling.** The output is a model or pattern capturing regularities in
the data; this is where the most science and technology can be brought
to bear. A **model** here (added) is a simplified representation of
reality, a formula or rule structure that maps inputs to a prediction.
Victor's notes add a practice the book does not state in this section:
try multiple approaches rather than assume one algorithm will win, e.g.
**logistic regression** (added: a linear classification model outputting
class probabilities), a **decision tree**, and a **random forest** (added:
an ensemble of many trees on random subsets of data and attributes,
combined by voting or averaging).

**Evaluation.** Look hard enough at any dataset and you will find
patterns; the aim is confidence they are true regularities, not sample
anomalies. Two settings matter. In the **lab**, testing is far easier,
cheaper, quicker, and safer than deploying straight from mining. But
evaluation must also check the original business goals. A model can pass
strict lab tests and still be impractical: detection systems (fraud,
spam, intrusion) can exceed 99% **accuracy** (added: fraction of
predictions correct) and still be economically infeasible because of
**false alarms** (added: flagging an instance positive when it is
negative). Stakeholders also want comprehensibility and assurance against
catastrophic mistakes. Since performance data on a deployed model is hard
to get, teams build testbeds mirroring production, and can extend
evaluation into the live system with randomized experiments: apply the
model to some customers and hold others as a **control group** (added:
individuals deliberately not given the treatment, so the treated group
has a baseline). Deployed systems also need instrumenting because the
world changes under the model. Victor's example nails the business half:
"The model is 92% accurate, but does it identify customers early enough
that the company can actually do something?"

**Deployment.** Results are put into real use to realize a return on
investment: integrate the churn model with churn management, or build the
fraud model into a system that opens cases for analysts. Sometimes the
mining *system* is deployed rather than a fixed model (online ad targeting
builds models per campaign), because the world changes faster than the
team can adapt or there are too many modeling tasks to hand-curate.
Deployment can be low-tech: printer-diagnosis rules were deployed by
taping a sheet of paper to the side of the printers. Victor's notes make
the identical point: integrating a model into software, producing a
report or dashboard, or changing a business process all count. Deploying
usually means re-coding for production, so the prototype goes "over the
wall" to developers, and that handoff is risky. Maxim: "Your model is not
what the data scientists design, it's what the engineers build." The fix
is to bring developers in early, increasingly as **data science
engineers**: software engineers with expertise in both production systems
and data science, who eventually own the product.

**Closing the loop.** Successful or not, deployment usually sends the
process back to Business Understanding; a second iteration can produce a
better solution or even new lines of business. You do not have to fail
at deployment to restart: evaluation alone can show results are not good
enough, which is the shortcut arrow, and in practice every stage should
have shortcuts back to every prior one. Victor's notes only: the two
phases people shortchange are Business Understanding and Deployment, and
those are the two that determine whether the project delivered anything.
The book does not say this in one sentence, but it lines up with its
emphasis on problem formulation and on deployment as where return is
realized.

One management consequence follows. The CRISP diagram looks deceptively
like a software development cycle, but data mining is exploratory,
closer to R&D, iterating on approaches and strategy rather than software
designs. Buy information first (pilot studies, throwaway prototypes,
literature review, experimental testbeds) rather than engineering
straight for deployment, and hire for problem formulation, fast
prototyping, sound assumptions, and experiment design rather than lines
of code.

---

## 6. kNN: the first algorithm you can run by hand

Modeling has been a box in a diagram so far. **k-nearest neighbors
(kNN)** makes it concrete: it labels a new point by looking at the labels
of the k points closest to it in the data space. It is supervised in
Section 1's sense, since the training points carry the correct answer
(the **label**) and the new point arrives without one.

### Classification

kNN needs three things: a value of **k** (how many neighbors to consult;
this is Section 1's "parameter"), labeled points in a multi-dimensional
space (each **feature** or **dimension**, added, is one measured
attribute and one axis of the space), and a **distance function**, a rule
for measuring how far apart two points are. For every new point: find its
k closest labeled points using the distance function, then assign the
label held by the majority of them. That is the **majority rule**.

One rule governs k: it **must not be a multiple of the number of
classes**, because the votes can then split evenly and leave no majority.
With two classes, k = 2, 4, 6, ... are bad choices, and (the notes add)
odd k avoids ties in the two-class case. (Also added: with three or more
classes the rule is necessary but not sufficient; three classes and k = 4
can still split 2-2-0, so real implementations carry a tie-break. Quote
the lecture's rule as stated on a quiz.)

The lecture's example has blue and green labeled points and a new black
point. Inside the blue cluster the answer is blue; right next to the
green cluster, green. The hard case is a point between the clusters. Its
five closest neighbors, in order of increasing distance, are labeled
blue, green, green, blue, green, so the counts are running totals:

```
k=1: blue 1, green 0  -> blue
k=2: blue 1, green 1  -> tie (no decision)
k=3: blue 1, green 2  -> green
k=4: blue 2, green 2  -> tie (no decision)
k=5: blue 2, green 3  -> green
```

Both ties land on multiples of 2, the class count: the rule in action.

The typical distance function is **Euclidean distance**, the
straight-line distance computed from the coordinate differences. In two
dimensions, with item i at (x_i, y_i) and item k at (x_k, y_k):

```
d(i,k) = sqrt( (x_i - x_k)^2 + (y_i - y_k)^2 )
```

Difference on each axis, square each, add, take the square root. In the
lecture figure the black point is item k and the blue point under the
red arrow is item i. The lecture calls the extension to n dimensions
trivial without writing it out; the notes add it, one squared-difference
term per axis:

```
d(i,k) = sqrt( (x1_i - x1_k)^2 + (x2_i - x2_k)^2 + ... + (xn_i - xn_k)^2 )
```

### Regression

Swap labels for values and the same machinery does regression. The three
requirements are unchanged except that each point carries coordinates
**and an associated value**, and the prediction step changes from a vote
to the **average** of the neighbors' values. Because it averages instead
of voting, k has **no restriction** here; ties were the only reason for
the classification rule.

The lecture's example is housing data: each row is a house with square
feet, year built, and sale price, plotted with square feet on x and year
built on y, each point labeled with its price. The new point is 2000 sq
ft, built 1999, with k = 4. Its four nearest neighbors have prices 480,
400, 370, and 410:

```
(480 + 400 + 370 + 410) / 4 = 1660 / 4 = 415
```

The prediction is 415. The transcript does not state the price units.

### Strengths and weaknesses

kNN is intuitive, simple to implement, and makes no assumptions about the
distribution of the underlying data, which is what **non-parametric**
(added) means. Against that: it is computationally expensive on large
data sets, since one prediction means computing the distance to every
stored point; it is sensitive to irrelevant features and noisy data,
because every dimension enters the distance (the transcript says
"relevant features," read as irrelevant, since a feature carrying no
class information still shifts the distances); and it suffers the
**curse of dimensionality** (added: as dimensions grow, the volume of the
space grows so fast that a fixed amount of data becomes sparse and
distances become less informative). High-dimensional spaces are
inherently sparse. The transcript's "cares of dimensionality" is a
transcription of this term.

---

## 7. How we know a model is any good

The 415 above is a prediction. Whether it is a *good* prediction is the
Evaluation stage of Section 5, and the kNN regression lecture supplies
the first concrete technique, covered more fully later: the **train/test
split**. Randomly split the data into two subsets. The **training data**,
typically about 80%, trains the algorithm (for kNN, the pool neighbors
are drawn from). The **test data**, typically about 20%, is held out:
pretend each result is unknown (**masking**), predict it, then unmask and
compare the prediction to the **ground truth**, the real known value.

What "compare" means depends on the task, the label-versus-value split
one more time. For classification, the metrics are named now and defined
later in the course: count **false positives** and **false negatives**,
and compute **precision**, **recall**, and the **F1 score**. The notes
add working definitions: a false positive predicts positive when the
truth is negative, a false negative the reverse; precision is, of the
points predicted positive, the fraction that truly are; recall is, of the
truly positive points, the fraction the model caught; F1 is the harmonic
mean of the two. (This is the false-alarm cost that made the
99%-accurate fraud model infeasible in Section 5.)

For regression the metrics are more numerical. With p_i the masked ground
truth for measurement i, q_i the predicted value, and n the count:

```
MAE  = (1/n) * sum_i |p_i - q_i|
RMSE = sqrt( (1/n) * sum_i (p_i - q_i)^2 )
```

**Mean absolute error** takes the absolute difference per point, sums,
and divides by n. **Root mean squared error** squares each difference,
sums, divides by n, and takes the square root. In a sentence (added): MAE
averages raw error sizes; RMSE squares first, so large errors count more.

---

## 8. The neighboring disciplines, and how to choose a method

The book closes Chapter 2 by placing data mining among the technologies
beside it. The dividing line: data mining is the **automated** search for
knowledge, patterns, or regularities in data, and the others mostly are
not. The book says it presents "six groups"; five technique headings
follow (the ones below), plus a closing section that applies them to
business questions.

**Statistics** has two senses. **Summary statistics** are the catchall
for computing numeric values of interest (sums, averages, rates), often
conditionally on subsets, chosen with attention to the business problem
and the **distribution** of the data. The 2004 Census Bureau Economic
Survey put mean U.S. income over $60,000, but income is **skewed**
(added: values pile up on one side with a long tail on the other, pulling
the mean toward the tail), so the median, $44,389, is the honest figure.
**Statistics** the field, a component of data science, covers
distributions, testing hypotheses, and estimating uncertainty.
**Hypothesis testing** asks what the chance is that an observed
difference is random variation; the book's example is Northeast churn of
22.5% against 15% nationwide, 1.5x the rate. A **confidence interval**
quantifies uncertainty as a range: churn is 15%, and 95% of the time it
falls between 13% and 17%. The relationship to mining is the important
part: data mining is hypothesis *generation*; hypothesis *testing* should
follow, generally on different data. **Correlation** likewise has a
general sense (variation in one quantity tells you something about
another) and a technical one (a specific formula such as linear
correlation).

**Database querying.** A **query** is a specific request for a subset of
data, or statistics about it, in a technical language such as **SQL**
(added: Structured Query Language) or a GUI such as **QBE**
(query-by-example; added: a form describing the records you want). A
query answers "who are the most profitable customers in the Northeast?"
if "profitable" is operationally definable, with no discovery of
patterns. Queries fit when the analyst already suspects a subpopulation:

```
SELECT * FROM CUSTOMERS WHERE AGE > 45 and SEX='M' and DOMICILE = 'NE'
```

Mining runs the other way: it could *produce* that query by discovering
the segment is predictive of churn. **OLAP** (On-line Analytical
Processing) is an easy-to-use GUI for exploring large data collections in
realtime, but its dimensions must be pre-programmed and it does manual or
visual exploration only, with no modeling or automatic pattern finding.

**Data warehousing.** A **data warehouse** collects and coalesces data
from across an enterprise, often from multiple transaction-processing
systems each with its own database. It is a facilitating technology, not
a necessity: most mining does not touch one, but a firm that integrates
sales, billing, and HR records can mine more broadly and deeply.

**Regression analysis** (added: the family of methods for modeling how a
numeric outcome depends on other variables) shares core methods with
data mining but differs in emphasis. Classic regression analysis explains
a particular dataset (why did these customers churn?). Data mining
extracts patterns that **generalize** to cases not in the analyzed data
(which not-yet-departed customers are best to target?), so it spends its
time testing patterns on new data and curbing the tendency to find
dataset-specific patterns. The book names the debate **explanatory versus
predictive modeling**; the notes add **overfitting** for the failure mode
being guarded against. Section 7's held-out test set is that guard in its
simplest form.

**Machine learning and data mining.** **Machine learning methods** are
the collection of methods for extracting predictive models from data,
developed contemporaneously in Machine Learning, Applied Statistics, and
Pattern Recognition. ML arose as a subfield of Artificial Intelligence,
concerned with an agent improving from experience. **Data Mining / KDD**
(Knowledge Discovery and Data Mining) started as an offshoot of ML and
remains closely linked. ML also covers robotics, computer vision, agency,
and cognition, none of which are KDD concerns; KDD is more
application-oriented and cares about the whole analytics process.

The book's test is four questions. "Who are the most profitable
customers?" is a database query. "Is there really a difference between
them and the average customer?" is hypothesis testing (typical result:
under 5% chance the difference is random). "Can I characterize them?"
starts with querying and becomes data science when mining finds what
differentiates them. "Will this new customer be profitable, and how much
revenue should I expect?" is data mining proper: classification
(whether) and regression (how much).

### Choosing a method

The lecture's recipe closes the loop back to Section 1. Consider first
the **type of problem** (classification, regression, and so on), the
**type of data** (different techniques suit specific kinds), and the
**metric of interest** (the measure of result quality you tune the
algorithm toward). Then pick an algorithm, specify its parameters where
needed, apply any applicable optimization techniques on top, and build a
full-fledged **evaluation pipeline** (added: the repeatable process for
measuring model quality on held-out data using the chosen metric). Every
piece now has a referent: problem type is Section 3's hinge, the metric
is Section 7, the parameter is k, and the evaluation pipeline is the
train/test split.

---

## 9. The thread

The module is one idea viewed from four heights. At the table: an extra
response column of labels or values separates supervised from
unsupervised work, and label versus value separates classification from
regression. At the catalog: the book's nine tasks sort along the same
lines, and mining a model is a different activity from using it. At the
process: CRISP-DM wraps modeling in the business question it must answer,
the data it must be honest about (leaks included), the evaluation it must
survive in the lab and then in the business, and the deployment where
value is realized or lost. At the keyboard: kNN is the whole loop in
miniature, a parameter k, a distance function, a vote or an average, a
held-out test set, and a metric that says whether 415 was a good guess.
Everything later in the course is a more capable algorithm dropped into
that same frame.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2120 Faculty and Assistants.*
