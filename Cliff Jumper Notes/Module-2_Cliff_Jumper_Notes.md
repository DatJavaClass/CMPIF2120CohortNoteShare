# Module 2: Cliff Jumper Notes

*One continuous lesson stitched from the Module 2 cliff notes of Applied
Predictive Modeling (CMPINF-2120): why raw data needs work before a model
sees it, the vocabulary for describing attributes and data sets, the
numbers that summarize a column and the histogram that shows it, how
numerical features get rescaled onto common footing, and how string
categories get turned into numbers at all.*

---

## 1. Why preprocessing exists: buses in the river

Module 1 ended with an evaluation pipeline and the promise that every
later algorithm drops into the same frame. Module 2 steps back to the raw
material. **Data preprocessing** (added definition) is the cleaning and
transforming of raw data into a form a model can use, and the lecture
opens with a real project to show why nobody skips it.

The lecturer was principal investigator on the **Pitt Smart Living
Project**, funded by the National Science Foundation, with the tagline of
building a smart city economy and information ecosystem to motivate
pro-social transportation behavior. The problem: a rider's app says the
bus arrives in two minutes, the bus arrives full, and the rider either
cannot board or rides crammed. The goal was an app that warns commuters
when their bus will be full, which needs high-quality real-time data. Two
add-ons rode along: incentives to take a later bus (a discount at the
corner coffee store, discounts on future tickets, the rider relaxes 15-20
minutes and both buses get less crowded) and multi-modal routing (a
shared bike, or a different stop and a different bus). COVID disrupted it
a little. It still produced a lot of data and a lot of models.

Then the map. Green pins are the actual bus stops. Red pins are places a
bus stopped that could not be matched to the proper stop for that bus,
and they come from two independent error sources.

**Human error.** One line of red pins runs down a completely different
route. The Port Authority explained it: someone was supposed to manually
change the bus's route and did not. A similar parallel line of red pins
sits just above the correct 61C route.

**GPS error.** Zoomed into downtown, the data shows buses in the water.
Nobody coded that route wrong. The technology could not find the true
location, because of the **urban canyon effect**: GPS needs line of sight
to at least three satellites for a proper fix, and downtown skyscrapers
block it. (The notes add that a 3D fix uses four satellites. Quote the
lecture's three on a quiz.)

The sanity check is the part worth keeping. If buses were really in the
river, it would be news. Pittsburgh's October 2018 bus in a sinkhole made
national news and memes. No river stories, so the water points are bad
data. Verify a data-error theory against outside evidence before you
trust it.

---

## 2. The vocabulary: attributes and data sets

Before cleaning anything, name what you have. An **attribute** (added
definition) is one recorded property of a data object, a column in a
table. Attributes split by what they hold.

String attributes come in three kinds. **Nominal**: categories with no
order. **Binary**: one of two options. **Ordinal**: categories with an
implied order, such as small, medium, large, extra large. Hold onto
nominal versus ordinal. Section 8 turns on it.

Numeric attributes come in two. **Discrete**: a finite or countably
infinite set of values (zip codes, counts), usually stored as integers,
and a 0/1 binary attribute can be treated as discrete. **Continuous**:
real-number values (temperature, height, weight). A computer cannot store
infinite digits, so continuous values get a fixed number of them,
typically as **floating point** (added definition: a format holding a
fixed count of significant digits plus an exponent). Modern formats
handle significant accuracy.

Data sets sort into three categories: records, ordered sets, and graphs.

**Record data**: every record has the same fixed set of attributes, rows
in an Excel spreadsheet. The lecture's example is 10 records with refund
(yes/no), marital status, income, and whether the person cheated. A
**data matrix** is the case where every attribute is numeric, stored as
an m by n matrix, m rows for objects and n columns for attributes.
**Document data** comes from text and gets its own pipeline below.
**Transaction data** is the products bought in one shopping trip. Storing
it as record data with one column per product is very sparse
(**sparsity**, added: most cells are empty or zero), so not ideal. That
one returns in an upcoming module.

**Graph data** (added definition) is objects plus the links between them:
a costar graph of actors linked by shared movies, a molecule, web pages
and their links. It has its own set of predictive modeling techniques.

**Ordered sets** cover **sequence data**, where the task is often to
predict what comes next, like a number-sequence puzzle, with sequences of
transactions and genomic data as cases (genomics not covered much here),
and **spatiotemporal data**, values indexed by both place and time, such
as monthly land and ocean temperature over many months for locations
around the world.

The document pipeline is short and worth memorizing in order. Six
one-line "documents" sized to fit a slide. Remove **stop words** (very
frequent words like "and", "the", and articles). Convert everything to
lowercase so matching ignores case. **Stem** each word down to its core,
so voted, vote, votes, and voting all map to "vote". Then build a **term
vector** per document: one component per term, valued by how many times
the term appears. Text is not a focus of this course, but that pipeline
is quiz fodder.

---

## 3. The preprocessing task list

The first lecture closes with the list that the rest of the module works
through.

**Missing data**: detect it and handle it. Simplest is to delete the
record. If the record is too valuable to delete, predict the missing
value using the mean of that attribute across everybody else. That is
**imputation** (added definition: filling a missing value with an
estimate), and the lecture's version is mean imputation.

**Outliers**: something clearly meant to be an exception to the norm.
Detect them, then handle them, and the simplest handling is removal.
Other ways exist.

Then three heavier tasks, more processing and more math, deferred to
later lessons: **feature scaling and normalization** (added: rescaling
numeric attributes onto a common range so one does not dominate),
**feature engineering and transformation** (added: building new
attributes or reshaping existing ones), and **encoding categorical
variables** (added: converting string categories into numbers). Scaling
is Section 7 and is built from the statistics in between, so the
module teaches those first. Encoding is Section 8.

---

## 4. Summarizing a column: where the center is

Take one numeric column. The first question is where its **central
tendency** sits, the center of the data.

The **mean** (average): add up the N data points and divide by N. Very
common and very effective as a measure of the center.

```
mean = ( x_1 + x_2 + ... + x_N ) / N
```

The **weighted arithmetic mean** multiplies each value by its weight,
adds them up, and divides by the sum of the weights, not by N. The
lecture says this one will definitely be used in the class.

```
weighted mean = sum( w_i * x_i ) / sum( w_i )
```

One assumption rides along: all weights are positive, so their sum can
never be zero and nothing divides by zero.

The **median**: sort the data and take the value at the border between
the lower and upper halves. Odd count, take the middle one: 3, 3, 5, 9,
11 has median 5. Even count, no single middle value, so average the two
middle ones: 3, 5, 7, 9 has median (5 + 7) / 2 = 6.

The **mode**: the value with the highest **frequency** (added: the number
of times a value appears). One value holds the top frequency, the
distribution is unimodal. Two tie, bimodal. Three or more tie (the
lecture's example is three values sharing the same top frequency), the
general name is multimodal (added term). Every value appears once, no
mode. The transcript garbles the reason, but the reading is plain:
calling every value the mode is pointless.

In the ideal, symmetric case mean, median, and mode are the same value.
The lecture is blunt that this almost never happens in real life. Data
are **skewed** (added definition: the three pulled apart in one
direction). Positively skewed: mean > median > mode. Negatively skewed:
mean < median < mode. The example is salaries, with frequency on the
y-axis (how many employees earn a given salary). Include the CEO and the
mean moves right of the median. The CEO salary is the **outlier** of
Section 3 (a clear exception to the norm), and it artificially inflates
the mean. The
lesson, stated once here and reused in Sections 5, 7, and 9: the average
is good, but it should not be the only number you get.

---

## 5. Summarizing a column: how spread out it is

**Dispersion** is how spread out the data are, and the lecture proves you
need it with three datasets, all values between 1 and 5, all with exactly
the same average of 3:

```
Dataset 1: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5   (uniform)
Dataset 2: all threes                    (pinpoint)
Dataset 3: 1, 5, 1, 5, 1, 5, 1, 5, 1, 5   (bimodal)
```

The average describes dataset 2 perfectly ("all the money is on three")
and dataset 1 reasonably. On dataset 3 it is a problem. Read those as
movie ratings, 1 worst and 5 best: half the audience hates the movie,
half loves it, and "an average movie" does not do it justice. Plot the
distribution to see how the data spread across the range. That is the
setup for Section 6.

The spread measures, in the order the lecture builds them:

**Range**: maximum minus minimum. (The transcript says "mean and the
max". Read min and max.)

**Quartiles** split the sorted distribution at five points of importance
(min, Q1, Q2, Q3, max), giving four chunks the lecture calls Q1 to Q4. A
**quartile** (added definition) is a cut point dividing sorted data into
four equal-count parts. Q1 is the 25th percentile, the bottom quarter
falls below it. Q2 is the median. To find Q1, take the data from the
minimum up to the median and find the median of that lower half (the
transcript says "from the mean", read minimum). Q3 is the median of the
upper half. A **percentile** (added definition) is the same idea with 100
sets instead of four: the p-th percentile is the value below which p
percent of the data fall.

The **boxplot** draws the five-number summary: minimum, Q1, median, Q3,
maximum. Box bottom is Q1, box top is Q3, the line inside is the median,
the bottom and top lines are the minimum and maximum. It is good for
comparing multiple sources or points in time. The height of the box is
the **interquartile range**:

```
IQR = Q3 - Q1
```

The lecture's outlier rule as stated: a value higher or lower than 1.5
times the IQR is an outlier. The standard form (added): below
Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR. The lecture figure marks its
outliers inside a yellow area.

**Variance** is the average of the squared differences between each value
and the mean x-bar (the transcript says "X hat"):

```
variance = ( (x_1 - x_bar)^2 + ... + (x_N - x_bar)^2 ) / N
```

Difference from the average, square it, add the squares, divide by N. The
**standard deviation** is its square root:

```
sd = sqrt( variance )
```

The average plus this one spread number is a good statistic for any
numerical data. (Added: dividing by N gives the population variance,
sample variance divides by N - 1, and the lecture divides by N. That
detail comes back in Section 7.)

---

## 6. Seeing the distribution: histograms

A **histogram** plots the frequencies of the data as rectangles, values
or ranges of values on the x-axis and frequency on the y-axis. The trick
is the **bucket** (also called a bin, added term): a range of values
combined into one rectangle. One bar per value is fine for a small
dataset. With real data you cannot see what is going on.

Two kinds. **Equal width**: every bucket covers the same portion of the
x-axis. **Equal frequency** (equal height): buckets are built so every
bar reaches about the same height.

The lecture's dataset is store prices from 1 to 30 with a count per
price:

```
price:  1  5  8 10 12 14 15 18 20 21 25 28 30
count:  2  5  2  4  1  3  6  8  7  4  5  2  3
```

Total count 52. (The count for 12 is never stated directly. It is implied
by "1 plus 3 makes 4" when 12 and 14 are added.) The trivial histogram
gives every price its own bucket, 13 bars, and is fine here because
there are few items.

**Equal width, three buckets.** Total range is max - min + 1 = 30 - 1 + 1
= 30, the +1 counting both integer endpoints (added), unlike the plain
range of Section 5. Width per bucket is 30 / 3 = 10. Start at the
minimum, so the ranges are 1-10, 11-20, 21-30. Add the counts inside
each:

```
Bucket 1 (1-10):   1, 5, 8, 10        -> 13
Bucket 2 (11-20):  12, 14, 15, 18, 20 -> 25
Bucket 3 (21-30):  21, 25, 28, 30     -> 14
```

Check: 13 + 25 + 14 = 52.

**Equal height, four buckets.** Total count 52, so the target height is
52 / 4 = 13 per bucket, "about" 13 because fractions of an item are not
allowed. The rule: walk the prices in order keeping a running frequency
(the lecturer compares it to counting cards). At the threshold, compare
the running total just before crossing 13 with the one just after, and
keep whichever is closest to 13. Going over does not lose. This is not
poker.

Bucket 1: 1 (2), 5 (7), 8 (9), 10 (13). Exactly 13, prices 1 through 10.
Bucket 2: 12 (1), 14 (4), 15 (10). Adding 18 would give 18. Between 10
and 18, 10 is closer to 13, so stop at 15, and 18 starts the next bucket.
Bucket 3: 18 (8), 20 (15). Between 8 and 15, 15 is closer, so include 20.
Bucket 4: 21 (4), 25 (9), 28 (11), 30 (14). Between 11 and 14, 14 is
closer, and the lecturer says to do that math on the last bucket too,
which confirms no leftovers.

```
Bucket 1:  1-10   frequency 13
Bucket 2: 12-15   frequency 10
Bucket 3: 18-20   frequency 15
Bucket 4: 21-30   frequency 14
```

Check: 13 + 10 + 15 + 14 = 52. Both examples were recomputed for these
notes and match the lecture exactly.

---

## 7. Putting features on the same footing

Now the first of the deferred tasks from Section 3. Most predictive
modeling techniques consider many **features** (added: one measured
attribute of a data item) at the same time, and numerical features in one
dataset come from different **domains** (added: the range of values an
attribute takes, in its own units). The lecture's example is working
adults: age about 18 to 70, income about 25,000 to 150,000, ranges
thousands of times apart in the same table. The fix is to transform the
features into a space that allows easy comparison across domains, and
each of the three transforms is built from a Section 4 or 5 statistic.

**Standardization**, also called variance scaling or **Z-score scaling**:
subtract the feature's mean, divide by its standard deviation.

```
z = (x - mean(x)) / std(x)
```

(The transcript says "divided by the variance" in one breath and "divide
by the standard deviation" in the next. Standard deviation is correct.)
The result has mean exactly 0 and variance 1, so standard deviation 1 as
well. A **z-score** (added definition) is the number of standard
deviations a value sits above or below the mean. Ideal for algorithms
that require, or at least perform better with, standardized input.

**Min-max scaling**: find the minimum and maximum, map every value into 0
to 1.

```
x' = (x - min(x)) / (max(x) - min(x))
```

(The transcript's denominator is "the maximum and the mean". It is
maximum minus minimum, or the minimum would not land on 0.)

**Robust scaling**: center on the median instead of the mean, divide by
the interquartile range instead of the standard deviation or the range.
The lecture states it in words. The formula (added):

```
x' = (x - median(x)) / (Q3(x) - Q1(x))
```

The transcript says "subtract from the first quartile" and one sentence
later "centered using the median". Median centering is what
scikit-learn's RobustScaler does, so memorize that version. It is ideal
for data that contains outliers, because outliers pull the mean, standard
deviation, min, and max, while the median and IQR mostly ignore them
(added). Section 4's CEO salary is the kind of outlier meant (added).

Side by side (added summary):

```
Method            Subtract    Divide by      Output
Standardization   mean        std deviation  mean 0, var 1
Min-max scaling   minimum     max - min      range 0 to 1
Robust scaling    median      IQR (Q3 - Q1)  median 0, IQR 1
```

The Python demo uses pandas, scikit-learn, and NumPy on 10 generated
rows, age 22 to 70 and income 35,000 to 140,000, with `describe()`
reporting mean, standard deviation, min, max, and the 25th, 50th, and
75th percentiles. Reported means: age 47, income 74,500. Each transform
is one call to its scaler (StandardScaler, MinMaxScaler, RobustScaler in
sklearn.preprocessing, class paths added). Standardization and robust
scaling produce negative values, so make sure the downstream algorithm
handles them. Min-max lands on 0 at the minimum and 1 at the maximum by
inspection.

One wrinkle the professor pointed at without fully explaining: after
StandardScaler, `describe()` shows a mean of 0 or extremely close, but a
standard deviation very close to 1, not exactly 1. With 5 rows it sat
further off, and it approaches 1 as the row count grows. The exact cause
(added) is the N versus N - 1 note from Section 5: StandardScaler divides
by n, `describe()` reports the sample standard deviation dividing by
n - 1, so the reported value is sqrt(n / (n - 1)), which is 1.118 at
n = 5 and 1.054 at n = 10.

---

## 8. Turning strings into numbers

The last deferred task. Most machine learning algorithms operate only on
numerical values, and real data is full of strings, so every string
column has to be mapped to numbers somehow. **Feature encoding** is that
mapping, from a **categorical** column (added: values are names from a
fixed set) to numeric ones. The lecture's opening example is three
levels, high, medium, low, mapped to 2, 1, 0. Many mappings are possible,
and the choice depends on the Section 2 distinction between nominal and
ordinal.

**Label encoding**: each category gets one integer, assigned in
alphabetical order of the values. You do not choose the order. **Ordinal
encoding**: each category gets one integer in an order you specify,
because the data has an inherent order. **One-hot encoding**: one new
binary column per category, and in any row exactly one of them is "hot"
(1 or True). For data with no inherent order, where imposing one would be
wrong. The lecture's decision rule: order irrelevant, label encode. Order
matters and you can state it, ordinal encode. No order exists and none
may be assumed, one-hot encode. Adopting a false order "by accident or on
purpose" risks wrong modeling results.

The demo is the Titanic data, a Kaggle competition that became a very
useful starter set for classification problems. Notebook 4.1 keeps five
columns and drops any row with a missing value (891 rows become 889,
added from a run):

```python
df = sns.load_dataset('titanic')
df = df[['survived', 'sex', 'embarked', 'class', 'who']].dropna()
```

survived is 0 or 1. sex is male or female. embarked is C Cherbourg, Q
Queenstown, S Southampton. class is Third, First, Second. who is man,
woman, or child.

**Label encoding on sex.** `LabelEncoder().fit_transform(df['sex'])`,
where **fit_transform** (added) learns the mapping and applies it in one
call. Checking with `df[['sex', 'sex_le']].drop_duplicates()` keeps only
unique combinations, so a proper mapping yields one row per category.
Result, verified by running the notebook: female 0, male 1. Alphabetical,
which is why female lands on 0.

**Ordinal encoding on class.** Alphabetical will not cut it when the data
has an order, so state it:

```python
ordinal_order = [['Third', 'Second', 'First']]
oe = OrdinalEncoder(categories=ordinal_order)
df['class_ordinal'] = oe.fit_transform(df[['class']])
```

Result: Third 0.0, Second 1.0, First 2.0, a higher number for a higher
class of service, returned as floats (added). Two bracket details
(added): LabelEncoder takes a 1-D Series, `df['sex']`, while
OrdinalEncoder takes a 2-D DataFrame, `df[['class']]`, and one category
list per column, hence the nested list. Any order you specify works. The
choice is yours, not the encoder's.

Notebook 4.2 does the same thing by hand with a dictionary and
`Series.map`: `education_map` sends High School, Bachelors, Masters, PhD
to 0, 1, 2, 3, and five rows of degrees come out 0, 1, 2, 3, 1, both
Bachelors rows landing on 1.

**One-hot encoding on embarked.**

```python
df_ohe = pd.get_dummies(df, columns=['embarked'], prefix='embarked')
```

New columns are prefix, underscore, value: embarked_C, embarked_Q,
embarked_S, and the original column is dropped (added, from a run). A
**dummy variable** (added) is one such 0/1 column flagging membership in
one category, and one-hot encoding creates as many of them as the
column's **cardinality** (added: its number of distinct categories). Rows
0, 1, and 5 show the three cases, S, C, and Q, with exactly one True per
row. One discrepancy worth knowing: the notebook cell's last line is
`df[['embarked']].drop_duplicates()`, so Jupyter displays the original
column, not the three dummy columns the lecture narrates. The dummy view
is the line above, computed but not shown.

The manual version in notebook 4.2 exposes the mechanism, "less of a
black box":

```python
df['gender_male'] = (df['gender'] == 'Male').astype(int)
df['gender_female'] = (df['gender'] == 'Female').astype(int)
```

The comparison yields True or False per row, and `astype(int)` turns that
into 1 or 0. Rows 0 and 3 are Male, rows 1 and 2 are Female.

Two notebook facts the lecture skips (added): notebook 4.2 imports
LabelEncoder, OneHotEncoder, ColumnTransformer, Pipeline,
LogisticRegression, and numpy and uses none of them, and it puts the
heading "Label Encoding (for ordinal data)" over education_order, a
dictionary identical to education_map, while the lecture calls the same
code ordinal encoding. Same code, two names. Neither notebook ships saved outputs, so
every output above came from re-running the cells.

---

## 9. The thread

The module is one question asked four ways: what does a model actually
receive? First, the raw feed, which has buses in the river until a human
checks the news. Second, the vocabulary, where nominal versus ordinal and
discrete versus continuous decide what can even be done to a column.
Third, the statistics, where a center and a spread describe a column, a
histogram shows it, and the CEO's salary teaches that the average alone
is not enough. Fourth, the transforms: the mean and standard deviation,
the min and max, the median and IQR each become a scaler that puts age
and income in the same space, and the nominal-or-ordinal call from the
vocabulary decides whether a string gets a label, an order, or a column
of its own. Module 1's evaluation pipeline sits downstream of all of it.
The model only sees what preprocessing lets through.

---

*Authored and directed by **DatJavaClass (Victor S)**, who conceived, structured, formatted, fact-checked, and edited these notes, with assistance by Claude. Some material may have been derived from assigned material, but has not been copied verbatim. For source materials please contact CMPINF-2120 Faculty and Assistants.*
