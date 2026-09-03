# Module 2: Cliff Jumper Notes

*One continuous lesson stitched from the nine Module 2 cliff notes of Applied
Predictive Modeling (CMPINF-2120): why raw data needs work before a model sees
it, the vocabulary for attributes and data sets, the numbers that summarize a
column and the histogram that draws it, how numerical features get rescaled
onto common footing, how string categories become numbers, and then the five
video arc on principal component analysis: rows as arrows, projection and
spread, the covariance matrix and its eigenvectors, what PCA is and why it
pays, the three step recipe in scikit-learn, and how to choose components and
read what comes back.*

---

## 1. Why preprocessing exists: buses in the river

Module 1 ended with an evaluation pipeline and the promise that every later
algorithm drops into the same frame (added bridge from the Module 1 notes).
Module 2 steps back to the raw material.
**Data preprocessing** (added definition) is the cleanup and reshaping you do
to raw data before modeling, so the model sees data that is complete,
consistent, and comparable. The lecture opens with a real project to show why
nobody skips it.

The professor was principal investigator on the **Pitt Smart Living Project**,
funded by the National Science Foundation, with the tagline of building a
smart city economy and information ecosystem to motivate cross-social
transportation behavior. The problem: you are at a bus stop, the app says two
minutes, the bus arrives full, and you either miss it or ride crammed. The
goal was an app that notifies commuters when their bus will be full, which
needs high-quality real-time information. Two add-ons rode along: incentives
to take a later bus (a coffee discount at the corner store, discounts on future
tickets, and the rider relaxes 15 to 20 minutes and boards a less crowded bus)
and alternative multi-modal routing (shared bikes, or walk to a different stop
and catch a different bus). COVID disrupted the project, but the team
collected a lot of data and built a lot of models from it.

Then the data. The lecture shows a map with red and green push pins. Red pins
are places a bus stopped that could not be matched to a proper bus stop for
that route. Green pins are the actual stops. Two different causes produced the
same symptom:

- **Human error.** One line of red pins runs down a completely different bus
  route. The Port Authority explained that someone was supposed to manually
  change the route the bus was assigned to and did not. A similar parallel
  track appears above the correct 61C line.
- **GPS error.** Zoomed into downtown, buses appear in the water. Nobody coded
  a route through the river. The GPS could not get a proper fix.

The second one has a name. The **urban canyon effect**: GPS devices need line
of sight to at least three satellites to get a proper fix, and tall downtown
skyscrapers block that line of sight, so positions drift. How do we know the
river data is wrong? If there were buses in the river, there would be a news
story. There was a bus in a Pittsburgh sinkhole in October 2018, and that made
national news and memes. The lesson underneath the joke (added note): domain
knowledge and common sense are part of validation, and different causes of
dirt need different fixes.

## 2. Naming the data: attributes and data sets

Before anything gets cleaned, the lecture fixes vocabulary. An **attribute**
(added definition) is one measured property of a record, a column in the
table, also called a feature or variable. String attributes come in three
flavors. **Nominal** strings have no order among them (added examples: color,
or the port a passenger boarded at). **Binary** strings offer one of two
options. **Ordinal**
strings carry an implied order, the lecture's example being small, medium,
large, extra large. Numeric attributes come in two. **Discrete** attributes
take a finite or countably infinite set of values (zip codes, counts), usually
stored as integers, and a 0/1 binary attribute can be treated as discrete too.
**Continuous** attributes take real numbers as values (temperature, height,
weight). A computer cannot store an infinite number of digits, so continuous
values are recorded with a fixed number of digits, typically as floating-point
variables, and modern floating-point formats handle significant accuracy.

Hold on to nominal versus ordinal. Section 7 turns on exactly that split.

Data sets fall into three categories: records, ordered sets, and graphs.
**Record data** gives each record a fixed set of attributes, rows in a
spreadsheet. A **data matrix** is record data where every attribute is
numeric, represented as an m by n matrix, m rows (one per object) and n columns
(one per attribute). That m by n shape is the form most modeling algorithms
want (added note), and it comes back in section 8 as "the whole data table as
one block of numbers." **Document data** comes from text and gets three
standard preprocessing steps: remove **stop words** (very frequent words that
carry little meaning on their own, like "and" and "the"), convert to lowercase
so matching ignores case, and **stemming** (reduce a word to its core, so
voted, vote, votes, and voting all become vote). After that, each document
becomes a **term vector**, one component per term, the value being how many
times that term appears. **Transaction data** is the grocery-trip list of
products bought at once, and storing it as one column per possible product
leaves most cells empty, the condition called **sparsity** (added definition).
An upcoming module covers transaction data. **Graph data** is data where the
relationships between objects are the point (added definition): a co-star
graph, molecules, web pages linking to each other. Ordered data includes
**sequence data** (predict what comes next, genomic data being the famous
case) and **spatiotemporal data** (indexed by both place and time, such as
monthly land and ocean temperature by location).

The preprocessing jobs themselves, simplest first: detect and handle missing
data, detect and handle outliers, feature scaling and normalization on
numerical values, feature engineering and transformation, and encoding
categorical variables. The last three involve more processing and more math
and get their own lessons. For missing data the two simple options are to
delete the record, or, if the record is too valuable to delete, predict the
missing value using the mean of everybody else for that attribute. Filling
with an estimate is **imputation** (added definition). An **outlier** is a
value that is clearly an exception to the norm, and the simplest handling is
to remove it. Other ways exist.

## 3. Summarizing a column: mean, median, mode, skew

The histogram lecture is about summarizing numerical data with as few numbers
as possible, then drawing it. Start with **central tendency** (added
definition), a single number that stands for the center of a set of values.

The **mean** (average): with N data points, add them up and divide by N.
Surprisingly common and very effective. The **weighted arithmetic mean**: when
each value carries a weight, multiply each value by its weight, add those
products up, and divide by the sum of the weights, not by N. The assumption is
that all weights are positive, so the sum of weights can never be zero and you
never divide by zero. The class will definitely use this one. Three quiz
scores of 88, 72, and 95 with weights 0.2, 0.3, and 0.5 have a plain mean of
85.0 and a weighted mean of (88 times 0.2 + 72 times 0.3 + 95 times 0.5)
divided by 1.0, which is 86.7. Dividing by the weight sum is what keeps the
answer on the same scale as the scores. Watch the denominator fail once:
with weights 0.4, 0.6, and 1.0 (sum 2) the weighted sum is 173.4, dividing by
N gives 57.8, and dividing by the weight sum gives the correct 86.7 (added
example).

The **median**: sort the values from lowest to highest and take the one on the
border between the lower half and the upper half. With an odd count that is
the middle value. With an even count there is no single middle value, so the
median is the mean of the two middle values. The values 15, 2, 9, 4, 11 sort
to 2, 4, 9, 11, 15 with median 9. Drop the 11 and the even-count median is
(4 + 9) / 2 = 6.5.

The **mode**: the value that appears most often, the one with the highest
frequency. One value at the top is **unimodal**, two tied at the top is
**bimodal**, and three or more can tie as well, the lecture's wording being
three numbers that all have the exact same frequency, and that frequency is
the highest over everybody else. If every value shows up only once there is
no mode. There is no point in calling every number the mode.

Now put the three together. In the ideal symmetric case the mean, median, and
mode are all the same value, and this almost never happens in real life. Real
data is skewed. **Skew** (added definition) is the direction a distribution's
tail stretches, which pulls the mean away from the median and mode.
**Positively skewed**: mean > median > mode. **Negatively skewed**: mean <
median < mode. The lecture's salary picture makes it concrete. Once the CEO is
included, the mean lands to the right of the median, because one huge value
drags the sum and therefore the mean upward, while the median only cares
which value sits in the middle of the sorted list. Eight salaries in
thousands, 42, 45, 48, 50, 52, 55, 60, 400, have mean 94.0 and median 51.0.
Nobody but the 400 earns anywhere near 94. The average is good, but it should
never be the only number you report.

## 4. Spread: variance, standard deviation, quartiles, boxplots

The same-mean demonstration is the reason spread gets its own numbers. The
lecture shows three data sets with exactly the same average: one spread
uniformly across its range, one pinned to a single value, one bimodal with
half the values at each extreme. The framing is movie ratings from 1 to 5
where half the audience hates the film and half loves it. Calling it "an
average movie" does not do it justice. **Dispersion** is how spread out the
data is around its center, and there are two ways to see it: plot the
distribution, or compute a number.

The numbers first, because the PCA videos run on them. **Variance**: for each
value, take its difference from the average, square that difference, add all
the squares up, and divide by n. **Standard deviation**: the square root of
the variance, which puts it back in the original units (last clause added).
Why square? Differences above and below the mean would cancel if you just
added them, since the sum of raw differences from the mean is always zero.
Squaring makes every difference positive and makes big misses count for more
than small ones (added explanation). The values 6, 8, 10, 12, 14 have mean 10,
differences -4, -2, 0, 2, 4, squares 16, 4, 0, 4, 16, sum 40, variance 40 / 5
= 8.0, standard deviation 2.83. The lecture's verdict: the average plus this
one extra number is a good statistic to have about any numerical data.

The version of spread that feeds a plot is the **quartiles**. Split the sorted data
at five points of importance, which makes four chunks. Q1 is the 25th
percentile, the bottom quarter of the data falls below it. Q2 is the median.
Q3 is the 75th percentile. The by-hand recipe: find the median, then take the
lower half from the minimum up to the median and find its median (that is
Q1), and do the same for the upper half from the median to the maximum (that
is Q3). **Percentiles** (added definition) are the same idea with 100 chunks.
A **boxplot** draws the **five-number summary** (minimum, Q1, median, Q3,
maximum): the box runs from Q1 to Q3, the line inside is the median, and the
bottom and top lines are the minimum and maximum. The height of the box is the
**interquartile range (IQR)**, Q3 minus Q1, and the definition normally used
is that a value more than 1.5 times the IQR beyond the box is an outlier. Ten
sorted values 3, 5, 7, 8, 12, 13, 14, 18, 21, 40 have median 12.5, Q1 = 7,
Q3 = 18, IQR = 11, and cutoffs at -9.5 and 34.5. The 40 is the outlier.

Two of these numbers do heavy lifting later. The IQR and the median drive
robust scaling in section 6. The variance is the whole engine of sections 11
and 12.

## 5. Histograms: equal width and equal frequency

A **histogram** represents the frequencies of the data as rectangles. The
trick is that you do not always give every value its own rectangle. You
combine values into **buckets** (bins, added definition), because on any real
data the one-value-per-bar version tells you nothing. Two main types. An
**equal width histogram** gives each bucket the same portion of the x-axis. An
**equal frequency (equal height) histogram** builds the buckets so that each
one's height on the y-axis is about constant, "about" because fractions are
not allowed.

Equal width recipe: range = max minus min plus 1, width = range divided by
the number of buckets, the first bucket starts at the minimum and covers one
width, and each next bucket starts where the last one ended. Then add up the
frequencies in each bucket. Take these prices and counts:

```
price:  1  4  7  9 11 12 15 17 19 22 24
count:  3  6  2  5  7  4  9  3  5  2  4
```

Fifty items. With four buckets the range is 24 - 1 + 1 = 24, the width is 6,
and the buckets are 1 to 6, 7 to 12, 13 to 18, and 19 to 24, holding 9, 18,
12, and 11 items. The heights vary, the widths do not. That is the whole
definition.

Equal frequency recipe: add up all the counts, divide by the number of
buckets for the target height, and walk the values in order adding counts
into the current bucket. When adding the next value would cross the target,
compare the running total just below the target with the running total just
above it, and whichever is closer to the target wins. It is a little bit like
counting cards in poker, except going over does not lose. The value that was
left out starts the next bucket. And do the math for the last bucket too,
even though it takes whatever is left: the professor's advice for multiple
choice is the same, look at all the answers before deciding. Same table with
three buckets and a target of 50 / 3 = 16.67: bucket 1 runs 3, 9, 11, 16 and
adding the next count of 7 would make 23, and 16 is closer, so bucket 1 is
prices 1 through 9 at 16. Bucket 2 runs 7, 11 and adding 9 makes 20, and 20 is
3.33 away against 5.67 for 11, so 15 is included and bucket 2 is prices 11
through 15 at 20. Bucket 3 takes the rest, 17 through 24, at 14. Check: 16 +
20 + 14 = 50. Heights near the target, widths nowhere near equal. That is the
trade an equal frequency histogram makes.

## 6. Putting features on common footing: standardization and scaling

Most predictive modeling techniques consider many features at the same time,
and those features come from different domains while sitting in the same
data set. The lecture's example is age, roughly 18 to 70 for working adults,
next to income, roughly 25,000 to 150,000. Anything that measures distance or
adds features together will let the feature with the biggest numbers
dominate purely because its numbers are bigger (added explanation). Income
would drown age. The fix is to transform the numerical features into a space
that allows easy comparison across domains, through standardization and
scaling. All three tools are one-liners from scikit-learn's preprocessing
module.

**Standardization**: subtract the feature's mean from each value, then divide
by the feature's standard deviation. Also called variance scaling or
**Z-score scaling**. The transformed feature has a mean of exactly zero and a
variance of one, so every feature ends up answering the same question, "how
many standard deviations from average is this value," regardless of the
original units. It is ideal for algorithms that require standardized input or
perform better with it. Two things to watch in the output. Negative values
appear (anything below the mean goes negative), so make sure the algorithm
you feed this to handles them. And the standard deviation reported by
`describe()` is very close to one, not exactly one, on a small data set. The
lecture had ten rows and saw the gap shrink when it went from five rows to
ten. The gap exists because `StandardScaler` divides by the population
standard deviation (divide by n) while pandas `describe()` reports the sample
standard deviation (divide by n minus 1), and the ratio between them is the
square root of n / (n - 1), which is 1.054 at the lecture's n = 10, 1.069 at
the n = 8 of the salary example below, and creeps toward 1 as n grows (added
explanation). Keep that n versus n minus 1 distinction handy. It returns in
section 11.

One value by hand makes the formula concrete (added example). Eight people
with heights 158, 163, 170, 172, 175, 181, 188, 195 cm have mean 175.25 and
population standard deviation 11.573, so the 195 sits 19.75 above the mean
and standardizes to 19.75 / 11.573 = 1.707. The shortest person, 158, lands
at -1.491, the negative value the lecture warns about. Both numbers now say
"how unusual is this value for its column," which is what makes a height and
a salary comparable.

**Min-max scaling**: find the feature's minimum and maximum, subtract the
minimum from each value, then divide by the distance from minimum to maximum.
The smallest value becomes exactly 0, the largest exactly 1, and everything
else lands in between in proportion. No negatives. The shape of the
distribution does not change, only its scale (added explanation). The
weakness is that the extremes define the range. Give those same eight people
salaries of 41,000, 52,000, 48,000, 61,000, 75,000, 58,000, 99,000, and
250,000, and the 250,000 outlier owns the value 1.0 while every other salary
is squeezed below 0.28.

**Robust scaling**: center on the median instead of the mean, and divide by
the interquartile range instead of the standard deviation or the min-to-max
range. The lecture's framing is that the transformed data is centered using
the median instead of the mean, and scaled by the interquartile range. The
median and the IQR barely move when an outlier shows up, while the mean, the
standard deviation, and especially the min and max jump, so robust scaling is
ideal for data that contains outliers. That same 250,000 salary comes out at
2.553 standardized, 1.000 min-max, and 6.35 robust: robust scaling pushes the
outlier furthest from the bulk of the column while the middle half of the
column lands inside a band one unit wide around zero (added explanation).

One mechanical note that matters for the PCA recipe later: all three scalers
are fit on the data (they learn the mean and standard deviation, the min and
max, or the median and IQR) and then applied. The word "fit" in
`fit_transform` is the learning step and "transform" is the application
(added explanation).

## 7. Turning strings into numbers: label, ordinal, one-hot

Most machine learning algorithms operate only on numerical values, and real
data is full of strings, so the strings have to be mapped to numbers in some
way. The lecture's opening example maps high, medium, and low to 2, 1, and 0.
There are many ways to do the mapping, and picking the wrong one quietly
tells the model something about the data that is not true. That is the whole
lesson, and it runs on the nominal versus ordinal split from section 2.

The data is the **Titanic** data set, which started as a Kaggle competition
and became a standard starter data set for classification problems. Five
attributes are kept (survived, sex, embarked, class, who), rows with any
missing values are dropped, and embarked holds C for Cherbourg, Q for
Queenstown, and S for Southampton.

**Label encoding** assigns each distinct category an integer. `LabelEncoder`
with `fit_transform` on the sex column produces a new column, and the
lecture's check idiom, select the original and encoded columns and call
`drop_duplicates()`, leaves one row per unique pair: female became 0, male
became 1. The order is alphabetical. `LabelEncoder` sorts the distinct values
and numbers them in that order, which is fine when the order carries no
meaning and nothing downstream treats the numbers as ranks (added). On a color column
holding red, blue, and green, blue becomes 0, green 1, red 2, regardless of
the order they appeared.

**Ordinal encoding** is for when pure alphabetical order will not cut it
because the data has an inherent order, and you specify it. `OrdinalEncoder`
with `categories=[['Third', 'Second', 'First']]` on the class column gives
Third 0, Second 1, First 2, a higher value for a higher class of service, and
any order you specify works. Two mechanics (added): `OrdinalEncoder` takes a
2-D input, so the column is passed with double brackets, and the categories
argument is a list of lists, one inner list per column. The lecture also does
the job by hand on a five-row education table: a dictionary with High School
0, Bachelors 1, Masters 2, PhD 3 as key-value pairs, applied with `map()`.
Same information either way. Shirt sizes S, M, L, XL are the everyday case
(added), since alphabetical order (L, M, S, XL) would be nonsense.

**One-hot encoding** is for the third case: there is no order and you should
not use one. If you adopt an order by accident or on purpose, you risk
assuming a wrong order in your modeling, and the results may not be what you
expect. A numeric code says to the model that Southampton (2) is "more" than
Cherbourg (0) and that Queenstown sits halfway between, and linear and
distance-based models take that literally (added). One-hot creates a new
binary column for each category, and for any one row exactly one of the new
columns is hot (1 or True) while the rest are 0 or False. One line does it:
`pd.get_dummies` on the embarked column with prefix embarked produces
embarked_C, embarked_Q, and embarked_S. The manual version is one comparison
per category, `(df['gender'] == 'Male').astype(int)`, which the lecture shows
not because it is better but to expose the mechanism. The cost (added): k
categories become k columns, fine for three ports, and a zip code column with
40,000 distinct values becomes 40,000 mostly-zero columns, which is the
sparsity problem from section 2.

The decision rule is in notebook 4.2's own headings. It files the
dictionary-plus-`map()` approach under "Label Encoding (for ordinal data)"
and `get_dummies` under "One-Hot Encoding (for nominal data)". Those
parentheses are the rule: ordinal data gets integer codes in a chosen order,
nominal data gets one-hot.

## 8. Rows as points, points as arrows

The rest of the module is a five video series on **principal component
analysis (PCA)**. Textbook treatments are heavy on math. This series builds
the intuition through linear algebra fundamentals, visuals on real data, and
a slow pace, and every notebook code cell carries a note that it exists only
to generate the figure. The concepts are the deliverable, not the code. The
thread to follow (added): a row of data is a point, a point is an arrow, an
arrow can be measured along any direction, and some directions capture more
of the data's spread than others. PCA is about finding those directions.

In applied predictive modeling a data point is just some measurements of a
real object. The warm-up data is eight vehicles measured on length in meters
and weight in kilograms, from the Smart Fortwo (2.7 m, 900 kg) to the
delivery van (6.5 m, 3,500 kg). Plot length on x and weight on y and each
vehicle is a dot (see the scatter plot from cell 5 of
5.1.geometry_of_data.ipynb). The notebook's phrasing of the key idea: a row of
numbers and a point in space are the same information in two forms.

A **vector** is an arrow from the **origin** (0, 0) out to the point, with a
direction and a length, also called magnitude. For the vehicles the origin is
natural, no length and no weight. The notation: a **scalar** is a single
number, written in regular type, occasionally a Greek letter (not because the
professor is from Greece, that is just the math convention). A vector is an
ordered list of numbers written in bold, and its numbers are its
**components**. Column form stacks them vertically and is the math
convention. Row form writes v = [4.6, 1500] side by side, the computing view,
one record in the data set. Same vector either way. **Theta** is the angle
between two arrows. Stack many vectors as rows and you get a **matrix**, the
whole data table as one block of numbers, the m by n data matrix from
section 2 under its linear algebra name.

Then the data set for the rest of the unit: the **Palmer Penguins**, real
body measurements for three species, Adélie, Chinstrap, and Gentoo, near
Palmer Station in Antarctica, collected by Dr. Kristen Gorman and made
available by Allison Horst, a little over 300 rows. Two measurements to
start: flipper length in millimeters and body mass in grams (the notebook
plots mass in hundreds of grams, a unit relabel so the two axes are visually
comparable, not standardization). For penguins the origin makes no sense. No
penguin is anywhere near zero flipper length and zero mass, so arrows from
(0, 0) would all point the same way toward a far-off corner and say nothing
about how penguins differ. What you actually want is how each penguin differs
from the average penguin, so measure from the **center of the cloud** (the
point whose coordinates are the mean of each attribute, added definition).
**Centering** (added definition) subtracts the mean of each attribute from
every value, so the cloud's center moves to (0, 0) and every arrow starts
there. Now each arrow's direction says which way that penguin differs from
average and its magnitude says how far. Vehicles or penguins, the picture is
the same: points in space, arrows for direction and magnitude, and a cloud
that leans in some direction. Which direction matters most? That is the
question the rest of the arc answers.

## 9. The dot product, the norm, and projection

Pick any direction, meaning a line through the center. The **dot product** of
a penguin's vector with that direction measures how far along that direction
the penguin reaches. Large positive when the vector points the same way as
the direction. Near zero when the vector is perpendicular to it. The
arithmetic form (added): multiply matching components and add, so [3, 4] dot
[4, 1] is 3 times 4 plus 4 times 1, which is 16. The notebook's geometric
form: v dot u equals the norm of v times the norm of u times the cosine of
theta, where the **norm** is the length of the arrow, defined properly in a
moment. Cosine of 0 degrees is 1, so two arrows pointing the same way give the
full product of their lengths. Cosine of 90 degrees is 0, so perpendicular
arrows give zero no matter how long they are. The cosine is the alignment
dial and the two norms scale it by how long the arrows are (added
explanation). In NumPy the `@` operator is the dot product for 1-D arrays
(added).

The norm, then: the square root of the dot product of the vector with
itself. For [3, 4] that is the square root of 9 + 16, so 5, the classic 3-4-5
triangle: the norm is the hypotenuse (added). Two vectors
are **orthogonal** when they meet at a right angle, which is the same as
saying their dot product is zero. A vector of length one is a **unit vector**,
and orthogonal plus unit length is **orthonormal**, a term the lecture flags
as one you will need soon. Dividing a vector by its own norm makes it a unit
vector with the same direction, and directions are always stored as unit
vectors in this material so that a dot product with one reads as a pure
distance along the line (added).

**Projection**: pick a direction, drop a perpendicular from the tip of the
arrow to the line, and the foot of that perpendicular is the projection. It
is how much of the penguin lies along that direction. The notebook's
statement of the link: the dot product IS the projection length, up to the
length of the direction, so with a unit-length direction the dot product is
exactly the signed distance from the center to the foot (added qualifier).
The leftover piece of the arrow that does not lie along the line is itself
orthogonal to the direction, which is what "foot of the perpendicular" means.
Cell 17 of the notebook shows one penguin vector against three direction
lines, aligned, partly aligned, and perpendicular, and the dot product shrinks
as the alignment drops.

Once by hand (added example). Take v = [3, 4] and the direction 30 degrees
above the x-axis, whose unit vector is [0.866, 0.5]. The dot product is
4.598, so the arrow of length 5 reaches 4.598 along that line. The foot of
the perpendicular is 4.598 times the unit vector, [3.982, 2.299]. The
leftover, v minus the foot, is [-0.982, 1.701], and its dot product with the
direction is zero up to floating-point crumbs: the piece that stuck out really
is perpendicular. Rotate the direction to 120 degrees and the same v projects
to only 1.964. Same arrow, different line, very different amount carried
over. That last sentence is the whole next section.

## 10. Spread depends on direction

Now project every penguin onto the line. The 2-D cloud collapses to a 1-D
spread of points along that direction, and how spread out those points are
depends on which direction you chose. Some directions squeeze the penguins
together. One direction spreads them out the most. The notebook measures
**spread** as the standard deviation of the projected values (added
definition), and the next video renames it variance.

Why the biggest spread is the goal: the more spread the projection keeps, the
more of the information in the cloud survives the squeeze from two numbers
per penguin down to one. A direction with tiny spread has crushed everyone
into nearly the same value and thrown the differences away. Cell 23 of the
notebook sweeps every direction from 0 to 180 degrees and shows the widest
one (spread 15.8) between two directions 45 degrees off on either side, one
of them the lecture's so-so direction (spread 11.4). Six students measured on
hours studied and quiz score (score in tens so the axes are comparable, the
notebook's relabel trick) make the same point in small numbers. The points
are [2, 5.5], [4, 6.2], [5, 7.0], [7, 7.4], [8, 8.5], and [10, 9.0]. Center
them, and projecting the cloud onto the hours axis alone gives spread 2.646,
onto the score axis alone 1.216, onto the 45-degree diagonal 2.721, and a
sweep of every whole-degree angle peaks at 24 degrees with spread 2.905. The widest direction is not
either original axis. It is a tilt of the two, because the cloud leans up and
to the right and the diagonal catches both kinds of variation at once.

This was done by hand, or by brute-force sweep. The question for video 2: can
the data itself tell us the direction of maximum spread without guessing?
That best direction has a name, and it comes from the data's covariance.

## 11. From spread to variance: the covariance matrix

First change of vocabulary: "spread" becomes **variance**. Spread is the
visual thing, variance is the mathematical thing. The variance of a
projection is the average squared distance of the projected points from their
mean, which is the section 4 variance applied to the projected values, and
the video 1 question becomes precise: which direction has the maximum
projected variance?

The refresher. The **variance of one feature** is the average squared
deviation from its mean. The **covariance of two features** is the average of
the product of their deviations: for each row, (x minus x bar) times (y minus
y bar), summed and divided by n. Positive when x above its mean goes with y
above its mean. Negative when one is high while the other is low. Near zero
when they are unrelated. The product of deviations works because it is
positive only when both deviations have the same sign, so rows that lean the
same way push the sum up and rows that disagree pull it down (added
explanation). The covariance of a feature with itself is its variance: put x
in for y and the product becomes a square. Two features with a covariance
well away from zero are **correlated** (added definition): they tend to move
together, or in opposite directions, rather than independently. Five days at
a lemonade stand, with
daily highs 68, 74, 80, 86, 92 (mean 80) and cups sold 31, 40, 52, 57, 70
(mean 50), give var(temp) = 72.0, var(cups) = 182.8, and cov(temp, cups) =
114.0. Every product of deviations came out zero or positive. Hot days sell
more cups.

The lecture divides by n, the **population form**, for a clean picture. The
**sample form** divides by n minus 1, the idea being that you assume you
already took one out, and most of the mathematical support in Python,
including NumPy's `np.cov`, uses the sample form by default. This is the same
n versus n minus 1 gap from section 6. It does not change any of the geometry
that follows: every entry scales by the same factor, so the ellipse keeps its
shape and its natural axes keep their directions, and only their lengths
scale (added explanation). Those axes and lengths get their proper names,
eigenvectors and eigenvalues, in section 12.

The **covariance matrix** packs it together. For two features it is a 2 by 2
matrix, written C: var(x) and cov(x, y) on the first row, cov(x, y) and
var(y) on the second. Variances on the diagonal, the covariance in both
off-diagonal spots, so the matrix is symmetric. For the lemonade data that is
[[72.0, 114.0], [114.0, 182.8]]. It has a shape you can see, the **data
ellipse**, and a tilted, stretched ellipse means the two features are
correlated (see the figure from cell 6 of 6.1Covariance_eigenvectors.ipynb).

The obvious objection: a matrix of four numbers, how is that an ellipse? The
four numbers do not become an ellipse directly. They define a formula, and
the ellipse is the set of points where that formula equals a fixed constant
c. Take any point (x, y) measured from the center and compute [x y] times C
inverse times [x y] as a column. Setting that equal to c is the equation of
an ellipse centered at the origin, bigger c giving a bigger ellipse, and the
two ellipses the notebook draws are at 1 and 2 standard deviations. The
**matrix inverse** (added definition) is the matrix that undoes C, the way
dividing undoes multiplying, since dividing by a matrix is not defined. What
the quantity means in practice: a distance measured in units of the data's
own spread. One step out along a wide direction counts the same as a small
step along a narrow one, so the ellipse is "all the points equally far from
the center, according to the data." On the lemonade C, the point (12, 20)
scores 2.43 while the point (6, -5) scores 91.91, even though (12, 20) is much
farther from the center by a ruler. The first lies along the long axis of the
cloud, where the data normally reaches. The second runs against the grain,
hot day and fewer cups, so the data treats it as far out.

Each entry has a visible job. Diagonal entries set how far the ellipse
stretches along each axis. The off-diagonal covariance tilts it. Covariance
zero with equal variances gives a perfectly round data ellipse. Bigger var(x)
stretches sideways, bigger var(y) stretches upward, and a nonzero covariance
adds a tilt, meaning the two features are correlated (cell 9 of the notebook
draws all four cases). And the link to what comes next: the ellipse's axes,
which way it points, and their lengths, how far it stretches each way, are
exactly the eigenvectors and eigenvalues of C.

## 12. Eigenvectors: the peak without the sweep

Instead of trying a few directions by hand, sweep through all of them and
compute the **projected variance** (the variance of the dot products of every
centered point with a unit vector along that direction, added definition) at
each angle. The result is a smooth curve with a single clear peak, the
direction of maximum spread (see the two-panel figure from cell 11 of the
notebook). On the lemonade data the sweep reads 72.0 at 0 degrees (which is
just var(temp)), 182.8 at 90 degrees (just var(cups)), peaks at 254.1 near 58
degrees, and bottoms out at 0.65 near 148 degrees. At 0 and 90 you get the
diagonal entries of C back, because projecting onto an axis is the same as
reading that feature. Every other angle mixes the two, and the mixing is
where the covariance earns its keep.

The payoff: you do not have to sweep at all. The problem has a mathematical
solution, and that solution is an eigenvector. An **eigenvector** of C is a
special direction that C leaves pointing the same way. Multiplying it by C
only stretches it, never rotates it. Most directions get knocked askew by C.
These hold their heading. The **eigenvalue** (lambda) is the stretch factor,
and here lambda is the variance along that eigenvector. The equation is

```
C v = lambda v
```

read as "C acting on v gives back v, just scaled." The top eigenvector points
along the maximum variance, the peak of the sweep. The rest are perpendicular
to it. Because C is symmetric, the eigenvectors come out orthonormal,
perpendicular and unit length, the terms from section 9. How to compute them
is well beyond this course. The professor only needs you to know the property
and what it buys you, and in practice the computer does it in one call.

The lemonade numbers check every claim. `np.linalg.eigh(C)` on the population
matrix gives eigenvalues 254.15 and 0.65, exactly the peak and the floor of
the sweep. The top eigenvector (0.5305, 0.8477) sits at 57.96 degrees, the
angle where the sweep peaked. The two eigenvectors have dot product 0 and
length 1 each. And C times (0.5305, 0.8477) is (134.83, 215.43), which is
254.15 times the same vector: same heading, longer. C times (1, 0), which is
not an eigenvector, gives (72, 114), a vector pointing at 57.7 degrees. The
matrix swung the temperature axis almost onto the top eigenvector. That is
what "knocked askew" looks like. One practical warning (added): eigenvector
sign is arbitrary. (0.5305, 0.8477) and (-0.5305, -0.8477) are the same line,
different libraries can return either, so compare directions, not signs.

```python
import numpy as np

temp = np.array([68., 74., 80., 86., 92.]) ## daily high, F
cups = np.array([31., 40., 52., 57., 70.]) ## cups sold
pts = np.column_stack([temp - temp.mean(), cups - cups.mean()])
C = np.cov(pts.T, bias=True) ## population form, divides by n
evals, evecs = np.linalg.eigh(C) ## symmetric, ascending order
order = np.argsort(evals)[::-1] ## largest first
evals, evecs = evals[order], evecs[:, order]
print(C, evals, evecs[:, 0], (pts @ evecs).var(axis=0), sep='\n')
```

Output:

```
[[ 72.  114. ]
 [114.  182.8]]
[254.14841222   0.65158778]
[0.53052505 0.84766926]
[254.14841222   0.65158778]
```

The last line rotates the points into the **eigen-basis** (using the
eigenvectors as the new axes, added definition) with one matrix multiply,
centered points times the eigenvector matrix, and prints the variance of each
new coordinate: 254.15 and 0.65, the eigenvalues. Each new coordinate is the
dot product of the point with one eigenvector, exactly the projection from
section 9. If you use the eigenvectors as your new axes the cloud straightens
out, the tilt disappears, and in the new coordinates the two axes are
**decorrelated**, meaning the covariance between them is zero (added
definition). The tilt was the covariance, and the rotation removed it. The
new axes are the **principal components**: PC1 is the direction of maximum
variance, PC2 the next, and PC1 is a composition of the original dimensions,
not a direct one-to-one mapping to flipper length or body mass. On the
penguin data PC1 is the orange arrow and PC2 the blue arrow in cell 13 of the
notebook, and PC2 is tiny, and therefore not as important, because flipper
length and body mass are so strongly correlated that almost all the
variation lives along a single direction. Hold that thought.

One caution for later: everything was drawn on axes the professor chose,
flipper in millimeters and body mass in hundreds of grams. The covariance
matrix, and therefore the eigenvectors, depend on those units. Change the
scale of an axis and the best direction can tilt. Count the lemonade cups in
dozens instead of single cups and the top eigenvector swings from 58 degrees
to about 7.5 degrees, nearly the temperature axis, because temperature is now
numerically the biggest thing left. The data did not change. The units did.
That is why real data with very different units gets standardized before
PCA. The flag is planted: scaling changes the answer.

## 13. The procedure has a name: PCA, and why it pays

Look at the data, identify the eigenvectors of its covariance matrix, and use
them as new axes, projecting the data onto them. That procedure is
**Principal Component Analysis**, and the new axes are the principal
components: PC1 captures the most variance, PC2 the most of what is left and
is perpendicular to PC1, and PC3, PC4, and onward continue the pattern.
Nothing here is new math. It is a name for what section 12 already built.

Why bother? A confession from the professor: the penguin data have four
features, not two. Bill length, bill depth, flipper length, and body mass.
Two were picked to make the visualization interesting. To see four features
by eye you would look at every pair, which is six separate scatter plots, a
**pair plot** (a grid with one scatter plot per pair of features and each
feature's histogram on the diagonal). With 10 features it is 45 plots. With
20 it is nearly 200. You cannot hold that in your head, and this is the
problem PCA solves. The count comes from k times (k - 1) divided by 2 (added
explanation), each feature paired with every other one and the division by 2
stopping you from counting flipper-versus-mass and mass-versus-flipper twice.
The flipper-versus-mass slice from the earlier videos is one tile of that
grid, picked by hand. PCA's promise is to find the single best 2-D view built
from all four features at once. So the goal is not "pick the two most
important features." It is "build two new axes out of all four" (added
emphasis).

Why it works: often some of the features are redundant. Height and weight
normally have a lot of correlation, since someone super tall and extremely
skinny is rare. Sharper still, a person's left arm and right arm are almost
identical, so there is no need to keep both. **Redundant features** (added
definition) carry overlapping information because they are correlated, and
knowing one tells you most of the other. In the pair plot many panels are
clearly tilted: flipper length and body mass rise together, bill depth trends
opposite to the others. That redundancy is what lets PCA compress four
features into two, or even one, without losing much, exactly as PC2 was tiny
on the two-feature slice.

Redundancy as a number (added example). Sixty adults measured on height,
left arm length, right arm length, and weight, generated so that left arm
tracks height, right arm is a near copy of left arm, and weight tracks height
loosely. The correlations come out 0.96 for height and left arm, 0.95 for
height and right arm, 0.99 for the two arms, and 0.75 to 0.78 for weight with
each of the others. Standardize, take the eigenvalues of the covariance
matrix, and the cumulative share of variance is 90.1 percent with one
component, 98.4 with two, 99.8 with three, 100 with four. Four measurements,
and one axis already holds 90 percent of the variation. The other two axes
are measurement noise wearing a lab coat. That cumulative share is the exact
quantity the next section reads off the penguins.

The two payoffs. **Dimensionality reduction**: keep the top few principal
components and drop the rest, so four features become two coordinates you
can plot, a single 2-D picture instead of six. **Decorrelation**: the new
axes are perpendicular and uncorrelated, each principal component carrying
independent information, with no redundancy left. And the picture from
section 12 was already PCA in two dimensions. With four features it is the
same idea in 4-D: the covariance matrix is 4 by 4, it has four eigenvectors,
still orthonormal and still sorted by variance, and you cannot draw the
rotation but the math does not care (added remark).

## 14. PCA in practice: three steps in scikit-learn

Now for real, with `StandardScaler` from `sklearn.preprocessing` and `PCA`
from `sklearn.decomposition`, on the full Palmer Penguins data: 344 penguins,
four numeric features. Real data is rarely clean, a couple of penguins are
missing measurements, and PCA needs complete rows, so the first practical
step is `dropna()` on the four columns, which costs two rows.

PCA in practice is three steps:

1. **Standardize** the features. Put them all on the same scale.
2. **Find the components.** The principal directions, section 12's
   eigenvectors of the covariance matrix.
3. **Project** the data onto the top few components.

Step 1 is why section 12 planted its flag. Millimeters versus grams have
completely different scales, and without standardization the largest-numbered
feature dominates the variance for no good reason, apples to oranges. It is
the section 6 tool, applied. After `StandardScaler` the mean of every feature
is 0 and the standard deviation is 1.0 (the notebook checks with NumPy's
population standard deviation, so exactly 1.0, where `describe()` would show
the small section 6 gap), so "one unit" means the same thing on every axis,
one standard deviation.

Step 2, with `X_std` holding the standardized features, is `pca = PCA()` then
`pca.fit(X_std)`, asking for all four components
for now. There is no new mathematics hiding in scikit-learn. `pca.components_`
stores the eigenvectors of the standardized covariance matrix, the principal
directions in 4-D. `pca.explained_variance_` stores the matching eigenvalues,
the variance along each direction, the lambda from C v = lambda v. And
scikit-learn returns them already sorted largest first, so the PC1, PC2
ordering comes for free. The trailing underscore is the scikit-learn
convention for "this exists only after fit" (added). Divide each eigenvalue by
the total and you get `explained_variance_ratio_`, the **explained variance
ratio**, the fraction of the data's total spread that each principal component
captures, nothing more than the eigenvalues normalized to sum to 1. The
professor's penguin numbers: PC1 68.8 percent, PC2 19.3, PC3 9.1, PC4 2.7.
Cumulative, two components reach 88 percent, three reach 97, four reach 100
(see the bar chart from cell 11 of 8.1.pca_in_practice.ipynb, bars for each
component's share and a line for the running total).

So how many to keep? PC1 and PC2 together capture the great majority of the
variance, and PC3 and PC4 are the small, barely-varying directions, the same
situation as the tiny PC2 on the two-feature slice, now in four dimensions.
Keeping the top two buys a picture you can actually see while giving up very
little. Two is the answer here, with the more principled rules saved for
section 15.

Step 3 is `PCA(n_components=2)` and `fit_transform` on the standardized
data, which returns the **scores** (added definition), the coordinates of
each row in the new axes, 342 rows by 2 columns, each score being the dot
product of the standardized row with one eigenvector, the section 9
projection. Four measurements compressed into one honest 2-D picture, built
from the numbers alone (cell 14 of the notebook). This is not "we picked a
feature." The principal components are combinations of the different
features. And the plot has no labels. Nothing about species was used anywhere
in the recipe.

The same recipe on a different data set, in numbers (added example). Five
chemical features from the wine data set that ships with scikit-learn
(alcohol, malic acid, flavanoids, color intensity, proline, 178 wines),
standardized and fit: the eigenvalues come out 2.156, 1.650, 0.607, 0.337,
0.279, already sorted, and they sum to 5.028, the total variance of five
standardized features (each has variance 1 in the population form, so about
5, the small excess coming from the n minus 1 denominator that
`explained_variance_` shares with `np.cov`). That is the "total variance is
conserved" observation from section 12 in scikit-learn's own numbers. Project
onto two components and the correlation between the PC1 and PC2 score columns
is 0.0. That is the decorrelation payoff from section 13, visible in a
number.

What happens if you skip step 1 is worth seeing once. Those same five wine
features fed to PCA raw put proline (raw variance 99,166.7 in the sample
form pandas reports, against 0.7 to 5.4 for the others) in charge: PC1 explains 99.99
percent of the variance and its direction is the proline axis, full stop. PCA
did not find structure. It found the feature with the biggest numbers. That
is the apples-to-oranges problem, and the whole reason step 1 exists.

## 15. Choosing components and reading the axes

Video 4 left three loose ends. Two components were kept by eye, so what is
the actual rule? The new axes were a black box, so what do PC1 and PC2 mean?
And the species labels were never used, so what happens when they finally
are?

The rule first. The bar chart of percent variance per component has an
official name, a **scree plot**, variance explained against component number,
and the professor's memory hook is that it is "screen" without the n. Two
standard rules read it. The **elbow**: follow the curve until it flattens,
and the bend is where extra components stop buying you much, the **point of
diminishing returns** (added definition). The professor's reading of the
penguin plot is that the PC1 point is the base of the arm, the PC2 point is
the elbow, and after that the arm flattens out completely, so stop after PC2.
The **variance target**: decide in advance how much total variance you want
to keep, often 90 percent, and take as many components as it takes to clear
that line. On the penguins the 90 percent target is the dashed line in cell 4
of 9.1.Choosing_components_and_reading_the_results.ipynb, two components
reach 88 percent, very close to the target, and the professor keeps two. Both
rules point the same way here. They do not have to (added remark). On the
wine features the cumulative shares run 42.9, 75.7, 87.7, 94.4, 100.0
percent, the elbow reading points to three (the big drop lands between PC2
and PC3, and after PC3 the curve is flat), a 90 percent target says four, and
an 80 percent target says three, so the one you pick depends on whether you
need a 2-D picture, a compact model input, or a 90 percent guarantee. As one
line, with `cum` holding the cumulative percentages, `np.searchsorted(cum,
90) + 1` is the variance-target rule.

The reveal. The same PC1 versus PC2 projection, the one built with no
knowledge of species, recolored by species (cell 7 of the notebook). The
professor's reading: very good separation between the Gentoo penguins and the
other two kinds, and some separation between the remaining two, though not as
clean. The point is that PCA never saw the labels. It looked only at the
physical measurements. It did not classify the penguins. It reduced four
numbers to two, and the biological groups fell out on their own, because the
data were naturally separated and the reduction preserved that.

The meaning of the axes. Someone will ask: if PC1 separates the species that
well, tell me what PC1 is and I will use it as a filter. The answer is that
PC1 and PC2 are not one existing attribute picked out. They are **composite
attributes**, weighted combinations of the values of the different
attributes, and the weights are called **loadings**. They live in
`pca.components_`, row 0 holding PC1's four weights and row 1 holding PC2's.
A score on PC1 is each standardized feature times its PC1 loading, summed
(added). The professor's reading of the penguin loadings (cell 10 of the
notebook): PC1 weighs bill length, flipper length, and body mass positively,
and bill depth negatively. PC2 has high values for bill length and bill depth
together, zero for flipper length, and very little for body mass. The
notebook's gloss on the same bars: moving along PC1 means a penguin that is
longer and heavier with a shallower bill, and that single axis is what most
separates Gentoos from the other two species, while PC2 is a bill-shape axis
largely independent of overall body size. So, roughly, and the professor says
it is not 100 percent: PC1 is body size and bill shallowness, PC2 is bill
shape. That is what the black box was doing all along.

One wine scored by hand makes "weighted combination" literal (added
example). The first wine's standardized values are 1.519, -0.562, 1.035,
0.252, 1.013 in the feature order above, and the PC1 loadings are 0.588,
-0.105, 0.367, 0.371, 0.609. Multiply pairwise and sum: 1.519 times 0.588,
plus -0.562 times -0.105, plus 1.035 times 0.367, plus 0.252 times 0.371,
plus 1.013 times 0.609, which is 2.042. scikit-learn's `transform` reports
2.042 for that wine's PC1 score. Same number. That is all a score is.

How to read any loadings table (added guidance): big absolute value
means the feature matters a lot to that component, near zero means it barely
participates, same sign as another feature means they move together along
that component, opposite sign means the component contrasts them, and the
overall sign of a component is arbitrary, so compare signs within a component
and not across runs.

What we gained and what we gave up. Four dimensions with some correlation,
reduced to two axes that keep about 88 percent of the total variance, a
picture that can be seen and read, and axes with the correlation removed. The
cost, stated honestly: the axes are blends, composites. "PC1" is not
something you can put a ruler on, and the roughly 12 percent of variation
living in PC3 and PC4 is discarded. For visualization and preprocessing that
trade is usually a bargain. When every last feature matters, it may not be.
That trade-off is the judgment PCA always asks of you.

## 16. Simpson's paradox, and closing the arc

One last lesson, and the reason coloring by group matters so much. Simpson
the mathematician, not Homer. Set PCA aside and look at two raw features,
bill length and bill depth, for all penguins at once (cell 14 of the
notebook, left panel pooled with one trend line, right panel split by species
with a trend line each). A **trend line** (added definition) is the straight
line fit through a scatter plot, whose slope sign says whether y rises or
falls with x. Pooled, the relationship is negative. Split by
species, every single group shows a positive relationship. The pooled trend
is not just weaker. It points the wrong way. That is **Simpson's paradox**:
the behavior of a data set as a whole is completely different from the
behavior of the data set once you separate it into groups. An aggregate can
reverse the pattern present in every subgroup. The explanation, in general
and here, is a **hidden variable** (added definition, a variable that affects
the relationship you are looking at but is not on either axis) that you are
not considering. Here it is species, and the professor also points to how
many measurements each species has. Ninety employees across three
departments make the same shape with salaries: pooled, every extra year of
experience costs about 3,030 dollars, while within each department every
extra year pays 1,180 to 1,440 dollars more, because support staff have the
most years and the lowest pay band. The pooled line is answering "which
department has the veterans?" when you asked "does experience pay?" This
happens often in data, and responsible data science means always asking
whether a grouping is secretly driving, or hiding, the pattern you see.

Now the whole arc in one pass. Data arrives dirty, from human error and from
GPS drift, and the first job is to know what kind of attributes and data sets
you are holding (sections 1 and 2). A column gets summarized by its center
and its spread, and the spread is where the mean stops being enough (sections
3 to 5). Features on different scales get standardized so no column wins by
having big numbers, and strings get encoded without inventing an order that
is not there (sections 6 and 7). Then the geometry: a row is a point, a point
is an arrow, an arrow projects onto a direction, and the spread of the
projections scores the direction (sections 8 to 10). The covariance matrix
holds every spread and every tilt at once, and its eigenvectors are the
directions of maximum variance with the eigenvalues as the variance along
each (sections 11 and 12). That procedure is PCA, it pays because correlated
features are redundant, and in scikit-learn it is standardize, fit, project
(sections 13 and 14). Read the scree plot to decide how many components to
keep, read the loadings to learn what the axes mean, color by group before
you trust a pooled trend (sections 15 and 16). The professor's own summary:
PCA eliminates the correlation among the data and reduces the number of
dimensions, but the transformation is not "pick the best attributes." It is
a blend of the attributes that generates the final dimensions. From here PCA
is a tool, a preprocessing step that turns many correlated features into a
few informative ones before they go to the models ahead.
