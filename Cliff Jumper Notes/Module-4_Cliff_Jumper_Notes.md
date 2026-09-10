# Module 4: Cliff Jumper Notes

*One continuous lesson stitched from the three Module 4 cliff notes of Applied
Predictive Modeling (CMPINF-2120): what association rule mining is for and the
diapers and beer story that made it famous, what market basket data looks like
once a receipt is stripped down to the items in the bag, how to count support
and decide what is frequent, the subset property that every association rule
algorithm rests on, how a frequent set turns into rules and how support and
confidence sort the useful rules from the rest, and the a priori algorithm that
finds the frequent sets without counting everything. Every worked example here
is the notes' own (a bagel cart parked outside the library) and every number
in it was run; the professor's own examples appear only as what the lecture
did. Anything added beyond the lecture is marked "(added ...)".*

---

## 1. The diapers and the beer

Module 3 left the supervised side of the course behind: clustering had no
answer column to learn from, only points and distances (added bridge from the
Module 3 notes). Module 4 stays on that side, but the data changes shape.
Instead of points with coordinates, the input is a list of the things people
bought together, and the question is which things travel together more than
you would expect.

The lecture opens with a story it calls an urban legend that happens to be
real. In the mid 1990s, when association rule mining became popular, analysts
looking at grocery store data found a strong correlation between diapers and
beer in the same shopping trip, usually between 5 and 7 p.m. on Fridays. The
explanation offered was new fathers: stuck at home with a baby over the
weekend, they bought the diapers and a few beers to drink at home instead of
going out. Whether or not the explanation is the whole truth, the pattern was
in the data, and the store could act on it. Put the chips between the diapers
and the beer. Offer a chips coupon with the beer. Nudge the customer who is
already in the aisle toward one more purchase.

**Market basket data** is the lecture's name for that list: the items people
buy together on one trip, one basket per shopping trip. Retailers collect it
aggressively.

**Association rule mining** is the task of discovering interesting and novel
patterns in that data, patterns that are not obvious but that a company can
use.

Two things in the story carry the whole module. First, the correlation is not
observable without the data. Nobody standing at the door on a Friday evening
sees carts piled with diapers and beer, because the pattern lives in a small
slice of the customers. Association rule mining looks for correlations
between items even when the slice is a low percentage of everyone who walks
in. Second, small slices are worth money at scale. The lecture's line is that
influencing 1% of a huge number of customers will cover the salary of the data
scientists who found the pattern. A 1% pattern in a small shop is a
curiosity; a 1% pattern across a supermarket chain is a budget line.

The story also shows the first practical complication. Real products come in
brands and sizes, and a pattern that exists at the level of "milk" can vanish
when the data says "organic 2% milk of one particular brand". People buy milk
and cereal together; they do not necessarily buy one specific milk with one
specific cereal often enough to register. So the analysis often generalizes
product types before counting, a point the third lecture returns to as a
product hierarchy.

One more thing the lecture wants noticed on the way out of the store: the
loyalty card. Loyalty cards were a novelty once and are mainstream now, and
the lecture's account of why is blunt. The card exists to gather data for
machine learning, association mining being the lecture's example. It links a
person or a household to an ID number so that every trip lands in one record,
and the store specials that once needed no card now require it, which pushes
everyone into using it and raises the quality of the data being mined. The
card is a data collection device first and a discount program second. That is
the lecture's closing point about how much these techniques already shape
daily life.

## 2. What a receipt becomes

A grocery checkout records a great deal: a customer ID, the date and time, the
list of things bought, quantities, prices, and how the bill was paid. Market
basket data keeps a small subset of that. What matters for this task is the
types of products in the bag, grouped by the trip that bought them. That
is the input to association rule mining: many baskets, each one a set of
product types, each with a transaction ID to tell it apart from the others.

Two facts about the shape of this data matter before any counting starts.
The number of items in one basket is much, much smaller than the number of
items the store sells. Nobody buys the whole store. And because the full
product names are too long to fit on a slide, the lecture replaces them with
letters: A, B, C, D, E, F, and more if needed, each standing for a real
product. Two housekeeping habits go with the letters. Items inside a basket
are listed in alphabetical order, which is not a rule of the math but makes
the tables faster for a human to scan, and the data is often laid out in a
columnar form with a binary flag under each item, which makes counting
easier still.

Here is the running case for the whole module. A bagel cart outside the
library keeps one line per customer, nothing but the items in the bag. Six
products, in letters the way the lecture does it:

    A = bagel, B = coffee, C = cream cheese, D = orange juice,
    E = muffin, F = tea

Ten receipts, each item listed once and alphabetically:

    T1   A, B, C
    T2   A, B, C, E
    T3   B, D
    T4   A, C
    T5   B, E
    T6   A, B, C, D
    T7   C, F
    T8   A, B, E
    T9   B, C
    T10  A, B, D

And the same ten receipts in the columnar form, a 1 where the item is in the
bag and a 0 where it is not:

    TID   A  B  C  D  E  F
    T1    1  1  1  0  0  0
    T2    1  1  1  0  1  0
    T3    0  1  0  1  0  0
    T4    1  0  1  0  0  0
    T5    0  1  0  0  1  0
    T6    1  1  1  1  0  0
    T7    0  0  1  0  0  1
    T8    1  1  0  0  1  0
    T9    0  1  1  0  0  0
    T10   1  1  0  1  0  0

Both layouts hold exactly the same information. The flag table earns its
keep in the next section, because a column sum is a count.

## 3. Items, itemsets, transactions

The lecture gives three definitions, and every later formula is built from
them.

An **item** is one product. A is an item, and A might be bread in real life.
On the cart, A is a bagel.

An **itemset** is a subset of all possible items, or equally a subset of the
items bought in one transaction. The lecture writes it as L, for instance
L = {A, B}. Order inside the braces does not matter: {A, B} and {B, A} are the
same itemset, and only one of the possible orderings is ever considered. The
alphabetical listing is a reading convenience, not part of the definition.

A **transaction** is an itemset together with a transaction ID. T6 on the cart
is the transaction whose itemset is {A, B, C, D}.

The **size** of an itemset is how many items it holds (added definition): {A}
has size 1, {A, B} size 2, {A, B, C} size 3. The lecture speaks of size 2
itemsets and size 3 itemsets throughout.

Because order is ignored, {A, B} and {B, A} are counted once, and that single
fact is what keeps the counting in section 12 from treating every ordering
of a set as its own set. (added note)

## 4. Counting support

The **support count** of an itemset is the number of transactions that contain
it. The lecture's phrase is how many times this combination of products showed
up in people's receipts. The **support percentage** is the support count
divided by the total number of transactions in the data.

The word "contain" is doing real work. A transaction contains an itemset when
every item of the set is in the basket, whatever else is there. The lecture
puts it as any superset doing the trick: to count {A} you do not care whether
A was bought alone, you count every receipt that has A in it at all. On the
cart there are ten receipts, so every percentage is the count over 10.

Size 1 first, and the flag table pays for itself here, because a column sum
is a support count:

    A  6   60%   T1, T2, T4, T6, T8, T10
    B  8   80%   T1, T2, T3, T5, T6, T8, T9, T10
    C  6   60%   T1, T2, T4, T6, T7, T9
    D  3   30%   T3, T6, T10
    E  3   30%   T2, T5, T8
    F  1   10%   T7

Coffee is in eight bags of ten. Tea is in one. Everything else sits between.

Size 2 next. To count {A, B}, walk the receipts and tick the ones that have
both letters: T1 yes, T2 yes, T3 no (B only), T4 no (A only), T5 no, T6 yes,
T7 no, T8 yes, T9 no, T10 yes. Five. The same walk for all fifteen pairs
(the number of pairs six items can make, which section 12 explains):

    AB  5   50%   T1, T2, T6, T8, T10
    AC  4   40%   T1, T2, T4, T6
    AD  2   20%   T6, T10
    AE  2   20%   T2, T8
    AF  0
    BC  4   40%   T1, T2, T6, T9
    BD  3   30%   T3, T6, T10
    BE  3   30%   T2, T5, T8
    BF  0
    CD  1   10%   T6
    CE  1   10%   T2
    CF  1   10%   T7
    DE  0
    DF  0
    EF  0

Some pairs never happen. Juice and muffin (DE) share no bag at all, and tea
appears only with cream cheese in T7. That is normal: most of the possible
pairs in any real store are empty, which is one of the reasons section 13
cares about how the counts are stored.

Size 3, showing only the sets with a nonzero count, since 13 of the 20
possible triples are zero:

    ABC  3   30%   T1, T2, T6
    ABD  2   20%   T6, T10
    ABE  2   20%   T2, T8
    ACD  1   T6
    ACE  1   T2
    BCD  1   T6
    BCE  1   T2

And size 4: ABCD appears once (T6), ABCE once (T2), nothing else.

Notice the direction the numbers move. As a set grows, its count can only
stay the same or fall, never rise, because a receipt that holds three
particular items also holds every pair among them, but a receipt that holds
the pair need not hold the third. That observation is the subset property in
disguise, and section 6 makes it official. (added explanation)

## 5. Frequent means past the line

**Minimum support**, written min_sup, is a threshold on the support count, or
on the support percentage if that is the preferred form, and an itemset is
**frequent** when its support reaches that threshold.

Where does the threshold come from? The lecture's answer is "magic": in this
class someone always hands it over. In real life it comes from people who ran
a similar analysis at the same company before, or from a bit of trial and
error across several thresholds. The data sets in the course are small and
the thresholds are set to match; real data sets are much, much bigger.

For the cart, the number handed over is min_sup = 3 transactions, which is
3 out of 10, or 30%. Read the tables in section 4 against that line:

    size 1:  A (6), B (8), C (6), D (3), E (3) are frequent.
             F (1) is not.
    size 2:  AB (5), AC (4), BC (4), BD (3), BE (3) are frequent.
             The other ten pairs are not.
    size 3:  ABC (3) is frequent. Nothing else is.
    size 4:  nothing.

Eleven frequent itemsets in all. Notice that D, E, BD, BE and ABC sit exactly
on the line at 3. Reaching min_sup counts; that is how the lecture applies
the threshold in its own tables.

The lecture works the same exercise on its own nine transaction table at two
thresholds, first 4 and then 2, and the results it reads off (A, B, C, AB,
BC and AC at 4; every single item, six pairs and the two triples ABC and
ABE at 2) are the ones to remember for a quiz. The point of running it twice
is that a threshold is a dial: lower it and more sets qualify, raise it and
fewer do, and nothing in the data tells you where to set it.

## 6. The subset property

Take ABC, the one frequent triple on the cart, count 3. Now cut it into
smaller pieces and look up each piece:

    AB  5    AC  4    BC  4
    A   6    B   8    C   6

Every piece is frequent, and every piece has a count at least as large as
the whole set's count of 3. That is not a coincidence of this cart. Every
receipt that holds A, B and C holds A and B, so the count of AB can never be
below the count of ABC. The same reasoning runs from size 2 down to size 1,
and from any size down to any smaller size.

That is the **subset property**: every subset of a frequent itemset is itself
frequent. The lecture's phrasing: if an itemset is frequent, any subset of it
must be frequent too; any way you cut it into smaller pieces, the pieces are
frequent as well.

The lecture's argument is the one above. If {A, B} is frequent, then every
time A and B show up together, both A and B show up, so each of them on its
own must appear at least as often as the pair. It cannot be that the pair
clears the line while A alone, or B alone, appears fewer times than the pair
does. The count of a set is a floor for the count of every one of its
subsets. (added last sentence)

Two warnings about direction, because the property is easy to read
backwards. It runs from a frequent set down to its subsets, not up. On the
cart, ABD has count 2 and is not frequent, yet two of its three pairs, AB (5)
and BD (3), are frequent. Frequent subsets do not make a frequent superset;
they only leave the door open. And the property does say something useful
about the upward direction, just the opposite thing: if a set is not
frequent, no superset of it (no set that contains it) can be, because
every superset's count is at most the set's count. Tea (F) has count 1, so
every set containing tea is dead before anyone counts it. (added
explanation)

The lecture's last sentence on the topic is the one that matters for the
rest of the module: almost all association rule algorithms are based on this
property. The a priori algorithm in section 10 is built on it, and the
downward warning above is exactly the lever it pulls.

## 7. From sets to rules

A frequent itemset says that some products travel together. An association
rule says it in a form a decision maker can read.

An **association rule** is a statement R of the form itemset 1 implies
itemset 2, written with an arrow. The itemset on the left hand side is a set
of products someone buys; the itemset on the right hand side is another set
they also buy. The two sides are **disjoint**, meaning they share no members:
{A, C} and {B} are disjoint, {A, C} and {A} are not. And the right hand side
is not empty.

The lecture notes that some simplified definitions restrict the right hand
side to a single item, and says this class will not use that restriction.
The meaning of a rule is that if a transaction includes itemset 1, it also
includes itemset 2. AB implies E and A implies BC are the lecture's two
examples of the form.

Getting from a frequent set to its rules is mechanical: generate every
possible rule. Take the set, find every way to split it into a left hand side
and a right hand side, and remember that order inside a side does not matter.
For a set of three items the lecture's procedure runs in two groups. First,
two items on the left and one on the right: pick each item in turn for the
right hand side and put everything else on the left. Then one on the left
and two on the right: pick each item in turn for the left. On the cart's ABC
that gives

    AB -> C     AC -> B     BC -> A
    A -> BC     B -> AC     C -> AB

six rules. Two splits are excluded. The right hand side cannot be empty, so
ABC -> nothing is out. The left hand side technically can be empty, and the
lecture mentions the empty left rule once to show that it makes no sense,
then ignores it for good. The count works out to 2 to the power of the set
size, minus those two, which for three items is 8 minus 2 = 6. (added
explanation)

## 8. Support and confidence of a rule

Six rules from one small set, and a real data set has thousands of frequent
sets. Which rules are worth keeping? The lecture gives two metrics.

The **support of a rule** is the support of the union of the two sides. For
the rule I implies J, it is the support of the itemset I together with J,
reported as a count or as a percentage exactly like itemset support.

The union of the two sides of a split is always the whole set you split. So
every rule generated from ABC has the same support, the support of ABC,
which is 3, or 30%. The lecture makes a point of this: no matter where you
cut, the two parts glue back into the original, so all the rules from one
frequent set share one support, and if one of them clears the minimum
support they all do. When the rules come from a frequent set, which is the
only place the algorithm takes them from, the support check is already
passed and does not need to be computed again.

What the support of a rule means: the fraction of transactions that involve
both sides together. Does this rule have enough samples behind it to merit a
look? If a pattern is very rare, it is probably not worth looking into even
if it is a pattern.

The **confidence of a rule** is the support of the whole rule, left side union
right side, divided by the support of the left hand side alone. The lecture
reads it as the probability that the right hand side appears given that the
left hand side already does.

Confidence does change from split to split, because the denominator is the
left hand side and the left hand side is what the split chooses. On the
cart the lecture's second threshold, the **minimum confidence** or
min_conf that a rule's confidence must reach, is set to 70%:

    two on the left, one on the right
    AB -> C    3/5 = 60.0%    fails
    AC -> B    3/4 = 75.0%    passes
    BC -> A    3/4 = 75.0%    passes
    one on the left, two on the right
    A -> BC    3/6 = 50.0%    fails
    B -> AC    3/8 = 37.5%    fails
    C -> AB    3/6 = 50.0%    fails

Read one of them aloud. AC -> B: of the four bags that held a bagel and
cream cheese, three also held a coffee, so three out of four, 75%. The
numerator is the same 3 every time; only the denominator moves. The rules
with a single item on the left fail because single items are common, so the
denominator is large. B -> AC has coffee on the left, coffee is in eight bags,
and only three of those eight also held the bagel and the cream cheese.

The empty left rule, for completeness: nothing implies ABC has confidence
sup(ABC) over the total number of transactions, 3/10 = 30%, because every
receipt contains the empty set. It says only that ABC is in 30% of bags,
which the support already said, so it is never kept.

A rule that reaches both thresholds is a **strong association rule**, and
finding strong rules is the stated goal of the a priori algorithm.

The two metrics divide the labor. Support asks whether a pattern is popular
enough to matter; confidence asks whether the conversion from the left side
to the right side is good. The lecture's own worked set, ABE at min_sup 2 and
min_conf 50%, passes four of its six rules and fails the two with a single
popular item on the left, the same shape as the cart's result, and those
figures are quiz material.

## 9. Strong rules on the cart

Rules come from every frequent set of size two or more, not only the biggest.
On the cart that means the five frequent pairs and the one frequent triple.
Each pair gives two rules; ABC gave six. With min_conf at 70%:

    A -> B     5/6 = 83.3%    passes
    B -> A     5/8 = 62.5%    fails
    A -> C     4/6 = 66.7%    fails
    C -> A     4/6 = 66.7%    fails
    B -> C     4/8 = 50.0%    fails
    C -> B     4/6 = 66.7%    fails
    B -> D     3/8 = 37.5%    fails
    D -> B     3/3 = 100%     passes
    B -> E     3/8 = 37.5%    fails
    E -> B     3/3 = 100%     passes

Together with the two survivors from ABC, the cart has five strong rules,
listed by confidence and then support:

    D -> B     100%     support 3
    E -> B     100%     support 3
    A -> B     83.3%    support 5
    AC -> B    75%      support 3
    BC -> A    75%      support 3

The story they tell is the kind of thing a cart owner could act on. Every
juice and every muffin left with a coffee. Five bagels in six did too. But
the reverse rules all fail: coffee is in eight bags of ten, so knowing
someone bought coffee says almost nothing about what else is in the bag. A
popular item makes a strong right hand side and a weak left hand side, which
is the same lesson the A -> BC and B -> AC failures taught in section 8.

## 10. The a priori algorithm

Everything so far counted every itemset there is. That works on ten receipts
and six products. It does not work on a supermarket. The a priori algorithm
finds the frequent itemsets without counting everything, and it does it by
leaning on the subset property. That is the **a priori algorithm**.

The idea in one sentence: start with the frequent single items, and to move
from size k to size k + 1, only build candidates whose every subset is
already known to be frequent. If any subset of a bigger set is not frequent,
that subset vetoes the bigger set and it is never counted. The lecture's
phrase for it is pruning out things that have no chance of being frequent.

The pseudocode, as the lecture lays it out:

    Init.  Take the minimum support and minimum confidence
           parameters. Generate all frequent itemsets of size 1.
    Loop over k = 2, 3, ...
      2a.  Generate the candidates of size k from the frequent
           itemsets of size k - 1: a join step that combines
           size k - 1 sets into size k sets, then a prune step
           that removes any candidate having even one subset
           that is not in the frequent list one level down.
      2b.  Count the surviving candidates against the data set.
      2c.  Keep the candidates whose count reaches min_sup.
           These are the frequent itemsets of size k.
      Repeat until the join and prune step produces no
           candidates at all.
    Then   Generate the association rules from the frequent
           itemsets by splitting each one every way, and keep
           the rules whose confidence reaches min_conf.

A **candidate itemset** is a set built in step 2a that might be frequent and
has earned a trip to the counting step (added definition). The lecture's
contrast is between candidates and the sets that are indeed frequent once
counted.

The candidate check in step 2a has a simple mechanical form the lecture
demonstrates. To find all the subsets of size k - 1 of a candidate of size
k, hide one item at a time. For a four item candidate, hide the first item
and look for the remaining three in the frequent list; then hide the second;
then the third; then the fourth. If every one of the four is found, the
candidate is good. If even one is missing, the candidate is out. The
lecture's words: any subset that is not frequent has ultimate veto power.
Only one is enough.

The lecture runs that check on a pretend frequent list of five size 3 sets
and two size 4 candidates, and finds one good and one vetoed. Its letters are
quiz material and section 11 uses the cart instead.

## 11. The walk on the cart

Min_sup stays at 3. Follow the algorithm step by step.

Init. Count every single item: A 6, B 8, C 6, D 3, E 3, F 1. The frequent
size 1 sets are L1 = {A, B, C, D, E}. Tea drops out.

k = 2, join. Pair up the members of L1. Five items make 10 pairs: AB, AC,
AD, AE, BC, BD, BE, CD, CE, DE. The five pairs that contain F (AF, BF, CF,
DF, EF) are never built, because F is not in L1. Six items could have made
15 pairs; the algorithm counts 10.

k = 2, prune. Hide one item from each candidate and look for the other in
L1. Every single item in every candidate is frequent, so nothing is vetoed
here; the veto already happened when F was left out of the join.

k = 2, count. AB 5, AC 4, AD 2, AE 2, BC 4, BD 3, BE 3, CD 1, CE 1, DE 0.

k = 2, keep. L2 = {AB, AC, BC, BD, BE}.

k = 3, join. Combine pairs from L2 into triples. The lecture keeps its sets
in lexicographic order, alphabetical here, and says that order improves the
efficiency of the subset lookups and is good algorithmically too. The
working rule is that two size 2 sets combine when they share their first
item, so that the union has exactly three items. (added explanation of the
join rule) Walking L2 in order:

    AB + AC -> ABC
    BC + BD -> BCD
    BC + BE -> BCE
    BD + BE -> BDE

Four candidates.

k = 3, prune. Hide one item at a time and look for the pair in L2:

    ABC:  hide A -> BC, in L2.  hide B -> AC, in L2.  hide C -> AB,
          in L2.  All three found.  Good candidate.
    BCD:  hide D -> BC, in L2.  hide C -> BD, in L2.  hide B -> CD,
          not in L2 (count 1).  Vetoed.
    BCE:  hide E -> BC, in L2.  hide C -> BE, in L2.  hide B -> CE,
          not in L2 (count 1).  Vetoed.
    BDE:  hide E -> BD, in L2.  hide D -> BE, in L2.  hide B -> DE,
          not in L2 (count 0).  Vetoed.

One missing subset was enough each time. Only ABC goes to the counting
step, and the three vetoed sets are never counted against the receipts.

k = 3, count. ABC = 3.

k = 3, keep. L3 = {ABC}.

k = 4, join. L3 has one member and nothing to combine it with, so no
candidates are produced. Stop.

The frequent itemsets found are A, B, C, D, E, AB, AC, BC, BD, BE and ABC,
the same eleven that section 5 read off the full tables. The difference is
in the work: section 4 counted all 15 pairs and looked at all 20 triples;
the algorithm counted 10 pairs and 1 triple. Then the rules step runs as
section 9 did, on every frequent set of size two or more, and keeps the same
five strong rules.

## 12. Why prune

The lecture answers "why do we bother pruning" with arithmetic. The number of
itemsets grows combinatorially with the number of products, and it gets out
of hand quickly.

**n choose k** is the number of ways to pick k items out of n when repetition
is not allowed and order does not matter. The formula is n factorial divided
by k factorial times (n minus k) factorial, where n factorial multiplies n by
every whole number below it down to 1, so 5 factorial is 5 times 4 times 3
times 2 times 1 = 120.

For the cart's six products:

    size 1:  6 choose 1 = 6
    size 2:  6 choose 2 = 15
    size 3:  6 choose 3 = 720 / (6 x 6) = 20

Six items, 41 possible sets of sizes one to three, before anyone counts
anything. Now let the subset property do its work. Tea is not frequent, so
every set containing tea is vetoed before counting, and only five items are
left to combine:

    size 2:  5 choose 2 = 10   (15 drops to 10)
    size 3:  5 choose 3 = 10   (20 drops to 10)

And the size 3 figure is an upper bound: after the size 2 counts came in,
the join and prune in section 11 left one triple to count, not ten. The
lecture's own version of this table uses seven products (7, 21 and 35 sets
of sizes one to three, with 7 choose 3 = 35 worked as the formula example)
and three infrequent items, and it watches 21 pairs shrink to 6 candidates.
Those are the figures a quiz would use. The cart shows the same mechanism
at a smaller scale: every infrequent single item takes a whole slice of the
combinations with it, and every infrequent pair takes a slice of the triples.

## 13. Counting in the real world

The lecture closes with a few implementation details that matter once the
data is large.

How to store the support counts. Strings are not used as keys; items are
mapped to integers first. Then there are three ways to hold the pair counts.
The naive method is a plain matrix with one cell for every pair i, j. The
triangular method does a little arithmetic on the index so that each
unordered pair is stored once instead of twice. The triples method uses a
hash table keyed by the pair, storing only the pairs that were actually seen.
On the cart the three sizes are 6 x 6 = 36 cells for the matrix, 15 cells for
the triangle (one per unordered pair, the 6 choose 2 of section 12), and 10
entries for the triples, since 10 of the 15 pairs have a nonzero count and
each entry holds two item numbers and a count, 30 numbers in all. Which
method wins depends on how many of the possible pairs ever occur: a store
with thousands of products and mostly empty pairs favors the triples. (added
worked comparison; the lecture names the three methods without sizing them)

A **hash table** is a structure that stores values under keys and finds a
value by computing where its key must be, rather than by searching (added
definition). Here the key is the pair and the value is its count.

Product hierarchies. Real data rarely has one fixed notion of a product.
Instead of one brand of low fat milk, the analysis can run at the level of
low fat milk, or milk, or drinks, and look for associations at any level
that turns out to be useful. This is the generalization point from the
diapers and beer lecture, now stated as a design choice rather than a
warning. Section 1's milk and cereal pattern exists at the "milk" level and
may not exist at the brand level; the hierarchy is how an analyst chooses
which level to count at.

Sequences and time. Two further complications the lecture names without
working: patterns over sequences of transactions rather than within one
basket, and patterns that change over time. The Friday evening in the
diapers story is already a hint of the second kind.

That closes the arc. A receipt becomes a set of letters. Sets get counted,
and a threshold someone hands over decides which are frequent. Frequent
sets obey the subset property, so an algorithm can build bigger candidates
only from smaller frequent ones and skip the rest. The frequent sets that
survive get split every way into rules, all sharing one support, and
confidence sorts the rules that predict from the rules that merely restate
what is popular. The five rules the cart produced are small, but the
mechanism that produced them is the one that found the diapers and the beer.
