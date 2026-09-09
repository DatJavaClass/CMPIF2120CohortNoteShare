# Module 3: Cliff Jumper Notes

*One continuous lesson stitched from the three Module 3 cliff notes of Applied
Predictive Modeling (CMPINF-2120): what clustering is and why it is on the
unsupervised side of the course, the two families of clustering algorithms,
the k-means loop step by step, where the first centroids come from, three
runs on one ten point set that show a point changing teams and a k that is
too small, the same loop in scikit-learn, and then how to score a clustering
and pick k: the silhouette index, the within cluster sum of squares, and the
elbow method. Every worked example here is the notes' own (a pizza shop
placing pickup lockers) and every number in it was run; the professor's own
examples appear only as what the lecture did.*

---

## 1. Groups without labels

Module 1 split the subject in two: supervised learning, where every record
carries an answer the model is trained to reproduce, and unsupervised
learning, where it does not (added bridge from the Module 1 notes). Module 3
sits on the unsupervised side, and its one task is clustering.

**Clustering** is the task of discovering groups and structures in the data
that are in some way similar to each other, so that the members of a cluster
are more alike among each other than they are to members of other clusters.
A **cluster** is such a group: points held together because they resemble
each other more than anything outside the group. Clustering is one type of
**unsupervised learning**, modeling on data with no labels and no known
answer to score against (added definition). Nobody tells the algorithm what
the right groups are. It has to find structure on its own, and, as the
second half of this module shows, it also has to be told how to judge
whether the structure it found is any good.

The lecture opens on a scatter of points and asks what the eye does with it.
With nothing to go on but the layout, the eye picks out three groups. That
is the intuition a clustering algorithm automates (added), and every
algorithm in this module is a different way of making "these points look
like they belong together" precise enough for a computer to act on.

## 2. Two families, and three questions

Clustering algorithms come in two families. **Hierarchical clustering**
builds clusters by repeatedly merging or splitting them, and it has two
subcategories. **Agglomerative** algorithms work bottom up: initially each
point is a cluster by itself, and the two nearest clusters are repeatedly
combined into one. **Divisive** algorithms work top down: start with
everything in one cluster and recursively split it. The more popular family
is **point assignment**: at any given time the algorithm maintains a set of
clusters, each point belongs to the cluster nearest to it, and that
membership can change over time until convergence. k-means, the algorithm
this module works with, is a point assignment algorithm. The difference
worth holding on to (added): a hierarchical algorithm never takes a merge
back, while point assignment keeps a working answer and lets points change
hands.

The key operation of hierarchical clustering, in the agglomerative case at
least, is to repeatedly combine the two nearest clusters. To do that, three
questions have to be answered first. How do you represent a cluster of more
than one point? How do you determine the nearness of clusters? When do you
stop combining clusters?

The first answer is that comparing clusters efficiently needs one
representative per cluster, and in the simpler case, using Euclidean
distance, that representative is the centroid. Two definitions carry the
rest of the module. **Euclidean distance** is ordinary straight line
distance: subtract the x values, subtract the y values, square both
differences, add, take the square root, sqrt((x1 - x2)^2 + (y1 - y2)^2)
(added definition). The **centroid** of a cluster is the average of all its
data points: average the x values, average the y values, and that pair is
the centroid. The professor's second way to think about it: the centroid is
the center of mass, as if the points were objects laid out on a plane. Later
he makes them balls on a tray, then beers on a tray, and the centroid is the
exact spot where you could hold the tray and it would not topple over. A
**point** here is one data record drawn as a dot at its two feature values,
x across and y up (added definition); every example in this module lives on
such a 2-D plane, and the professor notes that k-means works with numerical
data only, which is why a record has to be a point before k-means applies
(added).

The second answer: nearness between clusters is the distance between their
centroids, one number per pair however many points each cluster holds
(added restatement). The third: bottom up stops when all points have been
combined into one cluster, and top down stops when all clusters have been
split into single points. Both run to the extreme, so what you keep is the
sequence of merges, cut where the number of clusters suits you (added).

That is as far as the module takes hierarchical clustering. From here the
subject is the most popular algorithm, k-means.

## 3. The k-means loop

k-means is popular because it is very simple. You have to specify **k**, the
number of clusters to generate, before the algorithm runs. Given k, the
steps as the lecture gives them:

1. Pick k cluster centers at random. These are the initial cluster centers.
2. Assign every item, every point, to its nearest cluster center, for
   example using Euclidean distance.
3. Move each cluster center to the new mean of its assigned items.
4. Repeat the assignment and the recomputation until convergence.

The starting positions of the k centers are the **initial centroids**, also
called the **seed** (added definition). One full pass of assign then
recompute is an **iteration** (added definition). **Convergence** is the
point where the change in cluster assignment is less than a threshold,
ideally zero: nothing moves and nothing changes hands, so another pass
produces the same picture. You may also want a timeout parameter, so that
after a set number of iterations you stop, expecting little improvement
beyond that point.

The professor walks it on eight orange points on a 2-D plane with two random
initial centroids, green and blue. The assignment step has a shortcut: draw
the line connecting the two centroids and a dashed line through its exact
middle, perpendicular to it. Everything on the green side of that dashed
line is closer to the green centroid, everything on the other side is closer
to the blue one, so no distances have to be computed one at a time. That
dashed line is the **perpendicular bisector**, every position on it the same
distance from both centroids (added definition). The first pass puts three
points green and five blue. Each centroid then becomes the average of the
points carrying its color, moving off its random spot. On the reassignment
the bottom green point, which used to be blue, finds the green center much
closer now and changes. The professor calls this a team reassignment. The
recomputed centroids fall squarely in the middle of the four green points
and the four blue points, and the pass after that changes no color. No
changes means convergence, and the algorithm terminates.

## 4. Two pizza lockers

Here is the same loop on the notes' own numbers, and the story that every
worked example in this module uses. A pizza shop wants to place k pickup
lockers so every delivery address is close to one. An address is a point on
the town grid, x blocks east and y blocks north. A locker goes at the
centroid of the addresses it serves, and distances are Euclidean.

Eight addresses, numbered 1 to 8: (1,1), (2,1), (1,2), (2,3), (6,5), (7,5),
(6,6), (8,7). Two lockers, so k = 2. The initial centroids are picked from
the addresses themselves (section 5 says why): L1 is address 1 at (1,1) and
L2 is address 4 at (2,3).

Iteration 1, assignment. Address 4 is L2, so its distance is 0 and it goes to
L2. Address 5 at (6,5): to L1 the gaps are 5 east and 4 north, so
sqrt(25 + 16) = 6.403; to L2 the gaps are 4 and 2, so sqrt(16 + 4) = 4.472,
and L2 is closer. Working through the rest the same way, L1 gets addresses
1, 2 and 3, and L2 gets 4, 5, 6, 7 and 8.

Iteration 1, new centroids. L1 is the average of (1,1), (2,1) and (1,2): x is
(1 + 2 + 1) / 3 and y is (1 + 1 + 2) / 3, so L1 = (1.333, 1.333). L2 is the
average of (2,3), (6,5), (7,5), (6,6) and (8,7): x is 29 / 5 and y is
26 / 5, so L2 = (5.8, 5.2).

Iteration 2, assignment. Address 4 at (2,3) changes teams. Its distance to
the new L1 is sqrt(0.444 + 2.778) = 1.795, and to the new L2 it is
sqrt(14.44 + 4.84) = 4.391. A moment ago address 4 was a centroid; now it
belongs to the other cluster, because L2 was dragged northeast by the five
points around it (added explanation). L1 now has 1, 2, 3 and 4, and L2 has
5, 6, 7 and 8.

Iteration 2, new centroids. L1 = (6 / 4, 7 / 4) = (1.5, 1.75). L2 =
(27 / 4, 23 / 4) = (6.75, 5.75).

Iteration 3. Every address is still nearest the same locker and neither
centroid moves. That is convergence, and the run stops with lockers at
(1.5, 1.75) and (6.75, 5.75). Address 4 switching sides on the second pass is
the whole mechanism in miniature (added): a point between two groups follows
whichever center ends up nearer once the centers move to where the mass is.
One more number to keep for section 10: adding up how far each address sits
from its locker, in squared distance, gives 9.25 at convergence, 3.75 for
the west locker's four addresses and 5.5 for the east one's.

## 5. Where the first centroids come from

The professor's first summary point is that the assignment of the initial
centroids is very crucial. Had the initial points been picked elsewhere, the
run would have had a different outcome, or at least different steps to the
same outcome. He gives three approaches under three labels of his own.

**Expired**: pick completely random points anywhere in the point space, as
the eight point walk did purely as illustration. Not really used, and not
used anymore in this course. The chances of a good initial assignment are
slim, and it requires knowing a lot about the domain and the point space.

**Wired**: pick k random points from the existing set of points. Out of ten
points, if k is two, you pick two of those ten. This is what is typically
used and is the standard k-means approach. It guarantees you do not pick
points nowhere near any existing points, since starting from points that
exist leaves no centroid stranded in an empty region of the space (added),
but it is still sensitive to which points get chosen.

**Upgraded**: k-means++. Pick one random point from the set of points, then
choose the remaining centroids, number 2 through number k, from a
probability distribution that favors points farther away from those already
picked. For k = 2 the second is essentially the point at the highest
distance from the first, and three and four go the same way; from there, in
the professor's words, it gets a little bit more mathy. This explores the
space more and is shown to be a little bit better. **k-means++** is the name
of that scheme: the loop after it is ordinary k-means, only the choice of
starting centers changes (added definition), and it is scikit-learn's
default, which section 8 leans on (added note).

The professor closes the first lecture with the balance sheet. Advantages:
k-means is simple, it is very easy to understand, and it assigns items to
clusters automatically. Disadvantages: you have to pick the number of
clusters beforehand, and that k value is crucial; and all items are forced
into clusters, whether they would like that or not. As a result k-means is
too sensitive to outliers. An **outlier**, a point sitting far from the rest
of the data (added definition), still ends up in some cluster, and since a
centroid is an average, one far point pulls its centroid away from the group
it represents, which may skew your results. The next two sections show both
disadvantages happening on the map.

## 6. Ten addresses, three lockers, and a point that changes teams

The second lecture is all examples. The professor takes one set of 10 points
generated from a script and runs k-means three times: k = 3 from one seed,
k = 3 again from another, then k = 2. From this point on every run picks its
centroids out of the existing points, because good centroids are very
difficult to pick with no idea of where the data are, and picking from the
points you have guarantees they land inside the space the points occupy.

His first run seeds k = 3 at points 2, 4 and 9. Point 2 gets 1 and 5, point
9 gets 8 and 7, point 4 gets 3, 6 and 10, the clusters are colored red,
green and purple, and each centroid moves to its cluster's average (the
purple one leaves point 9 for the average of 8, 7 and 9). The second pass
changes no assignment, so the new centroids match the old: convergence, and
that is where we stop. His second run seeds at points 1, 6 and 8, a choice
made to produce a little movement between clusters. Point 10 comes out
closer to 8 than to 6 at first, then changes affiliation from purple to
green on the second assignment, because the green centroid moved down a
little and the purple centroid moved up only slightly. One more pass changes
nothing.

Now the same two stories on the notes' own map. Ten delivery addresses:

- 1 (1,1), 2 (2,2), 3 (1,3)
- 4 (8,1), 5 (9,2), 6 (8,3)
- 7 (4,8), 8 (5,9), 9 (6,8)
- 10 (5,5)

By eye there is a southwest group (1, 2, 3), a southeast group (4, 5, 6), a
north group (7, 8, 9), and address 10 alone in the middle of the map.

**Three lockers, seeded at addresses 1, 4 and 7** (red, green, purple).
Iteration 1, assignment: each address goes to the nearest of (1,1), (8,1)
and (4,8). Red takes 1, 2, 3; green takes 4, 5, 6; purple takes 7, 8, 9 and
10. Address 10 at (5,5) is the one worth checking: to (1,1) it is
sqrt(16 + 16) = 5.657, to (8,1) it is sqrt(9 + 16) = 5.0, to (4,8) it is
sqrt(1 + 9) = 3.162. Purple, so the lonely middle address is served by the
north locker. Iteration 1, recompute: red is ((1 + 2 + 1) / 3,
(1 + 2 + 3) / 3) = (1.333, 2.0), green is ((8 + 9 + 8) / 3, (1 + 2 + 3) / 3)
= (8.333, 2.0), purple is ((4 + 5 + 6 + 5) / 4, (8 + 9 + 8 + 5) / 4) =
(5.0, 7.5). Iteration 2: every address keeps its cluster, so the recomputed
centroids are identical. Converged on the second pass, exactly the shape of
the professor's first run. Address 10 has pulled the purple locker south of
the north group, since nothing in k-means lets a point opt out (added).

**Three lockers from a worse seed, addresses 10, 4 and 8.** The awkward case:
the lonely middle address is itself a seed. Iteration 1, assignment: red,
from address 10 at (5,5), takes 1, 2, 3 and itself; green, from address 4,
takes 4, 5, 6; purple, from address 8, takes 7, 8, 9. Check address 2 at
(2,2): to (5,5) it is sqrt(9 + 9) = 4.243, to (8,1) it is sqrt(36 + 1) =
6.083, to (5,9) it is sqrt(9 + 49) = 7.616, so red, and the whole southwest
group hangs off the middle address. Iteration 1, recompute: red is
((1 + 2 + 1 + 5) / 4, (1 + 2 + 3 + 5) / 4) = (2.25, 2.75), green is
(8.333, 2.0), purple is ((4 + 5 + 6) / 3, (8 + 9 + 8) / 3) = (5.0, 8.333).
Iteration 2, assignment: address 10 changes teams. To the red centroid
(2.25, 2.75) it is sqrt(7.5625 + 5.0625) = 3.553; to the purple centroid
(5.0, 8.333) it is 3.333. Purple wins by about two tenths of a block. The
point did not move; the centroid chasing it did. Iteration 2, recompute: red
back to (1.333, 2.0), green still (8.333, 2.0), purple (5.0, 7.5). Iteration
3 changes nothing. Identical lockers to the first seed, one extra iteration
to reach them. A different seed can change the path without changing the
destination, and it does not always end this politely (added), which is why
section 5's choices matter.

## 7. Two lockers for three neighborhoods

The professor's third run keeps the 10 points and drops to k = 2, seeded at
points 2 and 9, forcing data that looks like three clusters to the naked eye
into two. Point 2 initially takes 1, 5 and 10, point 9 takes 8, 7, 3, 4 and 6
but not 10. On the recompute the green center moves up significantly to
average the points it now holds, and the red center moves up and to the
right to counterbalance 1, 2 and 5 on one side against 10 on the other. Point
10 then changes teams from red to green, the red centroid moves inside the
triangle formed by points 1, 2 and 5, the green centroid moves a little to
the left because 10 is now part of that team, and the pass after that brings
no changes: convergence.

On the notes' map, k = 2 seeded at addresses 1 and 4. Iteration 1,
assignment: red, from (1,1), takes 1, 2, 3 and 7; green, from (8,1), takes 4,
5, 6, 8, 9 and 10. The north group gets split. Address 7 at (4,8) is 7.616
from (1,1) and 8.062 from (8,1), so red; address 8 at (5,9) is 8.944 from
(1,1) and 8.544 from (8,1), so green. Two addresses one block apart on each
axis land on opposite teams, the only comparison being distance to the two
seeds. Iteration 1, recompute: red is ((1 + 2 + 1 + 4) / 4,
(1 + 2 + 3 + 8) / 4) = (2.0, 3.5), green is ((8 + 9 + 8 + 5 + 6 + 5) / 6,
(1 + 2 + 3 + 9 + 8 + 5) / 6) = (6.833, 4.667). Iteration 2, assignment:
address 7 changes teams to green, 4.924 to the red centroid against 4.374 to
the green one. Red keeps 1, 2, 3 and green holds the rest. Iteration 2,
recompute: red (1.333, 2.0), green (45 / 7, 36 / 7) = (6.429, 5.143).
Iteration 3: no changes, converged.

What k = 2 did to the map: the north group and the southeast group were
welded into one cluster whose locker at (6.429, 5.143) sits in empty space
between them. Forcing three natural groups into two does not split the
difference. It parks a center where nobody lives (added). The spread of the
addresses around their lockers, the same squared distance total that came to
9.25 in section 4, rises from 16.333 with three lockers to 87.238 with two.
Section 10 names that number, and section 11 uses it to pick k.

## 8. The loop in scikit-learn

scikit-learn does the whole loop in one call (added example; the lectures
show no code, the notes supply it):

```python
import numpy as np
from sklearn.cluster import KMeans
arr_pts = np.array([[1,1],[2,2],[1,3],[8,1],[9,2],[8,3],
                    [4,8],[5,9],[6,8],[5,5]])
km = KMeans(n_clusters=3, n_init=10,
            random_state=0).fit(arr_pts) ## three lockers
print(km.labels_) ## cluster id per address
print(km.cluster_centers_.round(3)) ## locker sites
```

It prints labels `[2 2 2 0 0 0 1 1 1 1]` and centers (8.333, 2.0),
(5.0, 7.5), (1.333, 2.0): the three lockers of section 6, in label order. The
ids are only names, so which group is 0 carries no meaning. Read the labels
as who shares a locker: 1, 2, 3, then 4, 5, 6, then 7, 8, 9 and 10. Set
`n_clusters=2` and the centers come back as (6.429, 5.143) and (1.333, 2.0),
section 7's answer.

Two arguments deserve a word (added explanation). `n_init=10` runs the
algorithm ten times from ten different seeds and keeps the best result,
which is how the library defends against the worse seed of section 6.
`random_state=0` fixes those seeds so the call always prints the same thing.
And `init` was not set at all, because the default is k-means++, the
professor's upgraded approach. A fitted `KMeans` also carries one more
attribute, `inertia_`, which is the number section 10 is about to define.

## 9. Scoring a clustering: the silhouette index

Everything so far produces clusters. Nothing so far says whether they are
any good. k-means returns k clusters for whatever k you hand it, and nothing
inside the loop reports whether those groups mean anything (added). A score
is what lets you compare runs and choose k, and the third lecture supplies
two. There are multiple techniques for evaluating clustering quality; the
lesson works with the silhouette index and the within cluster sum of
squares.

The **silhouette index**, also called the silhouette coefficient, is a widely
used metric for evaluating the quality of clustering algorithms. It measures
how similar a data point is to its own cluster, which is cluster cohesion,
compared to the other clusters, which implies cluster separation, or how
well the clusters are separated from each other. **Cohesion** is how tightly
the members of one cluster sit together, and tight is good; **separation**
is how far one cluster sits from the others, and far is good (added
definitions).

It is computed one point at a time, from two quantities. a(i) is the average
distance from point i to all the other points in the same cluster, the
**intracluster distance**: how far you are from your own crowd. b(i) is the
minimum average distance from point i to the points in the nearest
neighboring cluster, the **intercluster distance**: take each other cluster
in turn, average the distance from i to every point in it, and keep the
smallest of those averages, how far you are from the nearest other crowd.
The silhouette score for point i is

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

In words (added explanation): the top is how much further the nearest other
cluster is than your own. Positive means your own cluster is the closer of
the two, which is what you want; negative means the neighbors are closer
than your own cluster mates. Dividing by the larger of the two distances
pins the answer between -1 and 1 whatever the grid units are. So the range
runs from -1 to +1. Plus one means the point is well clustered, far from the
other clusters. Zero means the point is on the boundary of two clusters, the
case where a(i) equals b(i) (added). Minus one means the point may be in the
wrong cluster. The overall silhouette score is the average of s(i) across
all data points, one number for the clustering, higher better. Being an
average, it can hide a few badly placed points, so read the per point scores
too (added).

By hand, on six delivery addresses: P1 (1,1), P2 (2,1), P3 (1,2), P4 (7,7),
P5 (8,7), P6 (4,4). The clustering under test puts cluster A = P1, P2, P3
and cluster B = P4, P5, P6. The distances needed, rounded to three decimals:
P1 to P2 and to P3 are both 1, P2 to P3 is 1.414, P4 to P5 is 1, P6 is 4.243
from P1, 3.606 from P2, 3.606 from P3, 4.243 from P4 and 5 from P5, and
across the two groups P1 is 8.485 and 9.220 from P4 and P5, P2 is 7.810 and
8.485, P3 is 7.810 and 8.602. With two clusters the nearest neighboring
cluster is always the other one, so b(i) is the average distance to all
three points on the far side.

- P1: a = (1 + 1) / 2 = 1.000, b = (8.485 + 9.220 + 4.243) / 3 = 7.316,
  s = (7.316 - 1.000) / 7.316 = 0.863.
- P2: a = (1 + 1.414) / 2 = 1.207, b = (7.810 + 8.485 + 3.606) / 3 = 6.634,
  s = 0.818.
- P3: a = 1.207, b = (7.810 + 8.602 + 3.606) / 3 = 6.673, s = 0.819.
- P4: a = (1 + 4.243) / 2 = 2.621, b = (8.485 + 7.810 + 7.810) / 3 = 8.035,
  s = 0.674.
- P5: a = (1 + 5) / 2 = 3.000, b = (9.220 + 8.485 + 8.602) / 3 = 8.769,
  s = 0.658.
- P6: a = (4.243 + 5) / 2 = 4.621, b = (4.243 + 3.606 + 3.606) / 3 = 3.818,
  s = (3.818 - 4.621) / 4.621 = -0.174.

The overall score is the average of the six, 0.610. Five points score high,
tight with their own cluster mates and far from the other side. P6 at (4,4)
is negative, the case the lecture calls a point that may be in the wrong
cluster: on average it is closer to cluster A than to its own cluster B.
Move P6 into cluster A and score again, and the per point values become
0.765, 0.754, 0.755, 0.859, 0.872, 0.174 with an overall score of 0.697. P6
is positive now but still small, because it genuinely sits between the
groups. The metric found the better arrangement and also said P6 will not be
comfortable anywhere. In Python it is two calls, `silhouette_samples` for
the per point values and `silhouette_score` for the mean, both in
`sklearn.metrics`, and on these six addresses they reproduce the numbers
above for both clusterings (added).

## 10. Within cluster sum of squares

The second metric is the one k-means itself is built around (added). The
**within cluster sum of squares (WCSS)** is the sum of the squared distances
between each point and the centroid of the cluster it belongs to. The
formula is a double sum, one sum over the clusters and one over the points
inside each cluster:

    WCSS = sum over clusters, sum over the points in each cluster,
           of (point minus its centroid) squared

In words: for every cluster, and for every point within that cluster, take
the difference between the data point and its centroid, square it, and add
all of those up into one total. The squaring removes the sign and makes a
far point count for much more than a near one (added). The intuition is that
a lower WCSS means the points are closer to their cluster centroid, so the
clusters are more compact. Lower is better.

Section 4 already computed one. Take the eight addresses at convergence:
(1,1), (2,1), (1,2), (2,3) around a locker at (1.5, 1.75), and (6,5), (7,5),
(6,6), (8,7) around a locker at (6.75, 5.75). The double sum says do one
cluster, then the other, then add. Squared distances to (1.5, 1.75): (1,1)
gives 0.25 + 0.5625 = 0.8125, (2,1) gives 0.8125, (1,2) gives 0.25 + 0.0625
= 0.3125, (2,3) gives 0.25 + 1.5625 = 1.8125, cluster total 3.75. Squared
distances to (6.75, 5.75): (6,5) gives 0.5625 + 0.5625 = 1.125, (7,5) gives
0.0625 + 0.5625 = 0.625, (6,6) gives 0.625, (8,7) gives 1.5625 + 1.5625 =
3.125, cluster total 5.5. WCSS = 3.75 + 5.5 = 9.25, the number scikit-learn
reports as `inertia_` for the same fit (added), and **inertia** is simply
scikit-learn's name for WCSS (added definition). No square roots appear: the
metric stays in squared distance, so only the comparison between totals
carries meaning (added).

Now the catch. As the number of clusters increases, WCSS will typically
decrease. More lockers means everyone is nearer to a locker, and at k equal
to the number of points WCSS reaches exactly zero, since every point is then
its own centroid: a perfect score that tells you nothing (added). However,
with too many clusters you can have an overfit. **Overfit** here is a k so
large that the clusters are the individual points back again, a model fitted
so closely to the data in front of it that it captures the noise along with
the structure (added definition). So WCSS cannot be minimized on its own to
pick k. It has to be read against how fast it is still falling (added), and
that reading is the elbow method.

## 11. The elbow, and closing the arc

The elbow method determines the best k for k-means. The procedure: identify
a quality metric, here the within cluster sum of squares; perform the
clustering for different values of k, that is, repeat k-means for many k
values; plot the quality metric on the y axis and the k value on the x axis;
and visually identify the **point of diminishing returns** for increasing k,
the k past which each additional cluster buys a much smaller improvement
than the clusters before it did (added definition). After some point, k can
be increased without adding much to the quality. The name comes from the
shape: the curve drops steeply, then flattens, and the corner between the
two looks like the elbow of an arm being extended a little. That bend is the
k you pick.

The professor runs a small dataset of 10 data points through k-means for
every k from 1 to 10. k = 1 is trivial, everything is a single cluster. k = 2
gives two clusters, one holding all the top left points and one holding all
the bottom right points. k = 3 splits the points into three, and visual
inspection says that is reasonable; k = 4, 5 and 6 split things a little
further each time and still seem reasonable. Past 7 you can tell that points
which were close to each other are being split apart, the same at 8 and 9,
and at k = 10 every point is its own cluster. The plot puts WCSS on the y
axis, lower better, against k on the x axis. From 1 to 2 the drop is really
big, and from there it slows to negligible. The lecture reads the elbow at
k = 2: beyond it you get some improvement, but you are effectively at the
point of diminishing returns, so it is not really worth it, and k = 2 is the
best choice for that data.

The same procedure on the notes' ten addresses from section 6, which are not
the professor's ten points, so the elbow lands somewhere else. Run k-means
for k = 1 to 10 and record the WCSS:

- k = 1: 162.500
- k = 2: 87.238
- k = 3: 16.333
- k = 4: 8.000
- k = 5: 6.333
- k = 6: 4.667
- k = 7: 3.000
- k = 8: 2.000
- k = 9: 1.000
- k = 10: 0.000

k = 1 checks by hand: one locker at the grand mean of all ten addresses,
(4.9, 4.2), with every address measured to it, and those squared distances
add to 162.5. k = 2 and k = 3 are section 7's 87.238 and section 6's 16.333.
At k = 10 every address is its own locker and every distance is zero. Now
read the drops. From 1 to 2 removes 75.262. From 2 to 3 removes another
70.905. From 3 to 4 removes only 8.333, and every step after that buys a few
units at most. The elbow is at k = 3. The second and third lockers pay for
themselves; the fourth does not. All the fourth does is make the address at
(5,5) a cluster of one, the overfit from section 10 arriving on schedule.

The silhouette index agrees, which is the check worth running (added). On
the same runs: k = 2 gives 0.446, k = 3 gives 0.678, k = 4 gives 0.545, then
0.376, 0.249, 0.088, 0.059 and 0.029 for k = 5 through 9. It peaks at k = 3,
the answer the elbow gave. The silhouette is undefined at k = 1, where there
is no other cluster and so no b, and at k equal to the number of points,
where nobody else is in your cluster and so no a, which is why it is
computed only for k from 2 up to n minus 1 (added explanation). The code,
with a fitted `KMeans` handing back the WCSS directly:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
arr_pts = np.array([[1,1],[2,2],[1,3],[8,1],[9,2],
                    [8,3],[4,8],[5,9],[6,8],[5,5]])
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10,
                random_state=0).fit(arr_pts)
    s = float("nan") ## undefined at both ends
    if 1 < k < 10:
        s = silhouette_score(arr_pts, km.labels_)
    print(k, round(km.inertia_, 3), round(s, 3))
```

Now the whole arc in one pass. Clustering finds groups in data that carries
no labels, and a cluster is represented by its centroid, its center of mass
(sections 1 and 2). k-means is the point assignment loop: choose k, seed k
centers, assign every point to its nearest center, move each center to the
mean of its points, repeat until nothing changes (sections 3 and 4). The
seed matters, so it comes from the data, or better from k-means++, and the
two costs of the method are that you supply k and that every point is forced
into some cluster (section 5). Three runs on one map show a point changing
teams because the centroids moved under it, a worse seed taking a longer
road to the same answer, and a k that is too small parking a locker where
nobody lives (sections 6 and 7); scikit-learn runs the whole loop in one
call and hands back the labels, the centers, and the inertia (section 8).
Then the judging: the silhouette weighs each point's own crowd against the
nearest other crowd and flags the misfits with a negative score (section 9),
the within cluster sum of squares measures compactness and typically falls
as k grows (section 10), and the elbow reads the rate of that fall to pick
the k where extra clusters stop paying for themselves, with the silhouette
as the second opinion (section 11). The professor's own summary of the
method stands as the summary of the module: k-means is simple, very
understandable, and assigns items to clusters automatically; its price is
that you must pick k beforehand and that all items are forced into clusters,
which can make it too sensitive to outliers. The metrics in the last lecture
are how you pay that price with your eyes open.
