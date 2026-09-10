## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## hierarchical clustering general use calculator, stdlib only
## general fork of hier_calc.py: any csv, any dimension, named points, ward linkage, height cut
## reuses helpers from kmeans_dbscan_calc_GenUse.py (same folder)
## no flags = prompts, -h = flags

import argparse, math, sys
import kmeans_dbscan_calc_GenUse as km
from kmeans_dbscan_calc_GenUse import hr, num, table, lst, dist, mean_pt, ask, pid, load

def members(c): return "{" + ", ".join(str(pid(i)) for i in sorted(c)) + "}"
def cent(c, pts): return mean_pt([pts[i] for i in c])

def link(a, b, pts, how): ## cluster to cluster distance
    match how:
        case "single": return min(dist(pts[i], pts[j]) for i in a for j in b)
        case "complete": return max(dist(pts[i], pts[j]) for i in a for j in b)
        case "average": return sum(dist(pts[i], pts[j]) for i in a for j in b) / (len(a) * len(b))
        case "ward": return math.sqrt(2 * len(a) * len(b) / (len(a) + len(b))) * dist(cent(a, pts), cent(b, pts))
        case _: return dist(cent(a, pts), cent(b, pts)) ## centroid

def diameter(c, pts): ## farthest pair inside a cluster
    if len(c) < 2: return 0.0, None
    return max(((dist(pts[i], pts[j]), (i, j)) for i in c for j in c if i < j), key=lambda t: (t[0], -t[1][0], -t[1][1]))

def show_matrix(names, clusters, pts, how):
    table([""] + names, [[names[i]] + [link(clusters[i], clusters[j], pts, how) if i != j else "-" for j in range(len(clusters))] for i in range(len(clusters))])

def agglomerative(pts, how, verbose=True):
    clusters, names = [{i} for i in range(len(pts))], [str(pid(i)) for i in range(len(pts))]
    merges, tree, leaf = [], {n: None for n in names}, {n: i for i, n in enumerate(names)} ## tree[name] = (left, right, height)
    while len(clusters) > 1:
        step = len(merges) + 1
        pairs = [(link(clusters[i], clusters[j], pts, how), i, j) for i in range(len(clusters)) for j in range(i + 1, len(clusters))]
        d, i, j = min(pairs) ## tie = lowest ids
        if verbose:
            hr(f"MERGE {step}: {how} distance between {len(clusters)} clusters")
            show_matrix(names, clusters, pts, how)
            print(f"nearest pair = {names[i]} {members(clusters[i])} and {names[j]} {members(clusters[j])} at {num(d)}")
        new, name = clusters[i] | clusters[j], f"M{step}"
        verbose and print(f"{name} = {members(new)}  centroid = {lst(cent(new, pts))}")
        merges.append((step, names[i], names[j], d, len(new)))
        tree[name] = (names[i], names[j], d)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [new]
        names = [n for k, n in enumerate(names) if k not in (i, j)] + [name]
    return merges, tree, leaf, names[0]

def dendrogram(tree, name, pre="", last=True):
    node = tree[name]
    print(pre + ("`- " if last else "|- ") + name + (f" (h = {num(node[2])})" if node else ""))
    if node:
        kids, pre = node[:2], pre + ("   " if last else "|  ")
        for k, child in enumerate(kids): dendrogram(tree, child, pre, k == len(kids) - 1)

def cut(tree, root, k=None, h=None): ## undo last k-1 merges, or all above h
    groups = [root]
    while not (k and len(groups) >= k):
        top = max((g for g in groups if tree[g] and (h is None or tree[g][2] > h)), key=lambda g: int(g[1:]), default=None)
        if top is None: break
        groups = [g for g in groups if g != top] + list(tree[top][:2])
    return groups

def leaves(tree, leaf, name): return {leaf[name]} if name in leaf else leaves(tree, leaf, tree[name][0]) | leaves(tree, leaf, tree[name][1])

def divisive(pts, k, verbose=True):
    clusters, splits = [set(range(len(pts)))], []
    target = k or len(pts)
    while len(clusters) < target:
        dia = [diameter(c, pts) for c in clusters]
        w = max(range(len(clusters)), key=lambda i: dia[i][0])
        if dia[w][1] is None: break
        c, (a, b) = clusters[w], dia[w][1]
        step = len(splits) + 1
        if verbose:
            hr(f"SPLIT {step}: widest cluster {members(c)}, diameter {num(dia[w][0])} between {pid(a)} and {pid(b)}")
            table(["#", f"d({pid(a)})", f"d({pid(b)})", "side"], [[pid(i), dist(pts[i], pts[a]), dist(pts[i], pts[b]), pid(a) if dist(pts[i], pts[a]) <= dist(pts[i], pts[b]) else pid(b)] for i in sorted(c)])
        left = {i for i in c if dist(pts[i], pts[a]) <= dist(pts[i], pts[b])}
        right = c - left
        verbose and print(f"-> {members(left)} centroid {lst(cent(left, pts))}  |  {members(right)} centroid {lst(cent(right, pts))}")
        splits.append((step, members(c), dia[w][0], members(left), members(right)))
        clusters = clusters[:w] + [left, right] + clusters[w + 1:]
    return clusters, splits

def report(pts, clusters, sil):
    lab = [0] * len(pts)
    for j, c in enumerate(clusters):
        for i in c: lab[i] = j
    cents = [cent(c, pts) for c in clusters]
    hr(f"RESULT: {len(clusters)} cluster(s)")
    for j, c in enumerate(clusters): print(f"C{j} {lst(cents[j])}: points {members(c)}")
    km.wcss(pts, cents, lab)
    if sil: km.silhouette(pts, lab)

def interactive(ap):
    argv = ["-d", ask("data: csv path, - for stdin, or rows x,y;x,y")]
    for flag, q in (("--id", "id column name or index (blank = numbered)"), ("--features", "feature columns (blank = every numeric column)")):
        v = ask(q)
        if v: argv += [flag, v]
    argv += ["--base", ask("point numbering starts at 0 or 1", str(km.BASE)), "--method", ask("method: agglomerative / divisive", "agglomerative")]
    if argv[-1] == "agglomerative":
        argv += ["--linkage", ask("linkage: centroid / single / complete / average / ward", "centroid")]
        h = ask("cut height (blank skips)")
        if h: argv += ["--height", h]
    k = ask("stop at k clusters (blank = full tree)")
    if k: argv += ["-k", k]
    if ask("standardize features? y/N", "n").lower() == "y": argv.append("--std")
    argv += ["--metric", ask("metric euclidean/manhattan/chebyshev", "euclidean")]
    if ask("silhouette? Y/n", "y").lower() == "n": argv.append("--no-sil")
    return ap.parse_args(argv)

def main():
    ap = argparse.ArgumentParser(description="hierarchical clustering on any data set, step by step")
    ap.add_argument("-d", "--data", help="csv file, - for stdin, or inline rows x,y;x,y (header optional)")
    ap.add_argument("--id", help="id column name or index, points print by name")
    ap.add_argument("--features", help="feature columns, default every numeric column")
    ap.add_argument("--method", choices=["agglomerative", "divisive"], default="agglomerative")
    ap.add_argument("--linkage", choices=["centroid", "single", "complete", "average", "ward"], default="centroid", help="centroid and ward assume euclidean")
    ap.add_argument("-k", type=int, help="clusters to report, default full tree")
    ap.add_argument("--height", type=float, help="agglomerative: cut the tree at this merge distance")
    ap.add_argument("--std", action="store_true", help="z-score features first")
    ap.add_argument("--metric", choices=["euclidean", "manhattan", "chebyshev"], default="euclidean")
    ap.add_argument("--no-sil", action="store_true", help="skip silhouette")
    ap.add_argument("--base", type=int, choices=[0, 1], default=km.BASE, help="point numbering start when no --id")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    if not a.data: sys.exit("need -d data")
    load(a.data, a.features, a.id)
    km.BASE, km.METRIC = a.base, a.metric
    if km.IDS and (len(set(km.IDS)) != len(km.IDS) or any(x[0] == "M" and x[1:].isdigit() for x in km.IDS)): sys.exit("ids must be unique and not look like merge names M1, M2")

    pts = [list(p) for p in km.DATA]
    if a.k and not 0 < a.k <= len(pts): sys.exit(f"k must be 1..{len(pts)}")
    hr("STEP 1: data set")
    table(["#"] + km.FEATURES, [[pid(i)] + list(p) for i, p in enumerate(pts)])
    if a.std:
        pts, mu, sd = km.zscore(pts)
        hr("STEP 2: standardize (z = (x - mean) / std, ddof=0)")
        table(["feature", "mean", "std"], [[f, m, s] for f, m, s in zip(km.FEATURES, mu, sd)])
        table(["#"] + km.FEATURES, [[pid(i)] + p for i, p in enumerate(pts)])

    if a.method == "divisive":
        clusters, splits = divisive(pts, a.k)
        hr("SPLIT TABLE (top down)")
        for st, c, d, l, r in splits: print(f"{st}: {c} at diameter {num(d)} -> {l} | {r}")
        report(pts, clusters, not a.no_sil)
        return
    merges, tree, leaf, root = agglomerative(pts, a.linkage)
    hr("MERGE TABLE (bottom up)")
    table(["step", "A", "B", "distance", "size"], merges)
    hr("DENDROGRAM (h = merge distance)")
    dendrogram(tree, root)
    if a.k or a.height is not None:
        groups = cut(tree, root, a.k, a.height)
        hr(f"CUT at k = {a.k}: stop after merge {len(pts) - a.k}" if a.k else f"CUT at height {num(a.height)}: undo every merge above it, {len(groups)} cluster(s)")
        report(pts, [leaves(tree, leaf, g) for g in groups], not a.no_sil)

if __name__ == "__main__": main()
