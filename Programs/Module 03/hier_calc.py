## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## hierarchical clustering bare metal calculator, stdlib only
## reuses helpers from kmeans_dbscan_calc.py (same folder)
## no flags = prompts, -h = flags

import argparse, sys
import kmeans_dbscan_calc as km
from kmeans_dbscan_calc import hr, num, table, lst, dist, mean_pt, ask

## edit the data set here, one row per point
FEATURES = ["x", "y"]
DATA = [(1, 1), (2, 2), (2, 7), (3, 8), (1, 3), \
        (4, 7), (8, 2), (7, 3), (9, 3), (6, 5)]

def pid(i): return i + km.BASE
def members(c): return "{" + ", ".join(str(pid(i)) for i in sorted(c)) + "}"

def link(a, b, pts, how): ## cluster to cluster distance
    match how:
        case "centroid": return dist(mean_pt([pts[i] for i in a]), mean_pt([pts[i] for i in b]))
        case "single": return min(dist(pts[i], pts[j]) for i in a for j in b)
        case "complete": return max(dist(pts[i], pts[j]) for i in a for j in b)
        case _: return sum(dist(pts[i], pts[j]) for i in a for j in b) / (len(a) * len(b)) ## average

def diameter(c, pts): ## farthest pair inside a cluster
    if len(c) < 2: return 0.0, None
    return max(((dist(pts[i], pts[j]), (i, j)) for i in c for j in c if i < j), key=lambda t: (t[0], -t[1][0], -t[1][1]))

def show_matrix(names, clusters, pts, how):
    table([""] + names, [[names[i]] + [link(clusters[i], clusters[j], pts, how) if i != j else "-" for j in range(len(clusters))] for i in range(len(clusters))])

def agglomerative(pts, how, verbose=True):
    clusters, names = [{i} for i in range(len(pts))], [str(pid(i)) for i in range(len(pts))]
    merges, tree = [], {str(pid(i)): None for i in range(len(pts))} ## tree[name] = (left, right, height)
    while len(clusters) > 1:
        step = len(merges) + 1
        pairs = [(link(clusters[i], clusters[j], pts, how), i, j) for i in range(len(clusters)) for j in range(i + 1, len(clusters))]
        d, i, j = min(pairs) ## tie = lowest ids
        if verbose:
            hr(f"MERGE {step}: {how} distance between {len(clusters)} clusters")
            show_matrix(names, clusters, pts, how)
            print(f"nearest pair = {names[i]} {members(clusters[i])} and {names[j]} {members(clusters[j])} at {num(d)}")
        new, name = clusters[i] | clusters[j], f"M{step}"
        verbose and print(f"{name} = {members(new)}  centroid = {lst(mean_pt([pts[p] for p in new]))}")
        merges.append((step, names[i], names[j], d, len(new)))
        tree[name] = (names[i], names[j], d)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [new]
        names = [n for k, n in enumerate(names) if k not in (i, j)] + [name]
    return merges, tree, names[0]

def dendrogram(tree, name, pre="", last=True):
    node = tree[name]
    print(pre + ("`- " if last else "|- ") + name + (f" (h = {num(node[2])})" if node else ""))
    if node:
        kids, pre = node[:2], pre + ("   " if last else "|  ")
        for k, child in enumerate(kids): dendrogram(tree, child, pre, k == len(kids) - 1)

def cut(tree, root, k): ## undo the last k-1 merges
    groups = [root]
    while len(groups) < k:
        top = max((g for g in groups if tree[g]), key=lambda g: int(g[1:]), default=None)
        if top is None: break
        groups = [g for g in groups if g != top] + list(tree[top][:2])
    return groups

def leaves(tree, name): return {int(name) - km.BASE} if tree[name] is None else leaves(tree, tree[name][0]) | leaves(tree, tree[name][1])

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
        verbose and print(f"-> {members(left)} centroid {lst(mean_pt([pts[i] for i in left]))}  |  {members(right)} centroid {lst(mean_pt([pts[i] for i in right]))}")
        splits.append((step, members(c), dia[w][0], members(left), members(right)))
        clusters = clusters[:w] + [left, right] + clusters[w + 1:]
    return clusters, splits

def report(pts, clusters, sil):
    lab = [0] * len(pts)
    for j, c in enumerate(clusters):
        for i in c: lab[i] = j
    cents = [mean_pt([pts[i] for i in c]) for c in clusters]
    hr(f"RESULT: {len(clusters)} cluster(s)")
    for j, c in enumerate(clusters): print(f"C{j} {lst(cents[j])}: points {members(c)}")
    km.wcss(pts, cents, lab)
    if sil: km.silhouette(pts, lab)

def interactive(ap):
    km.BASE = int(ask("point numbering starts at 0 or 1", str(km.BASE)))
    argv = ["--base", str(km.BASE)]
    km.FEATURES, km.DATA = FEATURES, DATA
    if ask("edit the data set? y/N", "n").lower() == "y": km.ask_data()
    argv += ["--method", ask("method: agglomerative / divisive", "agglomerative")]
    if argv[-1] == "agglomerative": argv += ["--linkage", ask("linkage: centroid / single / complete / average", "centroid")]
    k = ask("stop at k clusters (blank = full tree)")
    if k: argv += ["-k", k]
    if ask("standardize features? y/N", "n").lower() == "y": argv.append("--std")
    if ask("silhouette? Y/n", "y").lower() == "n": argv.append("--no-sil")
    return ap.parse_args(argv)

def main():
    global FEATURES, DATA
    ap = argparse.ArgumentParser(description="hierarchical clustering, step by step")
    ap.add_argument("--method", choices=["agglomerative", "divisive"], default="agglomerative")
    ap.add_argument("--linkage", choices=["centroid", "single", "complete", "average"], default="centroid")
    ap.add_argument("-k", type=int, help="clusters to report, default full tree")
    ap.add_argument("--std", action="store_true", help="z-score features first")
    ap.add_argument("--no-sil", action="store_true", help="skip silhouette")
    ap.add_argument("--base", type=int, choices=[0, 1], default=km.BASE, help="point numbering start")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    km.BASE = a.base
    if len(sys.argv) == 1: FEATURES, DATA = km.FEATURES, km.DATA ## prompts may have replaced them

    pts = [list(p) for p in DATA]
    if a.k and not 0 < a.k <= len(pts): sys.exit(f"k must be 1..{len(pts)}")
    hr("STEP 1: data set")
    table(["#"] + FEATURES, [[pid(i)] + list(p) for i, p in enumerate(pts)])
    if a.std:
        pts, mu, sd = km.zscore(pts)
        hr("STEP 2: standardize (z = (x - mean) / std, ddof=0)")
        table(["feature", "mean", "std"], [[f, m, s] for f, m, s in zip(FEATURES, mu, sd)])
        table(["#"] + FEATURES, [[pid(i)] + p for i, p in enumerate(pts)])

    if a.method == "divisive":
        clusters, splits = divisive(pts, a.k)
        hr("SPLIT TABLE (top down)")
        for st, c, d, l, r in splits: print(f"{st}: {c} at diameter {num(d)} -> {l} | {r}")
        report(pts, clusters, not a.no_sil)
        return
    merges, tree, root = agglomerative(pts, a.linkage)
    hr("MERGE TABLE (bottom up)")
    table(["step", "A", "B", "distance", "size"], merges)
    hr("DENDROGRAM (h = merge distance)")
    dendrogram(tree, root)
    if a.k:
        hr(f"CUT at k = {a.k}: stop after merge {len(pts) - a.k}")
        report(pts, [leaves(tree, g) for g in cut(tree, root, a.k)], not a.no_sil)

if __name__ == "__main__": main()
