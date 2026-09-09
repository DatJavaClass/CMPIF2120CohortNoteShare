## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## k-means bare metal calculator, stdlib only
## no flags = prompts, -h = flags

import argparse, math, random, sys

## edit the data set here, one row per point
FEATURES = ["x", "y"]
DATA = [(1, 1), (2, 2), (2, 7), (3, 8), (1, 3), \
        (4, 7), (8, 2), (7, 3), (9, 3), (6, 5)]
BASE = 0 ## point numbering, 1 to match lecture figures

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=10): return "".join(num(c).rjust(w) for c in cells)
def table(head, rows, w=10): print(row(head, w)); [print(row(r, w)) for r in rows]
def lst(v): return "(" + ", ".join(num(x) for x in v) + ")"
def dbg(tag, v): print(f"[dbg] {tag}: {v}")
def pid(i): return i + BASE ## display id

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

def dist(a, b): return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
def mean_pt(pts): return [sum(c) / len(pts) for c in zip(*pts)]
def nearest(p, cents): return min(range(len(cents)), key=lambda j: (dist(p, cents[j]), j)) ## tie = lower id

def zscore(pts):
    cols = list(zip(*pts))
    mu = [sum(c) / len(c) for c in cols]
    sd = [math.sqrt(sum((x - m) ** 2 for x in c) / len(c)) for c, m in zip(cols, mu)]
    return [[(x - m) / s if s else 0.0 for x, m, s in zip(p, mu, sd)] for p in pts], mu, sd

def init_centroids(pts, k, how, spec, seed):
    rng, n = random.Random(seed), len(pts)
    hr(f"INIT: {how}" + (f" seed={seed}" if how in ("random", "kmeans++") else ""))
    match how:
        case "index":
            idx = [i - BASE for i in parse_csv(spec, int)]
            if len(idx) != k or any(i < 0 or i >= n for i in idx): sys.exit(f"need {k} valid point ids")
        case "coords":
            cents = [parse_csv(c) for c in spec.split(";")]
            if len(cents) != k or any(len(c) != len(FEATURES) for c in cents): sys.exit(f"need {k} centroids as x,y;x,y")
            for j, c in enumerate(cents): print(f"C{j} = {lst(c)} (arbitrary point in space)")
            return cents
        case "random": idx = rng.sample(range(n), k)
        case _: ## kmeans++ or farthest
            idx = [rng.randrange(n)]
            print(f"C0 = point {pid(idx[0])} picked at random")
            while len(idx) < k:
                d2 = [min(dist(p, pts[i]) for i in idx) ** 2 for p in pts]
                tot = sum(d2)
                table(["#", "D^2", "prob"], [[pid(i), d2[i], d2[i] / tot if tot else 0] for i in range(n) if i not in idx])
                if how == "farthest": pick = max(range(n), key=lambda i: (d2[i], -i))
                else:
                    r, acc, pick = rng.random() * tot, 0, n - 1
                    for i in range(n):
                        acc += d2[i]
                        if acc >= r: pick = i; break
                print(f"C{len(idx)} = point {pid(pick)} ({'farthest' if how == 'farthest' else 'drawn by D^2 weight'})")
                idx.append(pick)
    for j, i in enumerate(idx): print(f"C{j} = point {pid(i)} {lst(pts[i])}")
    return [list(pts[i]) for i in idx]

def assign(pts, cents, verbose):
    lab = [nearest(p, cents) for p in pts]
    if verbose:
        table(["#"] + [f"d(C{j})" for j in range(len(cents))] + ["cluster"], [[pid(i)] + [dist(p, c) for c in cents] + [f"C{lab[i]}"] for i, p in enumerate(pts)])
    return lab

def update(pts, cents, lab, verbose):
    new = []
    for j, c in enumerate(cents):
        members = [p for p, l in zip(pts, lab) if l == j]
        if not members: new.append(c); verbose and print(f"C{j}: empty, centroid kept"); continue
        m = mean_pt(members)
        if verbose:
            ids = ", ".join(str(pid(i)) for i, l in enumerate(lab) if l == j)
            print(f"C{j} = mean of {{{ids}}} = " + lst([sum(col) for col in zip(*members)]) + f" / {len(members)} = {lst(m)}")
        new.append(m)
    return new

def kmeans(pts, cents, max_iter, verbose=True):
    lab = None
    for it in range(1, max_iter + 1):
        verbose and hr(f"ITERATION {it}: assign each point to nearest centroid")
        new_lab = assign(pts, cents, verbose)
        verbose and hr(f"ITERATION {it}: move centroids to cluster means")
        new_cents = update(pts, cents, new_lab, verbose)
        moved = max(dist(a, b) for a, b in zip(cents, new_cents))
        changed = sum(a != b for a, b in zip(lab, new_lab)) if lab else len(pts)
        verbose and print(f"reassigned = {changed}, max centroid move = {num(moved)}")
        lab, cents = new_lab, new_cents
        if changed == 0 and moved < 1e-12:
            verbose and print(f"converged at iteration {it}")
            return lab, cents, it
    verbose and print(f"timeout after {max_iter} iterations")
    return lab, cents, max_iter

def wcss(pts, cents, lab, verbose=True):
    tot, rows = 0, []
    for j, c in enumerate(cents):
        sq = [dist(p, c) ** 2 for p, l in zip(pts, lab) if l == j]
        tot += sum(sq)
        rows.append([f"C{j}", len(sq), sum(sq)])
    if verbose:
        hr("WCSS = sum over clusters, sum over points of d(point, centroid)^2")
        table(["cluster", "n", "sum d^2"], rows)
        print(f"WCSS = {' + '.join(num(r[2]) for r in rows)} = {num(tot)}")
    return tot

def silhouette(pts, lab):
    hr("SILHOUETTE: s = (b - a) / max(a, b)")
    k, rows, scores = max(lab) + 1, [], []
    if k < 2: print("needs k >= 2"); return None
    for i, p in enumerate(pts):
        own = [dist(p, q) for j, q in enumerate(pts) if j != i and lab[j] == lab[i]]
        others = [sum(dist(p, q) for j, q in enumerate(pts) if lab[j] == c) / lab.count(c) for c in range(k) if c != lab[i] and lab.count(c)]
        a, b = (sum(own) / len(own) if own else 0.0), min(others)
        s = (b - a) / max(a, b) if own and max(a, b) else 0.0 ## singleton = 0, sklearn rule
        scores.append(s)
        rows.append([pid(i), f"C{lab[i]}", a, b, s])
    table(["#", "cluster", "a (own)", "b (next)", "s"], rows)
    print(f"mean silhouette = {num(sum(scores))} / {len(scores)} = {num(sum(scores) / len(scores))}  (+1 tight, 0 boundary, -1 wrong cluster)")
    return sum(scores) / len(scores)

def elbow(pts, kmax, how, seed, max_iter):
    hr(f"ELBOW: WCSS for k = 1..{kmax}")
    rows, prev = [], None
    for k in range(1, kmax + 1):
        best = None
        for t in range(10): ## 10 restarts, keep lowest WCSS
            rng = random.Random(seed + t)
            idx = rng.sample(range(len(pts)), k)
            if how == "kmeans++" or how == "farthest":
                idx = [rng.randrange(len(pts))]
                while len(idx) < k: idx.append(max(range(len(pts)), key=lambda i: (min(dist(pts[i], pts[c]) for c in idx), -i)))
            lab, cents, _ = kmeans(pts, [list(pts[i]) for i in idx], max_iter, verbose=False)
            w = wcss(pts, cents, lab, verbose=False)
            if best is None or w < best: best = w
        rows.append([k, best, "" if prev is None else prev - best])
        prev = best
    table(["k", "WCSS", "drop"], rows, 12)
    drops = [r[2] for r in rows[1:]]
    for i in range(1, len(drops)): ## first drop under half the previous
        if drops[i] < 0.5 * drops[i - 1]: print(f"elbow rule -> k = {i + 1} (drop after it shrinks to {num(drops[i])})"); return
    print("no clear elbow in this range")

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def ask_data():
    global FEATURES, DATA
    FEATURES = ask("feature names, comma separated", ",".join(FEATURES)).split(",")
    print(f"rows as {','.join(FEATURES)}; blank line ends")
    rows = []
    while True:
        line = input(f"row {pid(len(rows))}: ").strip()
        if not line: break
        try: rows.append(tuple(float(x) for x in line.split(",")))
        except ValueError: print("  not numeric, retry"); continue
        if len(rows[-1]) != len(FEATURES): rows.pop(); print(f"  need {len(FEATURES)} values, retry")
    if rows: DATA = rows
    else: print("no rows, keeping built in data")

def interactive(ap):
    global BASE
    BASE = int(ask("point numbering starts at 0 or 1", str(BASE)))
    if ask("edit the data set? y/N", "n").lower() == "y": ask_data()
    argv = ["--base", str(BASE), "-k", ask("k", "3")]
    how = ask("init: index / coords / random / kmeans++ / farthest", "index")
    argv += ["--init", how]
    if how == "index": argv += ["--spec", ask("point ids for initial centroids", ",".join(str(pid(i)) for i in range(int(argv[3]))))]
    elif how == "coords": argv += ["--spec", ask("centroid coords x,y;x,y")]
    else: argv += ["--seed", ask("seed", "0")]
    if ask("standardize features? y/N", "n").lower() == "y": argv.append("--std")
    argv += ["--max-iter", ask("max iterations (timeout)", "100")]
    if ask("silhouette? Y/n", "y").lower() == "n": argv.append("--no-sil")
    e = ask("elbow sweep up to k = (blank skips)")
    if e: argv += ["--elbow", e]
    return ap.parse_args(argv)

def main():
    global BASE
    ap = argparse.ArgumentParser(description="k-means, step by step")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--init", choices=["index", "coords", "random", "kmeans++", "farthest"], default="index")
    ap.add_argument("--spec", help="index: ids 2,4,9  coords: 1,2;5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--std", action="store_true", help="z-score features first")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--no-sil", action="store_true", help="skip silhouette")
    ap.add_argument("--elbow", type=int, help="also sweep WCSS for k = 1..N")
    ap.add_argument("--base", type=int, choices=[0, 1], default=BASE, help="point numbering start")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    BASE = a.base

    pts = [list(p) for p in DATA]
    if not 0 < a.k <= len(pts): sys.exit(f"k must be 1..{len(pts)}")
    if a.init in ("index", "coords") and not a.spec: a.spec = ",".join(str(pid(i)) for i in range(a.k)) if a.init == "index" else sys.exit("coords init needs --spec")
    hr("STEP 1: data set")
    table(["#"] + FEATURES, [[pid(i)] + list(p) for i, p in enumerate(pts)])
    if a.std:
        pts, mu, sd = zscore(pts)
        hr("STEP 2: standardize (z = (x - mean) / std, ddof=0)")
        table(["feature", "mean", "std"], [[f, m, s] for f, m, s in zip(FEATURES, mu, sd)])
        table(["#"] + FEATURES, [[pid(i)] + p for i, p in enumerate(pts)])

    cents = init_centroids(pts, a.k, a.init, a.spec, a.seed)
    lab, cents, it = kmeans(pts, cents, a.max_iter)
    hr(f"RESULT after {it} iteration(s)")
    for j, c in enumerate(cents): print(f"C{j} {lst(c)}: points {{{', '.join(str(pid(i)) for i, l in enumerate(lab) if l == j)}}}")
    wcss(pts, cents, lab)
    if not a.no_sil: silhouette(pts, lab)
    if a.elbow: elbow(pts, min(a.elbow, len(pts)), a.init, a.seed, a.max_iter)

if __name__ == "__main__": main()
