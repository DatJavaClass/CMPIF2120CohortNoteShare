## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## itemset support and subset property bare metal calculator, stdlib only
## no flags = prompts, -h = flags

import argparse, itertools, math, sys

## edit the data set here, one transaction per row (A milk, C cereal,clear D sugar)
DATA = [("A", "B", "C"), ("B", "D"), ("B", "C"), ("A", "B", "E"), ("A", "C"), \
        ("B", "C"), ("A", "C"), ("A", "B", "C", "E"), ("A", "B", "D")]
BASE = 1 ## transaction numbering, 1 to match lecture tables

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=10): return "".join(num(c).rjust(w) for c in cells)
def table(head, rows, w=10): print(row(head, w)); [print(row(r, w)) for r in rows]
def dbg(tag, v): print(f"[dbg] {tag}: {v}")
def tid(i): return f"T{i + BASE}" ## display id

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

## itemset helpers, lecture 2
def iset(s): return frozenset(c for c in s.upper() if c.isalnum()) ## "A,B" "AB" "A B" all parse
def name(s): return "".join(sorted(s)) or "{}"
def items(data): return sorted({x for t in data for x in t})
def support(s, data): return sum(set(s) <= set(t) for t in data) ## any superset counts
def tids(s, data): return ", ".join(tid(i) for i, t in enumerate(data) if set(s) <= set(t)) or "none"
def pct(c, n): return f"{c}/{n} = {100 * c / n:.1f}%"
def subsets(s, k): return sorted((frozenset(c) for c in itertools.combinations(sorted(s), k)), key=name)
def all_itemsets(data, k): return subsets(items(data), k)
def min_count(min_sup, n): return math.ceil(min_sup * n) if 0 < min_sup < 1 else int(min_sup)

def sup_table(sets, data, min_sup, verbose=True):
    n, counts = len(data), {s: support(s, data) for s in sets}
    if verbose:
        print(row(["itemset", "count", "support"], 13) + "  frequent  in transactions")
        for s, c in counts.items(): print(row([name(s), c, pct(c, n)], 13) + f"  {'yes' if c >= min_sup else 'no':<8}  {tids(s, data)}")
        print(f"frequent (count >= {min_sup}): {', '.join(name(s) for s, c in counts.items() if c >= min_sup) or 'none'}")
    return counts

def show_data(rows): ## STEP 1, siblings reuse it
    data = [tuple(sorted(set(t))) for t in rows] ## sets, so duplicates and order drop
    it = items(data)
    hr("STEP 1: market basket data (TID + items, alphabetical)")
    for i, t in enumerate(data): print(f"{tid(i):>4}  {', '.join(t)}")
    print(f"{len(data)} transactions, {len(it)} items: {', '.join(it)}")
    return data, it, len(data)

def threshold(min_sup, n, tail=""): ## prints the magic number, returns the count
    ms = min_count(min_sup, n)
    print(f"\nmin_sup = {num(min_sup)} of {n} = ceil({num(min_sup * n)}) = {ms} transactions{tail}" if 0 < min_sup < 1 else f"\nmin_sup = {ms} transactions = {pct(ms, n)}{tail}")
    return ms

def flags(data):
    it = items(data)
    table(["TID"] + it, [[tid(i)] + [int(x in t) for x in it] for i, t in enumerate(data)], 5)
    print("1 = item in the basket, column sum = size 1 support count")

def subset_check(s, data, min_sup, full=False): ## full = every size, else k-1 only
    c, ok, fq = support(s, data), True, []
    print(f"{name(s)}: count {c} {'>=' if c >= min_sup else '<'} {min_sup}, {'frequent' if c >= min_sup else 'not frequent'}")
    for k in (range(len(s) - 1, 0, -1) if full else [len(s) - 1]):
        for sub in subsets(s, k):
            cs = support(sub, data)
            ok, fq = ok and cs >= c, fq + [cs >= min_sup]
            print(f"  size {k} subset {name(sub):<{len(s)}}  count {cs} >= {c}  {'ok' if cs >= c else 'VIOLATED'}, {'frequent' if cs >= min_sup else 'not frequent'}")
    if c >= min_sup: print(f"subset property {'holds' if ok and all(fq) else 'BROKEN'}: {sum(fq)} of {len(fq)} subsets frequent (every subset count >= {c} >= {min_sup})")
    else: print(f"{name(s)} is not frequent, the property makes no claim; subset counts still >= {c} ({sum(fq)} of {len(fq)} frequent on their own)")
    return ok

def storage(it, pairs): ## lecture 3, where do pair counts live
    n, nz = len(it), sum(1 for c in pairs.values() if c)
    hr("STEP 4: storing the size 2 counts (lecture 3)")
    print(f"n = {n} items: matrix n*n = {n}*{n} = {n * n} cells, triangular n(n-1)/2 = {n}*{n - 1}/2 = {n * (n - 1) // 2}, triples 3 per nonzero pair = 3*{nz} = {3 * nz}")
    print("triangular index for i < j (0 based) = i*(2n-i-1)/2 + j-i-1, one cell per unordered pair")
    table(["pair", "i", "j", "index", "count"], [[name(s)] + [i, j, i * (2 * n - i - 1) // 2 + j - i - 1, c] for s, c in pairs.items() for i, j in [tuple(it.index(x) for x in sorted(s))]], 8)
    if pairs.get(frozenset("CD"), 1) == 0: print("nobody buys cereal and sugar together, so the CD cell is wasted space, hence the hash of triples")

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def ask_data():
    global DATA
    print("transactions as items, e.g. A,B,C or ABC; blank line ends")
    rows = []
    while True:
        line = input(f"{tid(len(rows))}: ").strip()
        if not line: break
        s = iset(line)
        if s: rows.append(tuple(sorted(s)))
        else: print("  no items, retry")
    if rows: DATA = rows
    else: print("no rows, keeping built in data")

def interactive(ap):
    global BASE
    BASE = int(ask("transaction numbering starts at 0 or 1", str(BASE)))
    argv = ["--base", str(BASE)]
    if ask("edit the data set? y/N", "n").lower() == "y": ask_data()
    argv += ["--min-sup", ask("min support (count, or fraction under 1)", "2")]
    q = ask("query itemset, e.g. A,B,E (blank = full enumeration)")
    if q: argv += ["-q", q]
    if not q or ask("also enumerate every itemset? y/N", "n").lower() == "y": argv += ["--kmax", ask("largest itemset size to enumerate", "3")]
    return ap.parse_args(argv)

def main():
    global BASE
    ap = argparse.ArgumentParser(description="itemset support and the subset property, step by step")
    ap.add_argument("-q", "--query", help="itemset to check, e.g. A,B,E")
    ap.add_argument("--min-sup", type=float, default=2, help="min support count, or fraction under 1")
    ap.add_argument("--kmax", type=int, help="largest itemset size to enumerate, default 3")
    ap.add_argument("--base", type=int, choices=[0, 1], default=BASE, help="transaction numbering start")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    BASE = a.base

    data, it, n = show_data(DATA)
    hr("binary flag form")
    flags(data)
    ms = threshold(a.min_sup, n, " (magic number, someone gave it to us)")

    if a.query:
        s = iset(a.query)
        if not s or s - set(it): sys.exit(f"query needs items from {', '.join(it)}")
        hr(f"QUERY: itemset {name(s)}, support = transactions containing all of it")
        print(f"support({name(s)}) = {support(s, data)} = {pct(support(s, data), n)}, in {tids(s, data)}")
        subset_check(s, data, ms, full=True)
        if not a.kmax: return
    kmax, freq, pairs = min(a.kmax or 3, len(it)), {}, {}
    for k in range(1, kmax + 1):
        hr(f"STEP 2: size {k} itemsets, {len(it)} choose {k} = {math.comb(len(it), k)}, support = transactions containing the set")
        counts = sup_table(all_itemsets(data, k), data, ms)
        freq.update({s: c for s, c in counts.items() if c >= ms})
        if k == 2: pairs = counts
    hr(f"STEP 3: subset property, every subset of a frequent itemset is frequent (min_sup {ms})")
    big = sorted((s for s in freq if len(s) >= 2), key=lambda s: (len(s), name(s)))
    for s in big: subset_check(s, data, ms)
    if not big: print("no frequent itemset of size 2 or more, nothing to check")
    if pairs: storage(it, pairs)

if __name__ == "__main__": main()
