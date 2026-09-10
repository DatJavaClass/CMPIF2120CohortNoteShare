## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## itemset support and subset property general use calculator, stdlib only
## general fork of support_calc.py: any basket file, any item names
## no flags = prompts, -h = flags

import argparse, itertools, math, os, re, sys

DATA, ITEMS, TIDS = [], [], None ## filled by load()
BASE, LETTERS = 1, True ## numbering start, "ABC" = {A, B, C}

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=10): return "".join(num(c).rjust(w) for c in cells)
def table(head, rows, w=10): print(row(head, w)); [print(row(r, w)) for r in rows]
def dbg(tag, v): print(f"[dbg] {tag}: {v}")
def tid(i): return TIDS[i] if TIDS else f"T{i + BASE}" ## display id
def toks(s): return [t for t in re.split(r"\s*,\s*|\s+", s.strip()) if t]

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

def read_lines(spec): ## file path, - for stdin, or inline A,B;C,D
    try:
        if spec == "-": text = sys.stdin.read()
        elif os.path.isfile(spec): text = open(spec, encoding="utf-8-sig").read()
        else: text = spec.replace(";", "\n")
    except OSError as e: sys.exit(f"cannot read {spec}: {e}")
    lines = [toks(l) for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not lines: sys.exit("no baskets")
    return lines

def load(spec, tidcol=False, words=False): ## baskets, or a 0/1 flag table with header
    global DATA, ITEMS, TIDS, LETTERS
    lines, off = read_lines(spec), int(tidcol)
    body = [l[off:] for l in lines[1:]]
    if body and all(len(b) == len(lines[0]) - off and set(b) <= {"0", "1"} for b in body): ## flag table
        head, lines = lines[0][off:], lines[1:]
        DATA = [tuple(h for h, c in zip(head, b) if c == "1") for b in body]
    else: DATA = [tuple(sorted(set(l[off:]))) for l in lines]
    TIDS = [l[0] for l in lines] if tidcol else None
    ITEMS = items(DATA)
    LETTERS = not words and all(len(x) == 1 for x in ITEMS)

## itemset helpers
def iset(s): return frozenset(x for t in toks(s) for x in ([t] if t in ITEMS or not LETTERS else t)) ## "A,B" "A B" "AB"
def isets(s): return [x for x in (iset(p) for p in s.split(";")) if x] ## "AB;AC" -> frozensets
def name(s): return ("" if LETTERS else ",").join(sorted(s)) or "{}"
def names(sets): return (", " if LETTERS else "; ").join(name(s) for s in sets) or "none"
def items(data): return sorted({x for t in data for x in t})
def support(s, data): return sum(set(s) <= set(t) for t in data) ## any superset counts
def tids(s, data): return ", ".join(tid(i) for i, t in enumerate(data) if set(s) <= set(t)) or "none"
def pct(c, n): return f"{c}/{n} = {100 * c / n:.1f}%"
def subsets(s, k): return sorted((frozenset(c) for c in itertools.combinations(sorted(s), k)), key=name)
def all_itemsets(data, k): return subsets(items(data), k)
def min_count(min_sup, n): return math.ceil(min_sup * n) if 0 < min_sup < 1 else int(min_sup)
def width(strs, floor): return max([floor] + [len(s) + 2 for s in strs]) ## column fits longest name

def sup_table(sets, data, min_sup, verbose=True):
    n, counts, w = len(data), {s: support(s, data) for s in sets}, width([name(s) for s in sets], 13)
    if verbose:
        print(row(["itemset", "count", "support"], w) + "  frequent  in transactions")
        for s, c in counts.items(): print(row([name(s), c, pct(c, n)], w) + f"  {'yes' if c >= min_sup else 'no':<8}  {tids(s, data)}")
        print(f"frequent (count >= {min_sup}): {names(s for s, c in counts.items() if c >= min_sup)}")
    return counts

def show_data(rows): ## STEP 1, siblings reuse it
    data = [tuple(sorted(set(t))) for t in rows] ## sets, so duplicates and order drop
    it, w = items(data), width([tid(i) for i in range(len(rows))], 4) - 2
    hr("STEP 1: basket data (TID + items, sorted)")
    for i, t in enumerate(data): print(f"{tid(i):>{w}}  {', '.join(t)}")
    print(f"{len(data)} transactions, {len(it)} items: {', '.join(it)}")
    return data, it, len(data)

def threshold(min_sup, n, tail=""): ## prints the threshold, returns the count
    ms = min_count(min_sup, n)
    print(f"\nmin_sup = {num(min_sup)} of {n} = ceil({num(min_sup * n)}) = {ms} transactions{tail}" if 0 < min_sup < 1 else f"\nmin_sup = {ms} transactions = {pct(ms, n)}{tail}")
    return ms

def flags(data):
    it = items(data)
    table(["TID"] + it, [[tid(i)] + [int(x in t) for x in it] for i, t in enumerate(data)], width(it + [tid(i) for i in range(len(data))], 5) - 1)
    print("1 = item in the basket, column sum = size 1 support count")

def subset_check(s, data, min_sup, full=False): ## full = every size, else k-1 only
    c, ok, fq, w = support(s, data), True, [], len(name(s))
    print(f"{name(s)}: count {c} {'>=' if c >= min_sup else '<'} {min_sup}, {'frequent' if c >= min_sup else 'not frequent'}")
    for k in (range(len(s) - 1, 0, -1) if full else [len(s) - 1]):
        for sub in subsets(s, k):
            cs = support(sub, data)
            ok, fq = ok and cs >= c, fq + [cs >= min_sup]
            print(f"  size {k} subset {name(sub):<{w}}  count {cs} >= {c}  {'ok' if cs >= c else 'VIOLATED'}, {'frequent' if cs >= min_sup else 'not frequent'}")
    if c >= min_sup: print(f"subset property {'holds' if ok and all(fq) else 'BROKEN'}: {sum(fq)} of {len(fq)} subsets frequent (every subset count >= {c} >= {min_sup})")
    else: print(f"{name(s)} is not frequent, the property makes no claim; subset counts still >= {c} ({sum(fq)} of {len(fq)} frequent on their own)")
    return ok

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def ask_source(): ## shared data prompts, returns argv
    argv = ["-d", ask("data: basket file, - for stdin, or inline A,B,C;B,D")]
    if ask("first token of each line is a transaction id? y/N", "n").lower() == "y": argv.append("--tid")
    if ask("items are words, never split into letters? y/N", "n").lower() == "y": argv.append("--words")
    return argv + ["--base", ask("transaction numbering starts at 0 or 1", str(BASE))]

def add_source(ap): ## shared data flags
    ap.add_argument("-d", "--data", help="basket file (one per line), - for stdin, inline A,B,C;B,D, or a 0/1 flag table")
    ap.add_argument("--tid", action="store_true", help="first token of each line is the transaction id")
    ap.add_argument("--words", action="store_true", help="items are words, never split into letters")
    ap.add_argument("--base", type=int, choices=[0, 1], default=BASE, help="transaction numbering start")

def open_data(a): ## flags to loaded data, STEP 1
    global BASE
    if not a.data: sys.exit("need -d data")
    BASE = a.base
    load(a.data, a.tid, a.words)
    return show_data(DATA)

def interactive(ap):
    argv = ask_source() + ["--min-sup", ask("min support (count, or fraction under 1)", "2")]
    q = ask("query itemsets, e.g. A,B;A,C (blank = full enumeration)")
    if q: argv += ["-q", q]
    if not q or ask("also enumerate every itemset? y/N", "n").lower() == "y": argv += ["--kmax", ask("largest itemset size to enumerate", "3")]
    if ask("show the binary flag table? y/N", "n").lower() == "y": argv.append("--flags")
    return ap.parse_args(argv)

def main():
    ap = argparse.ArgumentParser(description="itemset support and the subset property, step by step")
    add_source(ap)
    ap.add_argument("-q", "--query", help="itemsets to check, e.g. A,B;A,C")
    ap.add_argument("--min-sup", type=float, default=2, help="min support count, or fraction under 1")
    ap.add_argument("--kmax", type=int, help="largest itemset size to enumerate, default 3")
    ap.add_argument("--flags", action="store_true", help="show the binary flag table")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    data, it, n = open_data(a)
    if a.flags: hr("binary flag form"); flags(data)
    ms = threshold(a.min_sup, n)

    for s in (isets(a.query) if a.query else []):
        if s - set(it): sys.exit(f"query needs items from {', '.join(it)}")
        hr(f"QUERY: itemset {name(s)}, support = transactions containing all of it")
        print(f"support({name(s)}) = {support(s, data)} = {pct(support(s, data), n)}, in {tids(s, data)}")
        subset_check(s, data, ms, full=True)
    if a.query and not a.kmax: return

    kmax, freq = min(a.kmax or 3, len(it)), {}
    for k in range(1, kmax + 1):
        hr(f"STEP 2: size {k} itemsets, {len(it)} choose {k} = {math.comb(len(it), k)}, support = transactions containing the set")
        freq.update({s: c for s, c in sup_table(all_itemsets(data, k), data, ms).items() if c >= ms})
    hr(f"STEP 3: subset property, every subset of a frequent itemset is frequent (min_sup {ms})")
    big = sorted((s for s in freq if len(s) >= 2), key=lambda s: (len(s), name(s)))
    for s in big: subset_check(s, data, ms)
    if not big: print("no frequent itemset of size 2 or more, nothing to check")

if __name__ == "__main__": main()
