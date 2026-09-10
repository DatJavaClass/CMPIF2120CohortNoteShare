## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## a-priori algorithm bare metal calculator, stdlib only
## reuses helpers from support_calc.py and rules_calc.py (same folder)
## no flags = prompts, -h = flags

import argparse, itertools, math, sys
import support_calc as sc
from support_calc import hr, table, ask, name, iset, pct, subsets, all_itemsets, sup_table, show_data, threshold
from rules_calc import gen_rules, rule_name, conf_pct, conf_frac

## lecture defaults for the two transaction free modes
FREQ, CAND, INFREQ = "ABC,ABD,ACD,ACE,BCD", "ABCD,ACDE", "D,F,Z"

def names(sets): return ", ".join(name(s) for s in sets) or "none"
def isets(s): return [x for x in (iset(p) for p in s.split(",")) if x] ## "ABC,ABD" -> frozensets
def by_name(sets): return sorted(sets, key=name)

def join(prev, k): ## prefix join, share the first k-2 items
    out = []
    for a, b in itertools.combinations(by_name(prev), 2):
        if name(a)[:k - 2] != name(b)[:k - 2]: continue
        out.append(a | b)
        print(f"{name(a)} + {name(b)} -> {name(a | b)}")
    if not out: print(f"no pair in L{k - 1} shares a {k - 2} item prefix, nothing to join")
    return out

def prune(cands, freq): ## subset property veto, hide one item
    keep, freq = [], set(freq)
    for c in cands:
        subs = subsets(c, len(c) - 1)
        miss = [s for s in subs if s not in freq]
        print(f"{name(c)}: " + ", ".join(f"hide {name(c - s)} -> {name(s)} {'ok' if s in freq else 'MISSING'}" for s in subs))
        print(f"  {f'all {len(subs)} subsets frequent, good candidate' if not miss else f'vetoed by {names(miss)}, one missing subset is enough'}")
        if not miss: keep.append(c)
    return keep

def apriori(data, ms):
    hr(f"INIT: L1 = every single item with count >= {ms}")
    sup = sup_table(all_itemsets(data, 1), data, ms)
    L, k = {1: [s for s, c in sup.items() if c >= ms]}, 2
    if not L[1]: print("stop: no frequent single item, nothing to join")
    while L[k - 1]:
        hr(f"STEP 2a: join L{k - 1} pairs into C{k}, then prune by the subset property")
        cands = join(L[k - 1], k)
        keep = prune(cands, L[k - 1]) if cands else []
        if cands: print(f"C{k} survivors: {names(keep)}")
        if not keep: print(f"stop: {'nothing to join' if not cands else 'every candidate vetoed'}, no C{k} to count"); break
        hr(f"STEP 2b: count C{k} against the {len(data)} transactions")
        counts = sup_table(keep, data, ms)
        sup.update(counts)
        hr(f"STEP 2c: L{k} = C{k} with count >= {ms}")
        L[k] = [s for s, c in counts.items() if c >= ms]
        print(f"L{k} = {names(L[k])}")
        if not L[k]: print(f"stop: every C{k} candidate fell under min_sup {ms}, no L{k} to join")
        k += 1
    hr("FREQUENT ITEMSETS")
    for j in L:
        if L[j]: print(f"L{j} ({len(L[j])}): {names(L[j])}")
    return L, sup

def rules(L, sup, n, ms, mc):
    hr(f"STEP 3: rules from every frequent itemset of size >= 2, conf(X -> Y) = sup(XY) / sup(X) >= {conf_pct(mc)}")
    strong, big = [], [s for j in L for s in L[j] if j >= 2]
    if not big: print("no frequent itemset of size 2 or more, no rules")
    for s in big:
        print(f"\n{name(s)}: rule support = sup({name(s)}) = {sup[s]} = {pct(sup[s], n)}, same for every split")
        for l, r, c, cf, p in gen_rules(s, sup, n, ms, mc, verbose=False):
            print(f"  {rule_name(l, r):<11} sup {c}  conf = {c}/{sup[l]} = {100 * cf:.1f}%  {'pass' if p else 'fail'}")
            if p: strong.append((l, r, c, cf))
    hr(f"STRONG RULES: support >= {ms} and confidence >= {conf_pct(mc)}, by confidence then support")
    for l, r, c, cf in sorted(strong, key=lambda t: (-t[3], -t[2], rule_name(t[0], t[1]))): print(f"{rule_name(l, r):<11} sup {pct(c, n):<14} conf {100 * cf:.1f}%")
    if not strong: print("none")

def check(freq, cands): ## lecture worked example, no data set
    hr("CHECK: candidate k itemsets against the frequent (k-1) list, no transactions needed")
    print(f"frequent: {names(by_name(freq))}")
    keep = prune(cands, freq)
    print(f"\ngood candidates: {names(keep)}, vetoed: {names([c for c in cands if c not in keep])}")

def combos(n, k, bad): ## why prune, the combinatorics
    it, g, f = [chr(65 + i) for i in range(n)], n - len(bad), math.factorial
    hr(f"COUNT: why prune, n = {n} items ({', '.join(it)}), itemset sizes 1..{k}")
    print(f"{n} choose {k} = {n}! / ({k}! * {n - k}!) = {f(n)} / ({f(k)} * {f(n - k)}) = {math.comb(n, k)}")
    print(f"infrequent items {', '.join(bad) or 'none'} veto every set containing them, {n} - {len(bad)} = {g} items left to combine")
    if any(x not in it for x in bad): print(f"({', '.join(x for x in bad if x not in it)} not among {it[0]}..{it[-1]}, the lecture counts it anyway)")
    table(["size", "formula", "all", "survive", "vetoed"], [[j, f"{n}!/({j}!*{n - j}!)", math.comb(n, j), math.comb(g, j), math.comb(n, j) - math.comb(g, j)] for j in range(1, k + 1)], 13)
    print(f"size 2: {n} choose 2 = {math.comb(n, 2)} drops to {g} choose 2 = {math.comb(g, 2)}" + (f"; size {k}: {math.comb(n, k)} drops to {math.comb(g, k)}" if k != 2 else ""))

def interactive(ap):
    match ask("mode: run / check / count", "run").lower():
        case "check": return ap.parse_args(["--check", "--freq", ask("frequent (k-1) itemsets", FREQ), "--cand", ask("candidate k itemsets", CAND)])
        case "count": return ap.parse_args(["--count", "-n", ask("n items", "7"), "-k", ask("itemset size k", "3"), "--infreq", ask("infrequent single items", INFREQ)])
    try: sc.BASE = int(ask("transaction numbering starts at 0 or 1", str(sc.BASE)))
    except ValueError: sys.exit("base must be 0 or 1")
    if ask("edit the data set? y/N", "n").lower() == "y": sc.ask_data()
    return ap.parse_args(["--base", str(sc.BASE), "--min-sup", ask("min support (count, or fraction under 1)", "2"), "--min-conf", ask("min confidence (fraction, or percent)", "0.5")])

def main():
    ap = argparse.ArgumentParser(description="a-priori algorithm, step by step")
    ap.add_argument("--min-sup", type=float, default=2, help="min support count, or fraction under 1")
    ap.add_argument("--min-conf", type=float, default=0.5, help="min confidence, fraction or percent")
    ap.add_argument("--base", type=int, choices=[0, 1], default=sc.BASE, help="transaction numbering start")
    ap.add_argument("--check", action="store_true", help="candidate check from a frequent list, no transactions")
    ap.add_argument("--freq", default=FREQ, help="check: frequent (k-1) itemsets")
    ap.add_argument("--cand", default=CAND, help="check: candidate k itemsets")
    ap.add_argument("--count", action="store_true", help="why prune: n choose k and the veto savings")
    ap.add_argument("-n", type=int, default=7, help="count: items, named A..")
    ap.add_argument("-k", type=int, default=3, help="count: itemset size")
    ap.add_argument("--infreq", default=INFREQ, help="count: infrequent single items")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    if a.check:
        freq, cands = isets(a.freq), isets(a.cand)
        if not freq or any(len(c) < 2 for c in cands): sys.exit("need --freq itemsets and --cand itemsets of size 2 or more")
        return check(freq, cands)
    if a.count:
        bad = sorted(iset(a.infreq))
        if not 0 < a.k <= a.n or len(bad) > a.n: sys.exit(f"need 1 <= k <= n and at most n infrequent items")
        return combos(a.n, a.k, bad)
    sc.BASE = a.base

    data, it, n = show_data(sc.DATA)
    mc = conf_frac(a.min_conf)
    ms = threshold(a.min_sup, n, f", min_conf = {conf_pct(mc)}")
    L, sup = apriori(data, ms)
    rules(L, sup, n, ms, mc)

if __name__ == "__main__": main()
