## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## a-priori algorithm general use calculator, stdlib only
## general fork of apriori_calc.py: any basket file, any item names, lift added
## reuses helpers from support_calc_GenUse.py and rules_calc_GenUse.py (same folder)
## no flags = prompts, -h = flags

import argparse, itertools, math, sys
import support_calc_GenUse as sc
from support_calc_GenUse import hr, num, table, ask, name, names, isets, pct, subsets, all_itemsets, sup_table, threshold, toks
from rules_calc_GenUse import gen_rules, rule_name, conf_pct, conf_frac, result, tail

def by_name(sets): return sorted(sets, key=name)

def join(prev, k): ## prefix join, share the first k-2 items
    out = []
    for a, b in itertools.combinations(by_name(prev), 2):
        if sorted(a)[:k - 2] != sorted(b)[:k - 2]: continue
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

def apriori(data, ms, kmax=0):
    hr(f"INIT: L1 = every single item with count >= {ms}")
    sup = sup_table(all_itemsets(data, 1), data, ms)
    L, k = {1: [s for s, c in sup.items() if c >= ms]}, 2
    if not L[1]: print("stop: no frequent single item, nothing to join")
    while L[k - 1] and (not kmax or k <= kmax):
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
    if kmax and k > kmax and L[k - 1]: print(f"\nstop: max itemset size {kmax} reached")
    hr("FREQUENT ITEMSETS")
    for j in L:
        if L[j]: print(f"L{j} ({len(L[j])}): {names(L[j])}")
    return L, sup

def rules(L, sup, n, ms, mc, ml):
    hr(f"STEP 3: rules from every frequent itemset of size >= 2, conf(X -> Y) = sup(XY) / sup(X) >= {conf_pct(mc)}" + (f", lift >= {num(ml)}" if ml else ""))
    out, big = [], [s for j in L for s in L[j] if j >= 2]
    if not big: print("no frequent itemset of size 2 or more, no rules")
    for s in big:
        print(f"\n{name(s)}: rule support = sup({name(s)}) = {sup[s]} = {pct(sup[s], n)}, same for every split")
        for l, r, c, cf, lf, p in gen_rules(s, sup, n, ms, mc, ml, verbose=False):
            print(f"  {rule_name(l, r):<{2 * len(name(s)) + 4}} sup {c}  conf = {c}/{sup[l]} = {100 * cf:.1f}%  lift {num(lf)}  {'pass' if p else 'fail'}")
            out.append((l, r, c, cf, lf, p))
    result(out, n, ml)

def check(freq, cands): ## candidate list vs frequent list, no data set
    hr("CHECK: candidate k itemsets against the frequent (k-1) list, no transactions needed")
    print(f"frequent: {names(by_name(freq))}")
    keep = prune(cands, freq)
    print(f"\ngood candidates: {names(keep)}, vetoed: {names([c for c in cands if c not in keep])}")

def combos(n, k, bad, who=""): ## why prune, the combinatorics
    g, f = n - bad, math.factorial
    hr(f"COUNT: why prune, n = {n} items, itemset sizes 1..{k}")
    print(f"{n} choose {k} = {n}! / ({k}! * {n - k}!) = {f(n)} / ({f(k)} * {f(n - k)}) = {math.comb(n, k)}")
    print(f"{bad} infrequent single item(s){f' ({who})' if who else ''} veto every set containing them, {n} - {bad} = {g} items left to combine")
    table(["size", "formula", "all", "survive", "vetoed"], [[j, f"{n}!/({j}!*{n - j}!)", math.comb(n, j), math.comb(g, j), math.comb(n, j) - math.comb(g, j)] for j in range(1, k + 1)], 13)
    print(f"size 2: {n} choose 2 = {math.comb(n, 2)} drops to {g} choose 2 = {math.comb(g, 2)}" + (f"; size {k}: {math.comb(n, k)} drops to {math.comb(g, k)}" if k != 2 else ""))

def interactive(ap):
    match ask("mode: run / check / count", "run").lower():
        case "check": return ap.parse_args(["--check", "--freq", ask("frequent (k-1) itemsets, ; separated"), "--cand", ask("candidate k itemsets, ; separated")] + (["--words"] if ask("items are words, never split into letters? y/N", "n").lower() == "y" else []))
        case "count": return ap.parse_args(["--count", "-n", ask("n items", "7"), "-k", ask("itemset size k", "3"), "--infreq", ask("infrequent single items (count or list)", "0")])
    return ap.parse_args(sc.ask_source() + ["--min-sup", ask("min support (count, or fraction under 1)", "2"), "--min-conf", ask("min confidence (fraction, or percent)", "0.5"), "--min-lift", ask("min lift (0 = off)", "0"), "--max-k", ask("stop at itemset size (0 = no limit)", "0")])

def main():
    ap = argparse.ArgumentParser(description="a-priori algorithm, step by step")
    sc.add_source(ap)
    ap.add_argument("--min-sup", type=float, default=2, help="min support count, or fraction under 1")
    ap.add_argument("--min-conf", type=float, default=0.5, help="min confidence, fraction or percent")
    ap.add_argument("--min-lift", type=float, default=0, help="min lift, 0 = off")
    ap.add_argument("--max-k", type=int, default=0, help="stop at this itemset size, 0 = no limit")
    ap.add_argument("--check", action="store_true", help="candidate check from a frequent list, no transactions")
    ap.add_argument("--freq", help="check: frequent (k-1) itemsets, ; separated")
    ap.add_argument("--cand", help="check: candidate k itemsets, ; separated")
    ap.add_argument("--count", action="store_true", help="why prune: n choose k and the veto savings")
    ap.add_argument("-n", type=int, default=7, help="count: items")
    ap.add_argument("-k", type=int, default=3, help="count: itemset size")
    ap.add_argument("--infreq", default="0", help="count: infrequent single items, a count or a list")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    if a.check:
        sc.LETTERS = not a.words
        if not a.freq or not a.cand: sys.exit("check needs --freq and --cand")
        freq, cands = isets(a.freq), isets(a.cand)
        if not freq or not cands or any(len(c) < 2 for c in cands): sys.exit("need --freq itemsets and --cand itemsets of size 2 or more")
        return check(freq, cands)
    if a.count:
        who = "" if a.infreq.isdigit() else ", ".join(toks(a.infreq))
        bad = int(a.infreq) if a.infreq.isdigit() else len(toks(a.infreq))
        if not 0 < a.k <= a.n or bad > a.n: sys.exit("need 1 <= k <= n and at most n infrequent items")
        return combos(a.n, a.k, bad, who)

    data, it, n = sc.open_data(a)
    mc = conf_frac(a.min_conf)
    ms = threshold(a.min_sup, n, tail(mc, a.min_lift))
    L, sup = apriori(data, ms, a.max_k)
    rules(L, sup, n, ms, mc, a.min_lift)

if __name__ == "__main__": main()
