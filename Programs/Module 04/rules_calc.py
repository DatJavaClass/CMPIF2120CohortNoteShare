## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## association rules bare metal calculator, stdlib only
## reuses helpers from support_calc.py (same folder)
## no flags = prompts, -h = flags

import argparse, sys
import support_calc as sc
from support_calc import hr, table, ask, name, iset, pct, subsets, all_itemsets, show_data, threshold

def rule_name(lhs, rhs): return f"{name(lhs)} -> {name(rhs)}"
def conf_pct(v, d=0): return f"{100 * v:.{d}f}%"
def conf_frac(v): return v / 100 if v > 1 else v ## 60 means 60%

def look(sup, s): ## no count, no guessing
    if s not in sup: sys.exit(f"no support count for {name(s)}, nothing to compute")
    return sup[s]

def gen_rules(s, sup, n, min_sup, min_conf, verbose=True): ## every split of s into lhs -> rhs
    c, rules = look(sup, s), []
    if verbose: print(f"rule support = sup({name(s)}) = {c} for every split (lhs union rhs is always {name(s)}), {c} {'>=' if c >= min_sup else '<'} {min_sup} so {'all' if c >= min_sup else 'no'} rules pass support")
    for k in range(len(s) - 1, 0, -1): ## lhs size, lecture order
        verbose and print(f"{k} on the left, {len(s) - k} on the right:")
        for lhs in subsets(s, k):
            rhs, cl = s - lhs, look(sup, lhs)
            conf = c / cl if cl else 0.0
            ok = c >= min_sup and conf >= min_conf
            rules.append((lhs, rhs, c, conf, ok))
            verbose and print(f"  conf({rule_name(lhs, rhs)}) = sup({name(s)}) / sup({name(lhs)}) = {pct(c, cl) if cl else '0/0 = 0%'} {'>=' if conf >= min_conf else '<'} {conf_pct(min_conf)}, {'pass' if ok else 'fail'}")
    return rules

def empty_note(s, c, n): ## the lecture mentions it once, so do we
    print(f"note: {{}} -> {name(s)} has conf = sup({name(s)}) / N = {pct(c, n)} (every transaction contains {{}}), useless, ignored from here on")

def result(rules, n):
    hr("STEP 4: rules passing min support and min confidence, by confidence then support")
    keep = sorted((r for r in rules if r[4]), key=lambda r: (-r[3], -r[2], rule_name(r[0], r[1])))
    if keep: table(["rule", "count", "support", "confidence"], [[rule_name(l, r), c, pct(c, n), conf_pct(cf, 1)] for l, r, c, cf, _ in keep], 14)
    else: print("no rule passes")

def interactive(ap):
    try: sc.BASE = int(ask("transaction numbering starts at 0 or 1", str(sc.BASE)))
    except ValueError: sys.exit("base must be 0 or 1")
    argv = ["--base", str(sc.BASE)]
    if ask("edit the data set? y/N", "n").lower() == "y": sc.ask_data()
    argv += ["-s", ask("frequent itemset to split into rules, e.g. A,B,E", "A,B,E")]
    argv += ["--min-sup", ask("min support (count, or fraction under 1)", "2"), "--min-conf", ask("min confidence (fraction)", "0.5")]
    if ask("rules from every frequent itemset instead? y/N", "n").lower() == "y": argv += ["--all", "--kmax", ask("largest itemset size", "3")]
    return ap.parse_args(argv)

def main():
    ap = argparse.ArgumentParser(description="association rules from a frequent itemset, step by step")
    ap.add_argument("-s", "--set", default="A,B,E", help="itemset to split into rules, e.g. A,B,E")
    ap.add_argument("--min-sup", type=float, default=2, help="min support count, or fraction under 1")
    ap.add_argument("--min-conf", type=float, default=0.5, help="min confidence fraction, default 0.5")
    ap.add_argument("--all", action="store_true", help="rules from every frequent itemset up to kmax (ignores -s)")
    ap.add_argument("--kmax", type=int, default=3, help="largest itemset size for --all")
    ap.add_argument("--base", type=int, choices=[0, 1], default=sc.BASE, help="transaction numbering start")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    sc.BASE = a.base

    data, it, n = show_data(sc.DATA)
    mc = conf_frac(a.min_conf)
    ms = threshold(a.min_sup, n, f", min_conf = {conf_pct(mc)} (both magic numbers, someone gave them to us)")

    if a.all: ## every frequent itemset of size 2 or more
        kmax, sup = min(a.kmax, len(it)), {}
        for k in range(1, kmax + 1):
            hr(f"STEP 2: size {k} support counts (min_sup {ms})")
            sup.update(sc.sup_table(all_itemsets(data, k), data, ms))
        targets = sorted((s for s, c in sup.items() if len(s) >= 2 and c >= ms), key=lambda s: (len(s), name(s)))
        if not targets: sys.exit("no frequent itemset of size 2 or more, no rules to make")
    else:
        s = iset(a.set)
        if not s or s - set(it): sys.exit(f"itemset needs items from {', '.join(it)}")
        if len(s) < 2: sys.exit(f"{name(s)} has one item, no split gives a rule with both sides")
        hr(f"STEP 2: support counts for {name(s)} and every subset (min_sup {ms})")
        sup = sc.sup_table([x for k in range(len(s), 0, -1) for x in subsets(s, k)], data, ms)
        if sup[s] < ms: print(f"{name(s)} is not frequent, the lecture only splits frequent sets; rules below fail support")
        targets = [s]

    rules = []
    for i, s in enumerate(targets):
        hr(f"STEP 3: rules from {name(s)}, every split into lhs -> rhs, {2 ** len(s) - 2} rules, conf = sup(lhs union rhs) / sup(lhs)")
        if i == 0: empty_note(s, sup[s], n)
        rules += gen_rules(s, sup, n, ms, mc)
    result(rules, n)

if __name__ == "__main__": main()
