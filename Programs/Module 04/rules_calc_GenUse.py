## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## association rules general use calculator, stdlib only
## general fork of rules_calc.py: any basket file, any item names, lift added
## reuses helpers from support_calc_GenUse.py (same folder)
## no flags = prompts, -h = flags

import argparse, sys
import support_calc_GenUse as sc
from support_calc_GenUse import hr, num, table, ask, name, isets, pct, subsets, all_itemsets, threshold, width

def rule_name(lhs, rhs): return f"{name(lhs)} -> {name(rhs)}"
def conf_pct(v, d=0): return f"{100 * v:.{d}f}%"
def conf_frac(v): return v / 100 if v > 1 else v ## 60 means 60%
def tail(mc, ml): return f", min_conf = {conf_pct(mc)}" + (f", min_lift = {num(ml)}" if ml else "") ## threshold() suffix

def look(sup, s): ## no count, no guessing
    if s not in sup: sys.exit(f"no support count for {name(s)}, nothing to compute")
    return sup[s]

def gen_rules(s, sup, n, min_sup, min_conf, min_lift=0, verbose=True): ## every split of s into lhs -> rhs
    c, rules = look(sup, s), []
    if verbose: print(f"rule support = sup({name(s)}) = {c} for every split (lhs union rhs is always {name(s)}), {c} {'>=' if c >= min_sup else '<'} {min_sup} so {'all' if c >= min_sup else 'no'} rules pass support")
    for k in range(len(s) - 1, 0, -1): ## lhs size, largest first
        verbose and print(f"{k} on the left, {len(s) - k} on the right:")
        for lhs in subsets(s, k):
            rhs, cl = s - lhs, look(sup, lhs)
            conf, cr = c / cl if cl else 0.0, look(sup, rhs)
            lift = conf * n / cr if cr else 0.0 ## conf / (sup(rhs) / N)
            ok = c >= min_sup and conf >= min_conf and lift >= min_lift
            rules.append((lhs, rhs, c, conf, lift, ok))
            verbose and print(f"  conf({rule_name(lhs, rhs)}) = sup({name(s)}) / sup({name(lhs)}) = {pct(c, cl) if cl else '0/0 = 0%'} {'>=' if conf >= min_conf else '<'} {conf_pct(min_conf)}, lift = conf / (sup({name(rhs)}) / N) = {num(conf)} / ({cr}/{n}) = {num(lift)}, {'pass' if ok else 'fail'}")
    return rules

def result(rules, n, min_lift=0):
    hr(f"STEP 4: rules passing min support and min confidence{' and min lift' if min_lift else ''}, by confidence then support")
    keep = sorted((r for r in rules if r[5]), key=lambda r: (-r[3], -r[2], rule_name(r[0], r[1])))
    if keep: table(["rule", "count", "support", "confidence", "lift"], [[rule_name(l, r), c, pct(c, n), conf_pct(cf, 1), lf] for l, r, c, cf, lf, _ in keep], width([rule_name(l, r) for l, r, *_ in keep], 14))
    else: print("no rule passes")

def interactive(ap):
    argv = sc.ask_source()
    if ask("rules from every frequent itemset? y/N", "n").lower() == "y": argv += ["--all", "--kmax", ask("largest itemset size", "3")]
    else: argv += ["-s", ask("itemsets to split into rules, e.g. A,B,E;A,C")]
    return ap.parse_args(argv + ["--min-sup", ask("min support (count, or fraction under 1)", "2"), "--min-conf", ask("min confidence (fraction, or percent)", "0.5"), "--min-lift", ask("min lift (0 = off)", "0")])

def main():
    ap = argparse.ArgumentParser(description="association rules from frequent itemsets, step by step")
    sc.add_source(ap)
    ap.add_argument("-s", "--set", help="itemsets to split into rules, e.g. A,B,E;A,C")
    ap.add_argument("--min-sup", type=float, default=2, help="min support count, or fraction under 1")
    ap.add_argument("--min-conf", type=float, default=0.5, help="min confidence, fraction or percent")
    ap.add_argument("--min-lift", type=float, default=0, help="min lift, 0 = off")
    ap.add_argument("--all", action="store_true", help="rules from every frequent itemset up to kmax (ignores -s)")
    ap.add_argument("--kmax", type=int, default=3, help="largest itemset size for --all")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    if not a.all and not a.set: sys.exit("need -s itemsets or --all")
    data, it, n = sc.open_data(a)
    mc = conf_frac(a.min_conf)
    ms = threshold(a.min_sup, n, tail(mc, a.min_lift))

    if a.all: ## every frequent itemset of size 2 or more
        kmax, sup = min(a.kmax, len(it)), {}
        for k in range(1, kmax + 1):
            hr(f"STEP 2: size {k} support counts (min_sup {ms})")
            sup.update(sc.sup_table(all_itemsets(data, k), data, ms))
        targets = sorted((s for s, c in sup.items() if len(s) >= 2 and c >= ms), key=lambda s: (len(s), name(s)))
        if not targets: sys.exit("no frequent itemset of size 2 or more, no rules to make")
    else:
        targets, sup = isets(a.set), {}
        for s in targets:
            if s - set(it): sys.exit(f"{name(s)} needs items from {', '.join(it)}")
            if len(s) < 2: sys.exit(f"{name(s)} has one item, no split gives a rule with both sides")
            hr(f"STEP 2: support counts for {name(s)} and every subset (min_sup {ms})")
            sup.update(sc.sup_table([x for k in range(len(s), 0, -1) for x in subsets(s, k)], data, ms))
            if sup[s] < ms: print(f"{name(s)} is not frequent, rules below fail support")

    rules = []
    for s in targets:
        hr(f"STEP 3: rules from {name(s)}, every split into lhs -> rhs, {2 ** len(s) - 2} rules, conf = sup(lhs union rhs) / sup(lhs)")
        rules += gen_rules(s, sup, n, ms, mc, a.min_lift)
    result(rules, n, a.min_lift)

if __name__ == "__main__": main()
