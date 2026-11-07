"""Lab 5 - edge cases and evidence over many random trees.

Part 1  edge cases: a single leaf, two leaves, all-equal leaves, negative
        utilities, and the effect of zero-padding the manual's 6-score tree.
Part 2  2000 random trees (depth 1-5, utilities -9..9 so ties are common, fixed
        seed): alpha-beta must return the same root value and root action as
        minimax and never evaluate more leaves.
All trees are synthetic. Exit code 0 if every check passes, else 1.
Run: python lab5_tests.py   (needs the other lab5_*.py files in the same folder)
"""
import itertools
import random
import sys

from lab5_minimax_ab import alphabeta_counted, minimax_counted
from lab5_ordering import apply_mask, build, flatten

results = []


def check(name, condition):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'} | {name}")


def nested_value(node, depth=0):
    """Minimax on an arbitrary nested tree (nodes may have any number of children)."""
    if not isinstance(node, list):
        return node
    vals = [nested_value(c, depth + 1) for c in node]
    return max(vals) if depth % 2 == 0 else min(vals)


def main():
    # ---------------------------------------------------------------- Part 1
    print("== Part 1: edge cases ==")
    mm, ab = minimax_counted([7]), alphabeta_counted([7])
    check("single leaf [7]: both return 7 after evaluating 1 leaf",
          mm["value"] == ab["value"] == 7 and len(mm["leaves"]) == len(ab["leaves"]) == 1)
    check("single leaf: no root action exists", mm["action"] is None and ab["action"] is None)

    mm, ab = minimax_counted([4, 9]), alphabeta_counted([4, 9])
    check("two leaves [4, 9]: MAX picks 9 (Right), both leaves evaluated",
          ab["value"] == 9 and ab["action"] == "Right" and len(ab["leaves"]) == 2)

    mm, ab = minimax_counted([5] * 8), alphabeta_counted([5] * 8)
    print("info | all-equal leaves: alpha-beta evaluates", len(ab["leaves"]), "of 8")
    check("all-equal leaves: same value 5, tie goes to Left, no extra work",
          mm["value"] == ab["value"] == 5 and ab["action"] == "Left"
          and len(ab["leaves"]) <= len(mm["leaves"]))

    neg = [-3, -5, -2, -9, -1, -7, -4, -6]
    mm, ab = minimax_counted(neg), alphabeta_counted(neg)
    check("negative utilities: values agree", mm["value"] == ab["value"])

    # zero-padding of the manual's own tree: 6 scores -> 8 leaves
    real = [3, 5, 2, 9, 0, 7]
    padded = real + [0, 0]
    pm, pa = minimax_counted(padded), alphabeta_counted(padded)
    unpadded = [[[3, 5], [2, 9]], [[0, 7]]]       # the missing branch simply does not exist
    unpadded_value = nested_value(unpadded)
    print(f"info | zero-padded tree {padded}: root value {pm['value']}, action {pm['action']}")
    print(f"info | same 6 scores with NO dummy branch: root value {unpadded_value}, "
          "action Right")
    check("padding assumption matters: padded root 5 (Left) vs. unpadded root 7 (Right)",
          pm["value"] == 5 and pm["action"] == "Left" and unpadded_value == 7)

    # ---------------------------------------------------------------- Part 2
    print()
    print("== Part 2: 2000 random trees (seed 5) ==")
    rng = random.Random(5)
    bad_value = bad_action = more_leaves = pruned_trees = 0
    leaves_mm = leaves_ab = 0
    by_depth = {}
    for depth in range(1, 6):
        mm_tot = ab_tot = 0
        for _ in range(400):
            scores = [rng.randint(-9, 9) for _ in range(2 ** depth)]
            mm, ab = minimax_counted(scores), alphabeta_counted(scores)
            bad_value += mm["value"] != ab["value"]
            bad_action += mm["action"] != ab["action"]
            more_leaves += len(ab["leaves"]) > len(mm["leaves"])
            pruned_trees += len(ab["leaves"]) < len(mm["leaves"])
            mm_tot += len(mm["leaves"])
            ab_tot += len(ab["leaves"])
        by_depth[depth] = (mm_tot, ab_tot)
        leaves_mm += mm_tot
        leaves_ab += ab_tot
    print("depth | minimax leaves (400 trees) | alpha-beta leaves | alpha-beta / minimax")
    for depth, (m, a) in by_depth.items():
        print(depth, m, a, f"{a / m:.2f}", sep=" | ")
    print(f"total | {leaves_mm} | {leaves_ab} | {leaves_ab / leaves_mm:.2f}")
    print(f"trees where alpha-beta evaluated fewer leaves: {pruned_trees} of 2000")
    check("alpha-beta root value equals minimax on all 2000 trees", bad_value == 0)
    check("alpha-beta root action equals minimax action on all 2000 trees", bad_action == 0)
    check("alpha-beta never evaluates more leaves than minimax", more_leaves == 0)
    check("alpha-beta evaluated fewer leaves in total", leaves_ab < leaves_mm)

    # ordering invariance on random depth-3 trees (all 128 swap combinations)
    invariant = True
    for _ in range(100):
        scores = [rng.randint(-9, 9) for _ in range(8)]
        ref = minimax_counted(scores)["value"]
        nested = build(scores)
        for bits in itertools.product((0, 1), repeat=7):
            order = flatten(apply_mask(nested, iter(bits)))
            if alphabeta_counted(order)["value"] != ref:
                invariant = False
    check("root value is unchanged by every branch ordering "
          "(100 random trees x 128 orderings)", invariant)

    passed = sum(results)
    print()
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
