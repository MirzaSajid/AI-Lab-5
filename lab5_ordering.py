"""Lab 5 - Task 3: reorder branches and observe how pruning changes.

Reordering the children of a node means swapping its left and right subtrees.
Because the trees are stored as leaf lists, this is done on a nested copy of
the tree and then flattened back to a leaf list (left-to-right leaf order), so
the unchanged supplied algorithms can be run on every ordering.

Part 1  four named orderings of each supplied tree: original, mirrored (every
        node swapped), best-first (MAX nodes try their largest child first, MIN
        nodes their smallest) and worst-first (the opposite).
Part 2  exhaustive check of ALL 2^7 = 128 swap combinations of each tree.
Only the order changes; the utilities are never changed. Minimax is order
independent, so its value and leaf count are used as the reference.
Run: python lab5_ordering.py   (needs lab5_minimax_ab.py in the same folder)
"""
import itertools
from collections import Counter

from lab5_minimax_ab import alphabeta_counted, minimax_counted, tree_depth

SCORES_51 = [3, 5, 2, 9, 0, 7, 0, 0]     # manual 5.1, 6 scores zero-padded to 8
SCORES_52 = [3, 5, 2, 9, 3, 5, 2, 9]     # manual 5.2


def build(scores):
    if len(scores) == 1:
        return scores[0]
    mid = len(scores) // 2
    return [build(scores[:mid]), build(scores[mid:])]


def flatten(node):
    return [node] if not isinstance(node, list) else flatten(node[0]) + flatten(node[1])


def value(node, depth):
    if not isinstance(node, list):
        return node
    vals = [value(c, depth + 1) for c in node]
    return max(vals) if depth % 2 == 0 else min(vals)


def apply_mask(node, bits):
    """Swap the children of every internal node whose bit is 1 (preorder bits)."""
    if not isinstance(node, list):
        return node
    swap = next(bits)
    left, right = apply_mask(node[0], bits), apply_mask(node[1], bits)
    return [right, left] if swap else [left, right]


def ordered(node, depth, best=True):
    """Best-first (or worst-first) ordering using the exact minimax values."""
    if not isinstance(node, list):
        return node
    kids = [ordered(c, depth + 1, best) for c in node]
    vals = [value(k, depth + 1) for k in kids]
    want_high = (depth % 2 == 0) == best      # MAX wants high first when best
    if (vals[0] < vals[1]) == want_high and vals[0] != vals[1]:
        kids.reverse()
    return kids


def run(scores):
    return minimax_counted(scores), alphabeta_counted(scores)


def main():
    for name, scores in (("5.1 (padded)", SCORES_51), ("5.2", SCORES_52)):
        nested = build(scores)
        n_internal = len(scores) - 1
        print(f"Tree {name}: leaves {scores}")
        print("ordering | leaf order | root value | root action | minimax leaves | "
              "alpha-beta leaves | skipped")
        variants = [
            ("original", nested),
            ("mirrored", apply_mask(nested, iter([1] * n_internal))),
            ("best-first", ordered(nested, 0, True)),
            ("worst-first", ordered(nested, 0, False)),
        ]
        values = set()
        for label, tree in variants:
            leaf_order = flatten(tree)
            mm, ab = run(leaf_order)
            values.update([mm["value"], ab["value"]])
            print(label, leaf_order, ab["value"], ab["action"], len(mm["leaves"]),
                  len(ab["leaves"]), len(mm["leaves"]) - len(ab["leaves"]), sep=" | ")
        print("Root value identical for all four orderings and both algorithms:",
              len(values) == 1)
        print()

        ref_value = next(iter(values))
        counts, same_value = Counter(), True
        total = 0
        for bits in itertools.product((0, 1), repeat=n_internal):
            leaf_order = flatten(apply_mask(nested, iter(bits)))
            mm, ab = run(leaf_order)
            same_value &= (mm["value"] == ab["value"] == ref_value)
            counts[len(ab["leaves"])] += 1
            total += 1
        mean = sum(k * v for k, v in counts.items()) / total
        print(f"All {total} child-swap orderings of tree {name}")
        print("alpha-beta leaves evaluated | number of orderings")
        for k in sorted(counts):
            print(k, counts[k], sep=" | ")
        print(f"fewest = {min(counts)}, most = {max(counts)}, mean = {mean:.2f} "
              f"(minimax always evaluates {len(scores)})")
        print("Root value unchanged in every ordering:", same_value)
        print()


if __name__ == "__main__":
    main()
