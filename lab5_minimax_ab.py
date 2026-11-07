"""Lab 5 - Tasks 1 and 2: Minimax and alpha-beta pruning with leaf counters.

Game trees are complete binary trees stored as a list of leaf utilities: node
(depth, index) has children (depth + 1, index * 2) and (depth + 1, index * 2 + 1).
Depth 0 is a MAX node, depths alternate MAX / MIN / MAX ... and the leaves sit
at depth = log2(number of leaves). Children are searched left to right.

minimax() and alphabeta() are the supplied reference functions (manual
Sections 5.1 and 5.2), kept unchanged apart from layout. minimax_counted() and
alphabeta_counted() follow exactly the same recursion but also count every
leaf evaluation, log every pruning decision and record the value of each root
child, so the chosen root action can be reported.

Counting convention: a leaf is "evaluated" each time the recursion reaches
depth == target and returns scores[index]. Root tie-break: the FIRST child
(Left) wins ties.
Run: python lab5_minimax_ab.py
"""
import math

INF = float("inf")


# ---- supplied reference code (manual Sections 5.1 and 5.2) ----------------
def minimax(curDepth, nodeIndex, maxTurn, scores, targetDepth):
    if curDepth == targetDepth:
        return scores[nodeIndex]
    if maxTurn:
        return max(
            minimax(curDepth + 1, nodeIndex * 2, False, scores, targetDepth),
            minimax(curDepth + 1, nodeIndex * 2 + 1, False, scores, targetDepth)
        )
    else:
        return min(
            minimax(curDepth + 1, nodeIndex * 2, True, scores, targetDepth),
            minimax(curDepth + 1, nodeIndex * 2 + 1, True, scores, targetDepth)
        )


def alphabeta(depth, index, maxTurn, scores, target, alpha, beta):
    if depth == target:
        return scores[index]
    if maxTurn:
        value = float('-inf')
        for i in range(2):
            value = max(value, alphabeta(depth + 1, index * 2 + i, False, scores, target, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for i in range(2):
            value = min(value, alphabeta(depth + 1, index * 2 + i, True, scores, target, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


# ---- instrumented versions -------------------------------------------------
def tree_depth(scores):
    n = len(scores)
    if n < 1 or n & (n - 1):
        raise ValueError("number of leaves must be a power of two")
    return int(math.log2(n))


def pick_action(child_values):
    """First child with the strictly largest value (ties -> Left)."""
    best_i, best_v = None, -INF
    for i, v in enumerate(child_values):
        if best_i is None or v > best_v:
            best_i, best_v = i, v
    return best_i


def action_name(i):
    return None if i is None else ("Left" if i == 0 else "Right")


def minimax_counted(scores):
    target = tree_depth(scores)
    leaves, root_children = [], []

    def rec(depth, index, max_turn):
        if depth == target:
            leaves.append(index)
            return scores[index]
        results = [rec(depth + 1, index * 2 + i, not max_turn) for i in range(2)]
        if depth == 0:
            root_children.extend(results)
        return max(results) if max_turn else min(results)

    value = rec(0, 0, True)
    action = pick_action(root_children) if root_children else None
    return dict(value=value, leaves=leaves, prunes=[], action=action_name(action),
                total=len(scores))


def alphabeta_counted(scores):
    target = tree_depth(scores)
    leaves, prunes, root_children = [], [], []

    def rec(depth, index, max_turn, alpha, beta):
        if depth == target:
            leaves.append(index)
            return scores[index]
        span = 2 ** (target - depth - 1)          # leaves under one child
        value = -INF if max_turn else INF
        for i in range(2):
            child = rec(depth + 1, index * 2 + i, not max_turn, alpha, beta)
            if depth == 0:
                root_children.append(child)
            if max_turn:
                value = max(value, child)
                alpha = max(alpha, value)
            else:
                value = min(value, child)
                beta = min(beta, value)
            if alpha >= beta:
                if i == 0:                        # the second child is skipped
                    first = (index * 2 + 1) * span
                    prunes.append(dict(depth=depth, index=index,
                                       kind="MAX" if max_turn else "MIN",
                                       alpha=alpha, beta=beta, value=value,
                                       skipped=list(range(first, first + span))))
                break
        return value

    value = rec(0, 0, True, -INF, INF)
    action = pick_action(root_children) if root_children else None
    return dict(value=value, leaves=leaves, prunes=prunes,
                action=action_name(action), total=len(scores))


def bottom_up(scores):
    """Minimax values level by level, from the leaves to the root."""
    target = tree_depth(scores)
    levels = [(target, "leaf", list(scores), [f"L{i} = {s}" for i, s in enumerate(scores)])]
    values = list(scores)
    for depth in range(target - 1, -1, -1):
        is_max = depth % 2 == 0
        fn = max if is_max else min
        calcs, new = [], []
        for k in range(0, len(values), 2):
            v = fn(values[k], values[k + 1])
            calcs.append(f"{'max' if is_max else 'min'}({values[k]}, {values[k + 1]}) = {v}")
            new.append(v)
        levels.append((depth, "MAX" if is_max else "MIN", new, calcs))
        values = new
    return levels


def order_text(idx):
    return ", ".join(str(i) for i in idx) if idx else "-"


def print_trace(title, scores):
    print(title)
    print("depth | type | calculations")
    for depth, kind, _, calcs in bottom_up(scores):
        print(depth, kind, "; ".join(calcs), sep=" | ")
    root = bottom_up(scores)[-1][2][0]
    res = minimax_counted(scores)
    print(f"Root value = {root}; root action = {res['action']} (first child wins ties)")
    print()


def main():
    # Supplied tree of Section 5.1 (6 scores padded with zeros to 8 leaves)
    scores51 = [3, 5, 2, 9, 0, 7]
    depth51 = math.ceil(math.log2(len(scores51)))
    scores51 = scores51 + [0] * (2 ** depth51 - len(scores51))
    # Supplied tree of Section 5.2
    scores52 = [3, 5, 2, 9, 3, 5, 2, 9]
    depth52 = int(math.log2(len(scores52)))

    print("Supplied code, as given in the manual")
    print("5.1 minimax on", scores51, "->", minimax(0, 0, True, scores51, depth51))
    print("5.2 alphabeta on", scores52, "->",
          alphabeta(0, 0, True, scores52, depth52, float('-inf'), float('inf')))
    print()

    print_trace("Task 1: Minimax bottom-up trace, tree 5.1 "
                f"(6 scores zero-padded to {scores51})", scores51)
    print_trace(f"Task 1: Minimax bottom-up trace, tree 5.2 {scores52}", scores52)

    trees = [("5.1 (padded)", scores51), ("5.2", scores52)]
    print("Task 2: leaf-evaluation counters")
    print("tree | algorithm | root value | root action | leaves evaluated | evaluation order (leaf indices)")
    results = {}
    for name, sc in trees:
        mm, ab = minimax_counted(sc), alphabeta_counted(sc)
        results[name] = (mm, ab)
        for algo, r in (("Minimax", mm), ("Alpha-beta", ab)):
            print(name, algo, r["value"], r["action"],
                  f"{len(r['leaves'])} of {r['total']}", order_text(r["leaves"]), sep=" | ")
    print()

    print("Alpha-beta pruning decisions")
    for name, _ in trees:
        mm, ab = results[name]
        if not ab["prunes"]:
            print(f"tree {name}: no pruning occurred (all {ab['total']} leaves evaluated)")
        for p in ab["prunes"]:
            print(f"tree {name}: pruned at {p['kind']} node (depth {p['depth']}, index {p['index']}) "
                  f"after its first child: alpha = {p['alpha']}, beta = {p['beta']} "
                  f"(alpha >= beta); skipped leaf indices {order_text(p['skipped'])}")
    print()

    print("Verification")
    for name, _ in trees:
        mm, ab = results[name]
        print(f"tree {name}: same root value = {mm['value'] == ab['value']}; "
              f"same root action = {mm['action'] == ab['action']}; "
              f"alpha-beta leaves {len(ab['leaves'])} <= minimax leaves {len(mm['leaves'])}: "
              f"{len(ab['leaves']) <= len(mm['leaves'])}")


if __name__ == "__main__":
    main()
