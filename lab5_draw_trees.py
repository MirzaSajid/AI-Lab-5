"""Lab 5 (optional) - draw the two supplied game trees as outputs/trees.png.

Needs matplotlib (the only third-party package in this lab; the report figure
was made with it). Shows every node's minimax value, MAX (up triangle) and MIN
(down triangle) layers, and, for each tree, the subtree that alpha-beta skips
(dashed grey) in the original left-to-right order.
Run: python lab5_draw_trees.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab5_minimax_ab import alphabeta_counted, bottom_up, tree_depth


def draw(ax, scores, title):
    depth = tree_depth(scores)
    levels = {d: vals for d, _, vals, _ in bottom_up(scores)}
    prunes = alphabeta_counted(scores)["prunes"]
    skipped = {i for p in prunes for i in p["skipped"]}
    pos = {}
    n = len(scores)
    for i in range(n):
        pos[(depth, i)] = (i, 0)
    for d in range(depth - 1, -1, -1):
        for k in range(2 ** d):
            x = (pos[(d + 1, 2 * k)][0] + pos[(d + 1, 2 * k + 1)][0]) / 2
            pos[(d, k)] = (x, depth - d)

    def pruned(d, k):                       # node lies fully inside a skipped block
        span = 2 ** (depth - d)
        return all(i in skipped for i in range(k * span, (k + 1) * span))

    for d in range(depth):
        for k in range(2 ** d):
            for c in (2 * k, 2 * k + 1):
                (x1, y1), (x2, y2) = pos[(d, k)], pos[(d + 1, c)]
                cut = pruned(d + 1, c)
                ax.plot([x1, x2], [y1, y2], color="#999999" if cut else "#333333",
                        ls="--" if cut else "-", lw=1.2, zorder=1)
    for (d, k), (x, y) in pos.items():
        cut = pruned(d, k)
        val = levels[d][k]
        face = "#eeeeee" if cut else ("#cfe2ff" if d % 2 == 0 and d < depth else
                                      "#ffe5b4" if d < depth else "#d9f2d9")
        edge = "#999999" if cut else "#222222"
        marker = "s" if d == depth else ("^" if d % 2 == 0 else "v")
        ax.scatter([x], [y], s=900, marker=marker, c=face, edgecolors=edge,
                   linewidths=1.4, zorder=2, linestyle="--" if cut else "-")
        ax.text(x, y - 0.02, str(val), ha="center", va="center", fontsize=11,
                fontweight="bold", color="#888888" if cut else "#111111", zorder=3)
    for i in range(n):
        ax.text(i, -0.42, f"L{i}", ha="center", fontsize=8, color="#555555")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.7, depth + 0.5)
    ax.axis("off")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.2))
    draw(axes[0], [3, 5, 2, 9, 0, 7, 0, 0],
         "Tree 5.1 (6 scores zero-padded to 8): root = 5, Left. Alpha-beta prunes nothing.")
    draw(axes[1], [3, 5, 2, 9, 3, 5, 2, 9],
         "Tree 5.2: root = 5, Left. Alpha-beta skips leaves L6, L7 (dashed grey).")
    fig.text(0.5, 0.005, "Up triangle = MAX node, down triangle = MIN node, square = leaf utility. "
             "Numbers are minimax values.", ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig("outputs/trees.png", dpi=170)
    print("saved outputs/trees.png")


if __name__ == "__main__":
    main()
