# Lab 5 - Adversarial Search: Minimax & Alpha-Beta Pruning

**Student:** Sajid Hussain (BSAIM-046) | **Course:** CAI601400 - Artificial Intelligence | **Lab:** 5 (CLO2)

Minimax and alpha-beta pruning on small complete binary game trees (MAX at the root,
alternating MAX/MIN layers, utilities at the leaves): a bottom-up hand trace, leaf-
evaluation counters for both algorithms, pruning decisions, and the effect of
reordering branches. The leaf utilities are the manual's synthetic examples.

Conventions: children are searched left to right; a leaf is "evaluated" each time the
recursion reaches it; the root tie-break picks the **first** (Left) child.
The manual's 5.1 tree has 6 scores that are **zero-padded** to 8 leaves; this padding is
an assumption that changes the answer (see the report, Section 8).

## Setup
Python 3.8+ (tested on 3.12). The four scripts below use the standard library only; no
virtual environment is needed.

```bash
python --version
```
`lab5_draw_trees.py` is optional and needs matplotlib (`pip install matplotlib`); it only
redraws `outputs/trees.png`.

## Run
```bash
python lab5_minimax_ab.py   # Tasks 1-2: bottom-up trace, leaf counters, pruning log
python lab5_ordering.py     # Task 3: named orderings + all 128 child-swap orderings
python lab5_tests.py        # edge cases + 2000 random trees (exit code 0 = all pass)
python lab5_draw_trees.py   # optional: redraw outputs/trees.png
```
Keep all `lab5_*.py` files in the same folder: the other scripts import
`lab5_minimax_ab.py`.

## Files
| File | Purpose |
|---|---|
| `lab5_minimax_ab.py` | Supplied minimax/alphabeta, counted versions, traces |
| `lab5_ordering.py` | Branch-reordering experiments (Task 3) |
| `lab5_tests.py` | Edge cases, padding check, 2000 random trees |
| `lab5_draw_trees.py` | Optional figure of both trees (matplotlib) |
| `outputs/*.txt`, `outputs/trees.png` | Captured outputs and tree figure |
| `Lab5_Report.docx` | Full report: traces, counts, ordering, analysis, reflection |

## Reproducing the outputs
```bash
python lab5_minimax_ab.py > outputs/minimax_ab_output.txt
python lab5_ordering.py   > outputs/ordering_output.txt
python lab5_tests.py      > outputs/tests_output.txt
```

## Sources
`minimax()` and `alphabeta()` in `lab5_minimax_ab.py` are the supplied reference
functions from the Lab 5 manual (Sections 5.1-5.2). All other code is my own. Textbook
and paper sources are listed in the References section of `Lab5_Report.docx`. The random
trees use synthetic data from a fixed seed. No credentials, private data or virtual
environments are included.
