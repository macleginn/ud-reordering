import importlib
import pathlib
from pprint import pprint

import UDLib as U
import Estimation as E
import Reordering as R


def conllu2trees(path):
    with open(path, 'r') as inp:
        txt = inp.read().strip()
        blocks = txt.split('\n\n')
    return [U.UDTree(*U.conll2graph(block)) for block in blocks]


# Learn Japanese ordering and apply it to English.
data_dir = pathlib.Path('data')
trees_ja = conllu2trees(data_dir / 'ja_pud-ud-test.conllu')
trees_en = conllu2trees(data_dir / 'en_ewt-ud-train.conllu')

importlib.reload(U)
importlib.reload(E)

estimates_ja = E.get_ml_directionality_estimates(trees_ja)
assert E.ordering_is_valid(estimates_ja)

importlib.reload(R)
reordered = []
for tree in trees_en:
    try:
        reordered.append(R.reorder(tree, estimates_ja))
    except ValueError:
        pprint(tree.id_lines[-1])
        continue


