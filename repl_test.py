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

data_dir = pathlib.Path('data')
ouput_dir = pathlib.Path(data_dir / 'reordered')

# English data
trees_en_ewt = conllu2trees(data_dir / 'en_ewt-ud-train.conllu')
trees_en_ewt_dev = conllu2trees(data_dir / 'en_ewt-ud-dev.conllu')
trees_en_ewt_test = conllu2trees(data_dir / 'en_ewt-ud-test.conllu')
trees_en_pud = conllu2trees(data_dir / 'en_pud-ud-test.conllu')

# Learn Japanese ordering and apply it to English.
trees_ja = conllu2trees(data_dir / 'ja_pud-ud-test.conllu')
estimates_ja = E.get_ml_directionality_estimates(trees_ja)
assert E.ordering_is_valid(estimates_ja)

for i, treebank in enumerate([
    trees_en_ewt_dev,
    trees_en_ewt_test
]):
    reordered = []
    for tree in treebank:
        try:
            reordered.append(R.reorder(tree, estimates_ja))
        except ValueError:
            pprint(tree.id_lines[-1])
            continue
    with open(ouput_dir / f'en_{["ewt_dev", "ewt_test"][i]}_reordered.conllu', 'w') as out:
        out.write('\n\n'.join(str(t) for t in reordered))

# Learn Arabic ordering and apply it to English.
trees_ar = conllu2trees(data_dir / 'ar_pud-ud-test.conllu')
estimates_ar = E.get_ml_directionality_estimates(trees_ar)
assert E.ordering_is_valid(estimates_ar)

for i, treebank in enumerate([
    trees_en_ewt,
    trees_en_ewt_dev,
    trees_en_ewt_test
]):
    reordered = []
    for tree in treebank:
        try:
            reordered.append(R.reorder(tree, estimates_ar))
        except ValueError:
            pprint(tree.id_lines[-1])
            continue
    with open(ouput_dir / f'en_{["ewt_train", "ewt_dev", "ewt_test"][i]}_reordered_ar.conllu', 'w') as out:
        out.write('\n\n'.join(str(t) for t in reordered))

