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


def report_reordering_progress(treebank, estimates):
    importlib.reload(R)
    for i, t in enumerate(treebank):
        print(i)
        try:
            _ = R.reorder(t, estimates, True)
        except ValueError as e:
            print(e)
            continue


data_dir = pathlib.Path('data')
ouput_dir = pathlib.Path(data_dir / 'reordered')

# English data
trees_en_ewt_train = conllu2trees(data_dir / 'en_ewt-ud-train.conllu')
trees_en_ewt_dev = conllu2trees(data_dir / 'en_ewt-ud-dev.conllu')
trees_en_ewt_test = conllu2trees(data_dir / 'en_ewt-ud-test.conllu')
trees_en_pud = conllu2trees(data_dir / 'en_pud-ud-test.conllu')

# Learn Japanese ordering and apply it to English.
trees_ja = conllu2trees(data_dir / 'ja_pud-ud-test.conllu')
estimates_ja = E.get_ml_directionality_estimates(trees_ja)
# ordering_ja = E.ordering_is_valid(estimates_ja)

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


# Learn German ordering and apply it to English.
trees_de = conllu2trees(data_dir / 'de_gsd-ud-train.conllu')
estimates_de = E.get_ml_directionality_estimates(trees_de)

for i, treebank in enumerate([
    trees_en_ewt_train,
    trees_en_ewt_dev,
    trees_en_ewt_test
]):
    reordered = []
    for tree in treebank:
        try:
            reordered.append(R.reorder(tree, estimates_de))
        except ValueError:
            pprint(tree.id_lines[-1])
            continue
    with open(ouput_dir / f'en_{["ewt_train", "ewt_dev", "ewt_test"][i]}_reordered_de_new.conllu', 'w') as out:
        out.write('\n\n'.join(str(t) for t in reordered))


# Learn Arabic ordering and apply it to English.
trees_ar = conllu2trees(data_dir / 'ar_pud-ud-test.conllu')
estimates_ar = E.get_ml_directionality_estimates(trees_ar)
report_reordering_progress(trees_en_ewt_train, estimates_ar)

for i, treebank in enumerate([
    trees_en_ewt_train,
    trees_en_ewt_dev,
    trees_en_ewt_test
]):
    reordered = []
    for tree in treebank:
        try:
            reordered.append(R.reorder(tree, estimates_ga))
        except ValueError:
            pprint(tree.id_lines[-1])
            continue
    with open(ouput_dir / f'en_{["ewt_train", "ewt_dev", "ewt_test"][i]}_reordered_ar_new.conllu', 'w') as out:
        out.write('\n\n'.join(str(t) for t in reordered))



for i, t in enumerate(trees_en_ewt_test):
    if 'nuclear' in str(t) and 'Iran' in str(t):
        print(i)

# Learn Irish ordering and apply it to English.
trees_ga = conllu2trees(data_dir / 'ga_idt_all.conllu')
estimates_ga = E.get_ml_directionality_estimates(trees_ga)
# report_reordering_progress(trees_en_ewt, estimates_ga)

importlib.reload(U)
for i in range(30, 41):
    trees_en_ewt_test[i].get_sentence()
    print("Irish:", R.reorder(trees_en_ewt_test[i], estimates_ga).get_sentence())
    print("Arabic:",R.reorder(trees_en_ewt_test[i], estimates_ar).get_sentence())
    print("Japanese:",R.reorder(trees_en_ewt_test[i], estimates_ja).get_sentence())
    print()

for i, t in enumerate(trees_en_ewt_train):
    print(i)
    try:
        _ = R.reorder(t, estimates_ga, True)
    except ValueError as e:
        print(e)
        continue

for i, treebank in enumerate([
    trees_en_ewt_train,
    trees_en_ewt_dev,
    trees_en_ewt_test
]):
    reordered = []
    for tree in treebank:
        try:
            reordered.append(R.reorder(tree, estimates_ga))
        except ValueError:
            pprint(tree.id_lines[-1])
            continue
    with open(ouput_dir / f'en_{["ewt_train", "ewt_dev", "ewt_test"][i]}_reordered_ga_new.conllu', 'w') as out:
        out.write('\n\n'.join(str(t) for t in reordered))

