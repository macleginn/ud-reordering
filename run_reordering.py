import sys
import json

from multiprocessing import Pool, cpu_count
from datetime import datetime

import UDLib as U
import ReorderingNew as R


if len(sys.argv) != 4:
    print('''
Example usage:
python3 run_reordering.py data/en_ewt-ud-test.conllu data/estimates/japanese_pud.json data/reordered/en_ewt-ud-test-reordered.conllu
''')
    sys.exit(1)

input_corpus_path = sys.argv[1]
estimates_path = sys.argv[2]
output_path = sys.argv[3]

input_trees = U.conllu2trees(input_corpus_path)
with open(estimates_path, 'r', encoding='utf-8') as inp:
    estimates = json.load(inp)


# Capture estimates from the global scope for mapping
def reorder_without_exceptions(tree):
    try:
        result = R.reorder_tree(tree, estimates)
    except Exception as e:
        print(f'Reordering failed : {e}')
        result = None
    return result


reordered_trees = []

n_cpus = cpu_count()
print(f'{n_cpus} CPUs are available.')
time_start = datetime.now()
with Pool(n_cpus) as p:
    for i, t_r in enumerate(p.map(
        reorder_without_exceptions,
        input_trees,
        len(input_trees) // n_cpus
    )):
        if t_r is not None:
            reordered_trees.append(t_r)
print()
print(f'{datetime.now() - time_start} seconds.')

with open(output_path, 'w', encoding='utf-8') as out:
    print('\n\n'.join(str(t) for t in reordered_trees), file=out)
