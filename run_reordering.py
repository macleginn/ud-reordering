import sys
import json

import UDLib as U
import ReorderingNew as R

if len(sys.argv) != 4:
    print('''
Example usage:
python3 run_reordering.py data/en_ewt-ud-test.conllu data/estimates/japanese_pud.json data/reordered/en_ewt-ud-test-reordered.conllu
''')
    sys.exit(1)

input_corpus_path = sys.argv[1]
estimates_path    = sys.argv[2]
output_path       = sys.argv[3]

input_trees = U.conllu2trees(input_corpus_path)
with open(estimates_path, 'r', encoding='utf-8') as inp:
    estimates = json.load(inp)
reordered_trees = []
for i, t in enumerate(input_trees):
    try:
        reordered_trees.append(R.reorder_tree(t, estimates))
    except Exception as e:
        print(f'Reordering failed on tree #{i}: {e}')
with open(output_path, 'w', encoding='utf-8') as out:
    print('\n\n'.join(str(t) for t in reordered_trees), file = out)
