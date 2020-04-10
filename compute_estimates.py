# Computes the estimates for corpora and dumps them into JSON files

import json
import importlib
import pathlib

import UDLib as U
import Estimation as E


def conllu2trees(path):
    with open(path, 'r', encoding='utf-8') as inp:
        txt = inp.read().strip()
        blocks = txt.split('\n\n')
    return [U.UDTree(*U.conll2graph(block)) for block in blocks]


def dump_estimates(est_dict, path):
    est_dict_str_keys = {}
    for deprel in est_dict:
        est_dict_str_keys[deprel] = {
            f'{r1}->{r2}': count for (r1, r2), count in est_dict[deprel].items()
        }
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(est_dict_str_keys, out, indent=2)


data_dir = pathlib.Path('data')
estimates_dir = pathlib.Path(data_dir / 'estimates')

# Japanese
trees_ja = conllu2trees(data_dir / 'ja_pud-ud-test.conllu')
estimates_ja = E.get_ml_directionality_estimates(trees_ja)
dump_estimates(estimates_ja, estimates_dir / 'japanese_pud.json')

# German
trees_de = conllu2trees(data_dir / 'de_gsd-ud-train.conllu')
estimates_de = E.get_ml_directionality_estimates(trees_de)
dump_estimates(estimates_de, estimates_dir / 'german_gsd.json')

# Arabic
trees_ar = conllu2trees(data_dir / 'ar_pud-ud-test.conllu')
estimates_ar = E.get_ml_directionality_estimates(trees_ar)
dump_estimates(estimates_ar, estimates_dir / 'arabic_pud.json')

# Irish
trees_ga = conllu2trees(data_dir / 'ga_idt_all.conllu')
estimates_ga = E.get_ml_directionality_estimates(trees_ga)
dump_estimates(estimates_ga, estimates_dir / 'irish_idt.json')
