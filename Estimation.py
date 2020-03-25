from collections import defaultdict, Counter
from itertools import combinations
from typing import List
from pprint import pprint

import UDLib


def ordering_is_valid(estimate_dict):
    '''Checks that a valid ordering was learned from the data.'''
    is_bigger  = lambda deprel, x, y: estimate_dict[deprel][(x, y)] > 0.5
    for deprel in estimate_dict:
        ordered_pairs = []
        for el1, el2 in estimate_dict[deprel]:
            if is_bigger(deprel, el1, el2):
                ordered_pairs.append((el2, el1))
            else:
                ordered_pairs.append((el1, el2))
    for x1, y1 in ordered_pairs:
        if x1 == y1:
            continue
        for x2, y2 in ordered_pairs:
            if x2 == y2:
                continue
            if y1 == x2:
                # If a goes before b and b goes before c,
                # a should go before c, i.e. there should be
                # a pair (a, c) in the list.
                if (x1, y2) not in ordered_pairs:
                    return False
    return True


def process_tree(UD_tree: UDLib.UDTree, UD_root: str, main_clause: bool, counter_dict):
    # Can't determine the ordering when virtual nodes are present.
    # Give up here if the root is a virtual node and exclude them from
    # the node's children later.
    if '.' in UD_root:
        return

    # Each head node has its own ordering of children
    root_rel = UD_tree.nodes[UD_root].DEPREL.split(':')[0]
    if root_rel not in counter_dict:
        counter_dict[root_rel] = defaultdict(Counter)
    local_counter_dict = counter_dict[root_rel]

    # Not calling this 'children' because we'll have to append
    # the root node to this list.
    nodes = [
        el for el in UD_tree.get_node_children(UD_root) \
        if '.' not in el \
        and UD_tree.nodes[el].DEPREL != 'punct'
    ]
    # If no children, nothing to count in this subtree.
    if not nodes:
        return

    # A node is on the same level as its immediate children as regards
    # linear order.
    nodes.append(UD_root)
    nodes.sort(key=int)
    for idx1, idx2 in combinations(nodes, 2):
        # Remove language-specific refinements
        deprel1 = UD_tree.nodes[idx1].DEPREL.split(':')[0]
        deprel2 = UD_tree.nodes[idx2].DEPREL.split(':')[0]
        key = tuple(sorted([deprel1, deprel2]))
        if key == (deprel1, deprel2):
            # Alphetical order preserved in the tree
            local_counter_dict[key][0] += 1
        else:
            # Alphabetical order was "flipped"
            local_counter_dict[key][1] += 1

    # Recurse
    for child_idx in nodes:
        if child_idx == UD_root:
            continue
        process_tree(
            UD_tree,
            child_idx,
            main_clause,
            counter_dict)


def get_ml_directionality_estimates(UD_trees: List[UDLib.UDTree]):
    counter_dict = {}
    for UD_tree in UD_trees:
        UD_root = UD_tree.get_real_root()
        process_tree(
            UD_tree,
            UD_root,
            True,
            counter_dict)
    # pprint(counter_dict_main)
    # pprint(counter_dict_dep)
    # return
    result_dict = {}
    for deprel, deprel_children_counter_dict in counter_dict.items():
        result_dict[deprel] = {}
        for rel_pair, counts in deprel_children_counter_dict.items():
            result_dict[deprel][rel_pair] = counts[1] / (sum(counts.values()))
    return result_dict