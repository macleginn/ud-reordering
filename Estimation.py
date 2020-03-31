from collections import defaultdict, Counter
from itertools import combinations
from typing import List
from pprint import pprint

import UDLib
import z3


def no_preference(estimate_dict, rel1, rel2):
    if (rel1, rel2) in estimate_dict:
        return abs(estimate_dict[(rel1, rel2)] - 0.5) < 0.2
    elif (rel2, rel1) in estimate_dict:
        return abs(estimate_dict[(rel2, rel1)] - 0.5) < 0.2
    else:
        raise ValueError(f"An unknown pair of relations: {rel1}, {rel2}")


def is_bigger(estimate_dict, rel1, rel2):
    if (rel1, rel2) in estimate_dict:
        return estimate_dict[(rel1, rel2)] > 0
    elif (rel2, rel1) in estimate_dict:
        return estimate_dict[(rel2, rel1)] < 0
    else:
        raise ValueError(f"An unknown pair of relations: {rel1}, {rel2}")


def check_expansion(deprel, estimates):
    '''Try to satisfy the learned ordering constraints
    by assigning linear indices to UD relations. If successful,
    return the indices; return None otherwise.'''

    s = z3.Solver()

    # Create variables for all participating relations.
    # Index them by their names.
    all_rels = set()
    for rel1, rel2 in estimates:
        all_rels.add(rel1)
        all_rels.add(rel2)
    var_dict = {
        rel: z3.Ints(rel)[0] for rel in all_rels
    }

    # Add learned constraints in the order of decreasing magnitude.
    ordered_constraints = list(estimates.items())
    ordered_constraints.sort(key = lambda x: abs(x[1]), reverse=True)
    for (i, ((rel1, rel2), num)) in enumerate(ordered_constraints):
        if rel1 == rel2:
            continue
        a = var_dict[rel1]
        b = var_dict[rel2]
        s.push()
        if num > 0:
            s.add(a > b)
        else:
            s.add(a < b)
        if s.check() == z3.unsat:
            print(f"{deprel:>10}: {i} most frequent constraints satisfiable.")
            s.pop()
            break
    else:
        print(f"{deprel:>10}: all constraints were satisfied.")
    s.check()
    model = s.model()
    ordering_dict = {
        rel: model[var_dict[rel]] for rel in all_rels
    }
    return ordering_dict


def ordering_is_valid(estimate_dict):
    '''Checks that a valid ordering was learned from the data
    by trying to find indices satysfying ordering constraints
    from the data. Dict[deprel: str,Dict[deprel: str, index: int]]
    is returned if the ordering is valid. None is returned otherwise.'''

    indices_dict = {}
    for deprel in estimate_dict:
        indices_for_deprel = check_expansion(deprel, estimate_dict[deprel])
        if indices_for_deprel is None:
            return None
        else:
            indices_dict[deprel] = indices_for_deprel
    return indices_dict


def process_tree(UD_tree: UDLib.UDTree,
                 UD_root: str,
                 main_clause: bool,
                 counter_dict):
    # Can't determine the ordering when virtual nodes are present.
    # Give up here if the root is a virtual node and exclude them from
    # the node's children later.
    if '.' in UD_root:
        return

    # Each head node has its own ordering of children
    root_rel = UD_tree.nodes[UD_root].DEPREL.split(':')[0]
    if root_rel not in counter_dict:
        counter_dict[root_rel] = defaultdict(int)
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
            local_counter_dict[key] -= 1
        else:
            # Alphabetical order was "flipped"
            local_counter_dict[key] += 1

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
    return counter_dict