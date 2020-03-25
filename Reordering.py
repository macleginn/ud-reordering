from functools import cmp_to_key
from typing import List
from copy import deepcopy

import UDLib as U


def print_tree_w_ordering(UD_tree: U.UDTree, ordering: List[str]):
    for idx in ordering:
        print(UD_tree.nodes[idx].FORM.lower(), end = ' ')
    print()


def make_cmp_func(UD_tree: U.UDTree, comparator_dict, parent_deprel):
    def dict_based_comparator(x, y):
        x = UD_tree.nodes[x].DEPREL.split(':')[0]
        y = UD_tree.nodes[y].DEPREL.split(':')[0]
        if x == y:
            return 0
        # Relations in key pairs go in alphabetical order.
        if x < y:
            multiplier = 1
        else:
            multiplier = -1
            x, y = y, x
        try:
            if comparator_dict[parent_deprel.split(':')[0]][(x,y)] > 0.5:
                return 1*multiplier
            else:
                return -1*multiplier
        except KeyError:
            # Fall back on the root dictionary
            if (x,y) in comparator_dict['root']:
                if comparator_dict['root'][(x,y)] > 0.5:
                    return 1*multiplier
                else:
                    return -1*multiplier
            else:
                # We don't know how to align these
                return 0
    return dict_based_comparator


def reorder_rec(input_tree: U.UDTree, comparator_dict, root_idx, output_arr):
    root_deprel = input_tree.nodes[root_idx].DEPREL
    key_func = cmp_to_key(
        make_cmp_func(input_tree, comparator_dict, root_deprel))
    nodes = input_tree.get_node_children(root_idx)
    if not nodes:
        # Emit the leave
        output_arr.append(root_idx)
        return
    if root_deprel == 'punct':
        raise ValueError(f"The punctuation node {root_idx} has children!")
    nodes.append(root_idx)

    # Leaving this nice hack here for possible future use.
    # good, bad = [], []
    # for x in mylist:
    #     (bad, good)[x in goodvals].append(x)
    # https://stackoverflow.com/a/12135169/3665134
    punct_idx = [el for el in nodes if input_tree.nodes[el].DEPREL == 'punct']
    nodes     = [el for el in nodes if input_tree.nodes[el].DEPREL != 'punct']
    nodes.sort(key=key_func)
    # Place punctuation to where it was relative to the root node
    # before sorting.
    for p_i in punct_idx:
        root_pos = nodes.index(root_idx)
        if int(p_i) < int(root_idx):
            nodes.insert(root_pos, p_i)
        else:
            nodes.insert(root_pos+1, p_i)
    # Process nodes in their linear order
    for n in nodes:
        if n == root_idx:
            # Emit the root node
            output_arr.append(root_idx)
        else:
            # Process subtrees
            reorder_rec(input_tree, comparator_dict, n, output_arr)


def reorder(input_tree: U.UDTree, comparator_dict):
    '''Reorders the constituents in the input tree in a top-bottom
    fashion based on the ordering from the comparator dict. Punctuation
    is placed to the left or to the right from the host depending on where
    it used to be positioned originally.'''

    # Do DFS on the tree ordering the nodes at each level, traversing
    # the children from left to right and emitting at the lower level.
    new_order = []
    root_idx = input_tree.get_real_root()
    reorder_rec(input_tree, comparator_dict, root_idx, new_order)

    # Transform the tree based on the new ordering.
    new_tree = deepcopy(input_tree)
    old_order = input_tree.keys
    idx_map = {
        old_idx: new_idx for old_idx, new_idx in zip(old_order, new_order)
    }


    return new_order