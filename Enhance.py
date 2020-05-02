from copy import deepcopy
from collections import Counter
from typing import List

import UDLib as U


def replace_relation(tree: U.UDTree, old_relation: str, new_relation: str):
    result = deepcopy(tree)
    for key in result.keys:
        # Replace in the nodes
        node = result.nodes[key]
        if node.DEPREL.startswith(old_relation):
            result.nodes[key].DEPREL = new_relation
        # Replace in the edges
        for i, edge in enumerate(result.graph[key]):
            if edge.relation.startswith(old_relation):
                result.graph[key][i].relation = new_relation
    return result


def children_of(trees: List[U.UDTree], relation: str):
    result = Counter()
    for tree in trees:
        for k, node in tree.nodes.items():
            if node.DEPREL == relation:
                for edge in tree.graph[k]:
                    if edge.directionality == 'down':
                        result[tree.nodes[edge.head].DEPREL] += 1
    return result


if __name__ == '__main__':
    import sys
    inp_path = sys.argv[1]
    out_path = sys.argv[2]

    # Collapse all types of nominal modification into one supertype
    trees = U.conllu2trees(inp_path)
    # compound -> nmod
    # nmod -> acl
    # amod -> acl
    result = [
            replace_relation(
                replace_relation(
                    replace_relation(tree, 'compound', 'nmod'),
                    'nmod', 'acl'),
                'amod', 'acl')
            for tree in trees
    ]
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write('\n\n'.join(str(tree) for tree in result))
