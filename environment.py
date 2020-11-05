# @Author: Jamil Hasibul <Jamil-h> <Jamil-h>
# @Date:   2020-10-02T21:51:53-05:00
# @Email:  mdhasibul.jamil@siu.edu
# @Filename: environment.py
# @Last modified by:   Jamil-h
# @Last modified time: 2020-10-03T14:59:45-05:00

import time
import gym
from gym import spaces
from treeEnvironment import *
LEAF_THRESHOLD=2




def building_tree(tree,dimension,cut_num):
    tree.print_nodes_to_cut()
    while(not tree.is_finish()):
        node=tree.get_current_node()
        if (tree.is_leaf(node)):
            node=tree.get_next_node()
        elif (tree.node_rules_same_in_given_dimension(node,dimension)):
            node=tree.get_next_node()
        else:
            tree.cut_current_node(dimension,cut_num)
        tree.print_nodes_to_cut()
        #time.sleep(2)
    print(tree)
    print(tree.compute_result())




def main():
    rules=load_rules_from_file("ruleFile")
    for rule in rules:
        print(rule)
    tree=Tree(rules,LEAF_THRESHOLD)
    building_tree(tree,0,2)
    tree=Tree(rules,LEAF_THRESHOLD)
    building_tree(tree,0,4)
    tree=Tree(rules,LEAF_THRESHOLD)
    building_tree(tree,1,2)
    tree=Tree(rules,LEAF_THRESHOLD)
    building_tree(tree,1,4)

if __name__ == '__main__':
    main()
