# @Author: Jamil Hasibul <Jamil-h> <Jamil-h>
# @Date:   2020-10-01T01:09:08-05:00
# @Email:  mdhasibul.jamil@siu.edu
# @Filename: treeEnvironment.py
# @Last modified by:   Jamil-h
# @Last modified time: 2020-10-03T00:25:26-05:00

import math
import random
import numpy as np
import re
import sys


class Rule:
    def __init__(self,priority,ranges):
        #each range is left inclusive and right exclusive, i.e., [left, right)
        self.priority=priority
        self.ranges=ranges
        self.names=["field1", "field2"]

    def is_intersect(self, dimension, left, right):   # intersect is different than contains
        return not (left >= self.ranges[dimension*2+1] or \
            right <= self.ranges[dimension*2])

    def is_intersect_multi_dimension(self, ranges):
        for i in range(2): # 2 fields total     # this is contains not just intersects
            if ranges[i*2] >= self.ranges[i*2+1] or \
                    ranges[i*2+1] <= self.ranges[i*2]:
                return False
        return True

    def sample_packet(self):
        field1 = random.randint(self.ranges[0], self.ranges[1] - 1)
        field2 = random.randint(self.ranges[2], self.ranges[3] - 1)
        packet = (field1, field2)
        assert self.matches(packet), packet
        return packet

    def matches(self, packet):
        assert len(packet) == 2, packet
        return self.is_intersect_multi_dimension([
            packet[0] + 0,  # Field1
            packet[0] + 1,
            packet[1] + 0,  # Field2
            packet[1] + 1,
        ])

    def is_covered_by(self, other, ranges): # other is another rule Usage: if other rule range contains
                                            # current rule range discard current rule
        for i in range(2):
            if (max(self.ranges[i*2], ranges[i*2]) < \
                    max(other.ranges[i*2], ranges[i*2]))or \
                    (min(self.ranges[i*2+1], ranges[i*2+1]) > \
                    min(other.ranges[i*2+1], ranges[i*2+1])):
                return False
        return True

    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            result += "%s:[%d, %d) " % (self.names[i], self.ranges[i * 2],
                                        self.ranges[i * 2 + 1])
        return result

# Read the rulefile and create the rules

def load_rules_from_file(file_name):
    rules = []
    rule_fmt = re.compile(r'(\d+)  '\
        r'(\d+)')
    for idx, line in enumerate(open(file_name)):
        elements = line[1:-1].split('\t')
        line = line.replace('\t', ' ')
        field1,field2= \
        (eval(rule_fmt.match(line).group(i)) for i in range(1, 3))
        #print(field1,field2)
        rules.append(
            Rule(idx, [
                field1, field1+ 1, field2, field2 + 1]))
    return rules

# Decimal to binary conversion
#output is [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] for 255,8
def to_bits(value, n):
    if value >= 2**n:
        print("WARNING: clamping value", value, "to", 2**n - 1)
        value = 2**n - 1
    assert value == int(value)
    b = list(bin(int(value))[2:])
    assert len(b) <= n, (value, b, n)
    return [0.0] * (n - len(b)) + [float(i) for i in b]

#Required to do one hot encoding of the states
#Partion level of each field is encoded with following value_function
#each field  [>=min, <max) -- 0%, 2%, 4%, 8%, 16%, 32%, 64%, 100%
#so each field count be cut into 8 different levels 0%

def onehot_encode(arr, n):
    out = []
    for a in arr:
        x = [0] * n
        for i in range(a):
            x[i] = 1
        out.extend(x)
    return out

class Node:

    def __init__(self, id, ranges, rules, depth):
        self.id = id
        self.ranges = ranges
        self.rules = rules
        self.depth = depth
        self.children = []
        self.action = None
        self.pushup_rules = None
        self.num_rules = len(self.rules)


    def match(self, packet):

        if self.children:
            for n in self.children:
                if n.contains(packet):
                    return n.match(packet)
            return None
        else:
            for r in self.rules:
                if r.matches(packet):
                    return r

    def is_intersect_multi_dimension(self, ranges):
        for i in range(5):
            if ranges[i*2] >= self.ranges[i*2+1] or \
                    ranges[i*2+1] <= self.ranges[i*2]:
                return False
        return True

    def contains(self, packet):
        assert len(packet) == 2, packet
        return self.is_intersect_multi_dimension([
            packet[0] + 0,  # Field 1
            packet[0] + 1,
            packet[1] + 0,  # Field 2
            packet[1] + 1,

        ])

    def is_useless(self):
        if not self.children:
            return False
        return max(len(c.rules) for c in self.children) == len(self.rules)

    def pruned_rules(self):
        new_rules = []
        for i in range(len(self.rules) - 1):
            rule = self.rules[len(self.rules) - 1 - i]
            flag = False
            for j in range(0, len(self.rules) - 1 - i):
                high_priority_rule = self.rules[j]
                if rule.is_covered_by(high_priority_rule, self.ranges):
                    flag = True
                    break
            if not flag:
                new_rules.append(rule)
        new_rules.append(self.rules[0])
        new_rules.reverse()
        return new_rules

    def get_state(self):
        state = []
        state.extend(to_bits(self.ranges[0], 8))
        state.extend(to_bits(self.ranges[1] - 1, 8))
        state.extend(to_bits(self.ranges[2], 8))
        state.extend(to_bits(self.ranges[3] - 1, 8))
        assert len(state) == 32, len(state)
        state.append(self.num_rules)
        return np.array(state)

    def __str__(self):
        result = "ID:%d\tAction:%s\tDepth:%d\tRange:\t%s\nChildren: " % (
            self.id, str(self.action), self.depth, str(self.ranges))
        for child in self.children:
            result += str(child.id) + " "
        result += "\nRules:\n"
        for rule in self.rules:
            result += str(rule) + "\n"
        if self.pushup_rules != None:
            result += "Pushup Rules:\n"
            for rule in self.pushup_rules:
                result += str(rule) + "\n"
        return result

class Tree:
    def __init__(
            self,
            rules,
            leaf_threshold,
            ):
        # hyperparameters
        self.leaf_threshold = leaf_threshold
        self.rules = rules
        self.root = self.create_node(
            0, [0, 2**8, 0, 2**8], rules, 1)
        self.current_node = self.root
        self.nodes_to_cut = [self.root]
        self.depth = 1
        self.node_count = 1

    def create_node(self, id, ranges, rules, depth):
        node = Node(id, ranges, rules, depth)
        return node

    def match(self, packet):
        return self.root.match(packet)

    def get_depth(self):
        return self.depth

    def get_current_node(self):
        return self.current_node

    def is_leaf(self, node):
        return len(node.rules) <= self.leaf_threshold

    def is_finish(self):
        return len(self.nodes_to_cut) == 0

    def update_tree(self, node, children):
        node.children.extend(children)
        children.reverse()
        self.nodes_to_cut.pop()
        self.nodes_to_cut.extend(children)
        self.current_node = self.nodes_to_cut[-1]

    def cut_current_node(self, cut_dimension, cut_num):
        return self.cut_node(self.current_node, cut_dimension, cut_num)

    def cut_node(self, node, cut_dimension, cut_num):
        self.depth = max(self.depth, node.depth + 1)
        node.action = ("cut", cut_dimension, cut_num)
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]
        range_per_cut = math.ceil((range_right - range_left) / cut_num)

        children = []
        assert cut_num > 0, (cut_dimension, cut_num)
        for i in range(cut_num):
            child_ranges = list(node.ranges)
            child_ranges[cut_dimension * 2] = range_left + i * range_per_cut
            child_ranges[cut_dimension * 2 + 1] = min(
                range_right, range_left + (i + 1) * range_per_cut)

            child_rules = []
            for rule in node.rules:
                if rule.is_intersect(cut_dimension,
                                     child_ranges[cut_dimension * 2],
                                     child_ranges[cut_dimension * 2 + 1]):
                    child_rules.append(rule)

            # result="node:%d\tRange:%s\nRules: "%(self.node_count,str(child_ranges))
            # for rule in child_rules:
            #     result += str(rule) + "\n"
            # print(result)
            child = self.create_node(self.node_count, child_ranges,
                                     child_rules, node.depth + 1)
            children.append(child)
            self.node_count += 1

        self.update_tree(node, children)
        return children

    def get_next_node(self):
        self.nodes_to_cut.pop()
        if len(self.nodes_to_cut) > 0:
            self.current_node = self.nodes_to_cut[-1]
        else:
            self.current_node = None
        return self.current_node

    def check_contiguous_region(self, node1, node2):
        count = 0
        for i in range(5):
            if node1.ranges[i*2+1] == node2.ranges[i*2] or \
                    node2.ranges[i*2+1] == node1.ranges[i*2]:
                if count == 1:
                    return False
                else:
                    count = 1
            elif node1.ranges[i*2] != node2.ranges[i*2] or \
                    node1.ranges[i*2+1] != node2.ranges[i*2+1]:
                return False
        if count == 0:
            return False
        return True

    def merge_region(self, node1, node2):
        for i in range(5):
            node1.ranges[i * 2] = min(node1.ranges[i * 2], node2.ranges[i * 2])
            node1.ranges[i * 2 + 1] = max(node1.ranges[i * 2 + 1],
                                          node2.ranges[i * 2 + 1])

    def node_rules_same_in_given_dimension(self,node,dimension):
        dimension_values=[]
        for rule in node.rules:
            dimension_values.append((rule.ranges[dimension*2],rule.ranges[dimension*2+1]))
        return len(set(dimension_values))<=1    #check if all the dimension values are same or not

    def compute_result(self):
        # memory space
        # non-leaf: 2 + 16 + 4 * child num
        # leaf: 2 + 16 * rule num
        # details:
        #     header: 2 bytes
        #     region boundary for non-leaf: 16 bytes
        #     each child pointer: 4 bytes
        #     each rule: 16 bytes
        result = {"bytes_per_rule": 0, "memory_access": 0, \
            "num_leaf_node": 0, "num_nonleaf_node": 0, "num_node": 0}
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                next_layer_nodes.extend(node.children)

                # compute bytes per rule
                if self.is_leaf(node):
                    result["bytes_per_rule"] += 2 + 16 * len(node.rules)
                    result["num_leaf_node"] += 1
                else:
                    result["bytes_per_rule"] += 2 + 16 + 4 * len(node.children)
                    result["num_nonleaf_node"] += 1

            nodes = next_layer_nodes

        result["memory_access"] = self._compute_memory_access(self.root)
        result["bytes_per_rule"] = result["bytes_per_rule"] / len(self.rules)
        result[
            "num_node"] = result["num_leaf_node"] + result["num_nonleaf_node"]
        return result

    def _compute_memory_access(self, node):
        if self.is_leaf(node) or not node.children:
            return 1
        else:
            return 1 + max(
                self._compute_memory_access(n) for n in node.children)

    def get_stats(self):
        widths = []
        dim_stats = []
        nodes = [self.root]
        while len(nodes) != 0 and len(widths) < 30:
            dim = [0] * 2
            next_layer_nodes = []
            for node in nodes:
                next_layer_nodes.extend(node.children)
                if node.action and node.action[0] == "cut":
                    dim[node.action[1]] += 1
            widths.append(len(nodes))
            dim_stats.append(dim)
            nodes = next_layer_nodes
        return {
            "widths": widths,
            "dim_stats": dim_stats,
        }

    def stats_str(self):
        stats = self.get_stats()
        out = "widths" + "," + ",".join(map(str, stats["widths"]))
        out += "\n"
        for i in range(len(stats["dim_stats"][0])):
            out += "dim{}".format(i) + "," + ",".join(
                str(d[i]) for d in stats["dim_stats"])
            out += "\n"
        return out

    def print_stats(self):
        print(self.stats_str())

    #layer details is controlled by layer_num
    def print_layers(self, layer_num=5):
        nodes = [self.root]
        for i in range(layer_num):
            if len(nodes) == 0:
                return

            print("Layer", i)
            next_layer_nodes = []
            for node in nodes:
                print(node)
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes

    def print_nodes_to_cut(self):
        result=""
        for node in self.nodes_to_cut:
            result += str(node.id)+","
        print("nodes to cut "+result)

    def __str__(self):
        result = ""
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                result += "%d; %s; %s; [" % (node.id, str(node.action),
                                             str(node.ranges))
                for child in node.children:
                    result += str(child.id) + " "
                result += "]\n"
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes
        return result
