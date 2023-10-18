# borrowing heavily from question/answer here:
# https://codereview.stackexchange.com/questions/124449/simple-weighted-directed-graph-in-python
from pprint import pprint
import numpy as np

# we have a trailing dd amount
# we have an acct target amount
# we have a risk, reward, win % for each trade
# trailing dd / risk gives num risk steps from hwm to lose
# target / risk gives num initial risk steps to win
# any loss adds 1 additional risk steps to win, removes 1 from steps to loss
# any win adds (reward/risk) additional steps to loss up to initial value, removes (reward/risk) steps to win

class WinLossNode:

    def __init__(self, wins_left, losses_left):
        self.wins = wins_left
        self.losses = losses_left
        self._win_node = False
        self._loss_node = False

        if self.wins <= 0:
            self._win_node = True

        if self.losses <= 0:
            self._loss_node = True

    def __eq__(self, other):
        if self.is_win_node() and other.is_win_node():
            return True
        elif self.is_loss_node() and other.is_loss_node():
            return True
        elif (self.wins, self.losses) == (other.wins, other.losses):
            return True
        else:
            return False

    def __hash__(self):
        if (not self.is_win_node()) and (not self.is_loss_node()):
            return hash((self.wins, self.losses))
        elif self.is_win_node():
            return hash((-1, 0))
        elif self.is_loss_node():
            return hash((0, -1))

    def is_win_node(self):
        return self._win_node

    def is_loss_node(self):
        return self._loss_node

    def __str__(self):
        if self.is_win_node():
            return "win node"
        elif self.is_loss_node():
            return "loss node"
        else:
            return "({},{})".format(self.wins, self.losses)


def index_in_array(arr, wl_node):
    for idx in range(len(arr)):
        if arr[idx] == wl_node:
            return idx
    return -1


class WinLossDiGraph:
    """This class implements a directed, weighted graph with nodes represented by integers. """

    def __init__(self):
        """Initializes this digraph."""
        self.nodes = set()
        self.children = dict()
        self.parents = dict()
        self.edges = 0
        self._start_node = WinLossNode(1,1)
        self._node_arr = []
        self._adjacency_matrix = []
        self._q_matrix = []
        self._r_matrix = []
        self._n_matrix = []
        self._b_matrix = []

    def build_graph_from_start_node_and_params(self, win_probability, reward_risk, initial_wins_req,
                                               initial_losses_allowed):
        # initial node
        start_node = WinLossNode(initial_wins_req, initial_losses_allowed)
        self._start_node = start_node
        self.add_win_loss_child_nodes_recursion(start_node, win_probability, reward_risk, initial_losses_allowed)

    def add_win_loss_child_nodes_recursion(self, node: WinLossNode, win_probability, reward_risk,
                                           initial_losses_allowed):

        # any loss adds 1 additional risk steps to win, removes 1 from steps to loss
        # any win adds (reward/risk) additional steps to loss up to initial value, removes (reward/risk) steps to win
        if node.is_win_node():
            return  # stop recursion
        if node.is_loss_node():
            return
        if len(self.get_children_of(node)) == 2:
            return
        # win child
        steps_to_loss_for_win_child = min(node.losses + reward_risk, initial_losses_allowed)
        win_child_node = WinLossNode(node.wins - reward_risk, steps_to_loss_for_win_child)
        # loss child
        loss_child_node = WinLossNode(node.wins + 1, node.losses - 1)
        self.add_node(win_child_node)
        self.add_node(loss_child_node)
        self.add_arc(node, win_child_node, win_probability)
        self.add_arc(node, loss_child_node, 1 - win_probability)
        self.add_win_loss_child_nodes_recursion(win_child_node, win_probability, reward_risk, initial_losses_allowed)
        self.add_win_loss_child_nodes_recursion(loss_child_node, win_probability, reward_risk, initial_losses_allowed)

    def add_node(self, node):
        """If 'node' is not already present in this digraph,
           adds it and prepares its adjacency lists for children and parents."""
        for other_node in self.nodes:
            if node == other_node:
                # print("node {} already in list".format(node))
                return
        # print("adding node {}".format(node))

        self.nodes.add(node)
        self.children[node] = dict()
        self.parents[node] = dict()

    def add_arc(self, tail, head, weight):
        """Creates a directed arc pointing from 'tail' to 'head' and assigns 'weight' as its weight."""
        tail_absent = True
        head_absent = True
        for other_tail in self.nodes:
            if tail == other_tail:
                tail_absent = False

        if tail_absent:
            self.add_node(tail)

        for other_head in self.nodes:
            if head == other_head:
                head_absent = False

        if head_absent:
            self.add_node(head)

        self.children[tail][head] = weight
        self.parents[head][tail] = weight
        self.edges += 1

    def has_arc(self, tail, head):
        return tail in self.nodes and head in self.children[tail]

    def get_arc_weight(self, tail, head):
        if tail not in self.nodes:
            raise ValueError("The tail node is not present in this digraph.")

        if head not in self.nodes:
            raise ValueError("The head node is not present in this digraph.")

        if head not in self.children[tail].keys():
            raise ValueError("The edge ({}, {}) is not in this digraph.".format(tail, head))

        return self.children[tail][head]

    def remove_arc(self, tail, head):
        """Removes the directed arc from 'tail' to 'head'."""
        if tail not in self.nodes:
            return

        if head not in self.nodes:
            return

        del self.children[tail][head]
        del self.parents[head][tail]
        self.edges -= 1

    def remove_node(self, node):
        """Removes the node from this digraph. Also, removes all arcs incident on the input node."""
        if node not in self.nodes:
            return

        self.edges -= len(self.children[node]) + len(self.parents[node])

        # Unlink children:
        for child in self.children[node]:
            del self.parents[child][node]

        # Unlink parents:
        for parent in self.parents[node]:
            del self.children[parent][node]

        del self.children[node]
        del self.parents[node]
        self.nodes.remove(node)

    def __len__(self):
        return len(self.nodes)

    def number_of_arcs(self):
        return self.edges

    def get_parents_of(self, node):
        """Returns all parents of 'node'."""
        if node not in self.nodes:
            return []

        return self.parents[node].keys()

    def get_children_of(self, node):
        """Returns all children of 'node'."""
        if node not in self.nodes:
            return []

        return self.children[node].keys()

    def clear(self):
        del self.nodes[:]
        self.children.clear()
        self.parents.clear()
        self.edges = 0

    def print_graph(self):
        for node in self.nodes:
            print("node: {}".format(node))
            for child in self.get_children_of(node):
                print("child {} of node {}, weight {}".format(child, node, self.children[node][child]))

    def node_array(self):
        arr = []
        for node in self.nodes:
            if (not node.is_win_node()) and (not node.is_loss_node()):
                arr.append(node)
        win_node = WinLossNode(0, 1)
        loss_node = WinLossNode(1, 0)
        arr.append(win_node)
        arr.append(loss_node)
        return arr

    def generate_adj_matrix(self):
        node_arr = self.node_array()
        self._node_arr = node_arr
        # str = "["
        # for i in node_arr:
        #     str += i.__str__()
        # str += "]"
        # print(str)
        adj_matrix = []
        # For user input
        # A for loop for row entries
        for row in range(len(node_arr)):
            a = []
            # A for loop for column entries
            for column in range(len(node_arr)):
                a.append(0)
            adj_matrix.append(a)

        for i in range(len(node_arr)):
            node_i = node_arr[i]
            children_of_node_i = self.get_children_of(node_i)
            for j in range(len(node_arr)):
                if node_arr[j] in children_of_node_i:
                    adj_matrix[i][j] = self.children[node_i][node_arr[j]]

                elif i == len(node_arr) - 2 and j == len(node_arr) - 2:
                    adj_matrix[i][j] = 1

                elif i == len(node_arr) - 1 and j == len(node_arr) - 1:
                    adj_matrix[i][j] = 1

                else:
                    adj_matrix[i][j] = 0

        self._adjacency_matrix = np.asmatrix(adj_matrix)
        # pprint(self._adjacency_matrix)
        self._q_matrix = np.asmatrix(self._adjacency_matrix[:len(node_arr)-2,:len(node_arr)-2])
        self._r_matrix = np.asmatrix(self._adjacency_matrix[:len(node_arr)-2, len(node_arr)-2:])
        inv_n_matrix = np.identity(len(node_arr)-2) - self._q_matrix
        self._n_matrix = np.asmatrix(np.linalg.inv(inv_n_matrix))
        # pprint(self._n_matrix)
        self._b_matrix = np.matmul(self._n_matrix, self._r_matrix)
        # pprint(self._b_matrix)

    def get_win_prob_from_start_node(self):
        for i in range(len(self._node_arr)-2):
            if self._node_arr[i] == self._start_node:
                return self._b_matrix[i, 0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # we have a trailing dd amount
    print("Estimate odds of passing a prop eval with trailing drawdown \n"
          "given a single setup with a defined bracket and win percentage.\n"
          "Costs ignored. \n"
          "NOT FINANCIAL ADVICE. \n"
          "DO YOUR OWN RESEARCH. \n"
          "NO GUARANTEE OUR MATH IS CORRECT. \n"
          "RISK DISCLAIMER: https://www.prop-alpha.com/disclaimer")
    while True:
        # main program
        trailing_dd = float(input("Enter Trailing Drawdown Amount in Currency: "))
        # we have an acct target amount
        account_target = float(input("Enter Account Target Amount in Currency: "))
        # we have a risk, reward, win % for each trade
        stop_width = float(input("Enter Stop Size in Currency: "))
        tp_width = float(input("Enter Take Profit Size in Currency: "))
        win_pct = float(input("Enter Estimated Win Percent: "))
        num_loss = trailing_dd / stop_width
        num_win = account_target / stop_width
        rr_ratio = tp_width / stop_width
        win_frac = win_pct/100.0
        wldg = WinLossDiGraph()
        wldg.build_graph_from_start_node_and_params(win_frac, rr_ratio, num_win, num_loss)
        # wldg.print_graph()
        wldg.generate_adj_matrix()
        win_prob = wldg.get_win_prob_from_start_node()*100
        print(f"Estimated Probability of Success: {win_prob:.1f}%")
        while True:
            answer = str(input('Run again? (y/n): '))
            if answer in ('y', 'n'):
                break
            print("invalid input.")
        if answer == 'y':
            continue
        else:
            print("Goodbye")
            break
