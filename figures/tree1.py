import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout

G = nx.DiGraph()

G.add_node(1, level=1)
G.add_nodes_from([2, 3], level=2)
G.add_nodes_from([4, 5, 6, 7], level=3)
G.add_nodes_from([8, 9, 10, 11, 12, 13], level=4)
G.add_nodes_from([14, 15], level=5)

G.add_edges_from([(1, 2), (1, 3)])
G.add_edges_from([(2, 4), (2, 5), (3, 6), (3, 7)])
G.add_edges_from([(5, 8), (5, 9), (6, 10), (6, 11), (7, 12), (7, 13)])
G.add_edges_from([(10, 14), (10, 15)])

frame = plt.gca()
frame.axes.get_xaxis().set_ticks([])
nx.draw(G, pos=graphviz_layout(G, prog='dot'), node_size=100, arrows=False)

plt.show()
