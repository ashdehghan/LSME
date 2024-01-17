import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

class LSME:


	def __init__(self, G):
		self.G = G.to_undirected()


	def build_embeddings(self, emb_dim, receptive_field, sample_size):
		A_set = []
		for x in tqdm(list(self.G.nodes)):
			nb, nb_list = self.get_neighborhood_of_x(x, receptive_field)
			A_mean = self.get_adj_matrix(nb, nb_list, x, self.G.nodes[x], sample_size)
			A_set.append(A_mean)


	def get_neighborhood_of_x(self, x, receptive_field):
		res = [[x]]
		visited_nodes = [x]
		nn_1 = [x]
		while True:
			nn_2 = []
			while len(nn_1) > 0:
				i = nn_1.pop(0)
				nn = set(list(self.G.neighbors(i)))
				nn = list(set(nn).difference(visited_nodes))
				nn = list(set(nn).difference(nn_2))
				nn_2 += list(nn)
			nn_1 = nn_2.copy()
			if (len(visited_nodes) + len(nn_2)) >= receptive_field:
				sample_size = receptive_field - len(visited_nodes)
				nn_2 = list(np.random.choice(np.array(nn_2), sample_size, replace=False))
				visited_nodes += nn_2
				res.append(nn_2)
				break
			visited_nodes += nn_2
			res.append(nn_2)
		return res, visited_nodes


	def get_adj_matrix(self, nb, nb_list, node_id, node_obj, sample_size):
		"""
			This method will create a sub-graph using the node list.
			It will then create a adj matrix from that sub-graph.
		"""
		H = self.G.subgraph(nb_list)
		A_list = []
		for ii in range(sample_size):
			node_map = {}
			counter = 0
			for i in nb:
				random.shuffle(i)
				degree_map = np.array(H.degree(i))
				nodes = degree_map[:,0]
				degrees = degree_map[:,1]
				nodes = [x for _, x in sorted(zip(degrees, nodes), reverse=True)]
				for j in nodes:
					node_map[j] = counter
					counter += 1
			HH = nx.relabel_nodes(H, node_map, copy=True)
			A = nx.to_numpy_matrix(HH, nodelist=sorted(list(HH.nodes)))
			A_list.append(A)
		A_mean = np.mean(A_list, axis=0)
		plt.imshow(A_mean)
		plt.xticks([])
		plt.yticks([])
		# plt.show()
		# exit(0)
		plt.savefig("./test_folder/"+str(node_id)+".png", bbox_inches='tight')
		return A_mean