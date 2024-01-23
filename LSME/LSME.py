import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

from LSME.conv_encoder import Conv_Encoder

class LSME:


	def __init__(self, G):
		self.G = G.to_undirected()
		self.conv_encoder = Conv_Encoder()

		


	def build_embeddings(self, emb_dim, receptive_field, sample_size):
		A_set = []
		node_ids = []
		node_labels = []
		A_size_list = []
		for x in tqdm(list(self.G.nodes)):
			node_ids.append(x)
			node_labels.append(self.G.nodes[x]["role"])
			nb, nb_list = self.get_nodes_x_hops_away(x, receptive_field)
			A_mean = self.get_adj_matrix(nb, nb_list, x, self.G.nodes[x], sample_size)
			A_size_list.append(A_mean.shape[0])
			A_set.append(A_mean)

		A_set = self.format_signature_matrices(A_set, A_size_list)

		embeddings = self.conv_encoder.encode(A_set, emb_dim)
		embeddings = pd.DataFrame(embeddings)
		embeddings.columns = ["emb_"+str(i) for i in range(embeddings.shape[1])]
		embeddings["node_id"] = node_ids
		embeddings["role"] = node_labels
		return embeddings


	def format_signature_matrices(self, A_set, A_size_list):
		mod_set = []
		max_size = max(A_size_list)
		for A in A_set:
			A_size = A.shape[0]
			if A_size < max_size:
				delta = max_size - A_size
				mod_set.append(np.pad(A, [(0, delta), (0, delta)], 'constant', constant_values=0.0))
			else:
				mod_set.append(A)
		return np.array(mod_set)



	def get_nodes_x_hops_away(self, node, receptive_field):
		"""
			This method will compute the number of neighbors x hops away from
			a given node.
		"""
		node_dict = {node:0}
		dist_dict = {0:[node]}
		node_list = [node]
		keep_going = True
		visited_nodes = [node]
		while keep_going:
			n = node_list.pop(0)
			nbs = self.G.neighbors(n)
			for nb in nbs:
				if (nb not in node_dict) and (nb != node):
					node_list.append(nb)
					dist_to_source = len(nx.shortest_path(self.G, source=node, target=nb)) - 1
					node_dict[nb] = dist_to_source
					if dist_to_source > receptive_field:
						keep_going = False
						break
					visited_nodes.append(nb)
					if dist_to_source not in dist_dict:
						dist_dict[dist_to_source] = [nb]
					else:
						dist_dict[dist_to_source].append(nb)
			if len(node_list) == 0:
				keep_going = False
		visited_nodes = list(set(visited_nodes))
		return dist_dict, visited_nodes



	def get_adj_matrix(self, nb, nb_list, node_id, node_obj, sample_size):
		"""
			This method will create a sub-graph using the node list.
			It will then create a adj matrix from that sub-graph.
		"""
		nb = list(nb.values())
		H = self.G.subgraph(nb_list)
		A_list = []
		for ii in range(sample_size):
			node_map = {}
			counter = 0
			for i in nb:
				random.shuffle(i)
				nodes = i[:]
				# degree_map = np.array(H.degree(i))				
				# nodes = degree_map[:,0]
				# degrees = degree_map[:,1]
				# nodes = [x for _, x in sorted(zip(degrees, nodes), reverse=True)]
				for j in nodes:
					node_map[j] = counter
					counter += 1
			HH = nx.relabel_nodes(H, node_map, copy=True)
			A = nx.to_numpy_matrix(HH, nodelist=sorted(list(HH.nodes)))
			A_list.append(A)
		A_mean = np.mean(A_list, axis=0)
		# for ii in range(len(A_list)):
		# 	plt.imshow(A_list[ii])
		# 	plt.xticks([])
		# 	plt.yticks([])
		# 	# plt.show()
		# 	# plt.savefig("./test_folder/"+str(node_id)+".png", bbox_inches='tight')
		# 	plt.savefig("./test_folder/"+str(ii)+".png", bbox_inches='tight')
		
		# plt.imshow(A_mean)
		# plt.xticks([])
		# plt.yticks([])
		# # plt.show()
		# plt.savefig("./test_folder/"+str(node_id)+".png", bbox_inches='tight')

		return A_mean