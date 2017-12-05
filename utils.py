import numpy as np
from sklearn.cluster.bicluster import SpectralCoclustering
import os.path
import pickle
from lstm import *
import json

def load_models(file_name):
    if os.path.isfile(file_name):
        content = pickle.load(open(file_name))
        return content[0], content[1], content[2], content[3]
    else:
        print "Model Not Found"

def get_matrix(model):
	return model.E.data.numpy()[1:]/model.C.data.numpy()[1:]

def get_edges(matrix, word_clusters, hidden_clusters):
	res = np.zeros((len(word_clusters), len(hidden_clusters)))
	for i in range(len(word_clusters)):
		for j in range(len(hidden_clusters)):
			for w in word_clusters[i]:
				for h in hidden_clusters[j]:
					res[i][j] += matrix[w][h]
			res[i][j] /= len(word_clusters[i]) * len(hidden_clusters[j])
	return res.tolist()

def get_clusters(data):
	coclusters = SpectralCoclustering(n_clusters=5, random_state=0)
	coclusters.fit(data)
	word_clusters = []
	hidden_clusters = []
	for i in range(5):
		wc = coclusters.get_indices(i)[0]
		hc = coclusters.get_indices(i)[1]
		word_clusters.append(wc.tolist())
		hidden_clusters.append(hc.tolist())
	return word_clusters, hidden_clusters

def get_details():
	model, word_id, tag_id, cells = load_models("lstm.model")
	embeds = get_pretrained_embedings("glove.6B.200d.txt", word_id, 200)
	word_id, id_word, tag_id, id_tag = build_features_lower("PTBSmall", "train.tagged")
	data = get_matrix(model)
	word_clusters = get_clusters(data)[0]
	hidden_clusters = get_clusters(data)[1]
	edges = get_edges(data, word_clusters, hidden_clusters)
	res = list()
	for i, cluster in enumerate(word_clusters):
		res.append(list())
		centroid = np.zeros(200)
		for idx in cluster:
			centroid += embeds[idx]
		centroid /= len(cluster)
		temp = list()
		for j, idx in enumerate(cluster):
			weight = 20.0-np.linalg.norm(embeds[idx]-centroid)
			temp.append({"name":id_word[idx], "weight":weight, "response":data[idx].tolist()})
		res[i] = sorted(temp, key=lambda k: k['weight'], reverse=True)[:20]

	return res, hidden_clusters, edges

print json.dumps(get_details()[0])
print json.dumps(get_details()[1])
print json.dumps(get_details()[2])






