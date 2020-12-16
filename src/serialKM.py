from createData import X, y

import configobj
import collections
import numpy as np
from time import perf_counter as pc
from sklearn.metrics.cluster import adjusted_rand_score

config = configobj.ConfigObj('env.b')
numClusters = int(config['NUM_CLUSTERS'])



def create_centroids():
    initial_centroid = []
    for i in range(numClusters):
        initial_centroid.append(X[i])

    initial_centroid = np.vstack(initial_centroid)

    return initial_centroid



def create_distance_matrix(initial_centroid):
    distMatrix = np.zeros((len(X), len(initial_centroid)))

    for j in range(len(initial_centroid)):
        for i in range(len(X)):
            distMatrix[i][j] = np.linalg.norm(initial_centroid[j]-X[i])

    return distMatrix



def update_centroid(initial_centroid, serial_cluster, numDataPerCluster):
    updated_centroid = np.zeros((len(initial_centroid), len(initial_centroid[0])))

    for k in range(1, numClusters + 1):
        indices = [i for i, j in enumerate(serial_cluster) if j == k]
        updated_centroid[k-1] = np.divide((np.sum([X[i] for i in indices], axis=0)).astype(np.float), numDataPerCluster[k])

    return updated_centroid



start_time = pc()

initial_centroid = create_centroids()

flag = True
while flag == True:
    serial_cluster = []

    distMatrix = create_distance_matrix(initial_centroid)

    for i in range(len(distMatrix)):
        serial_cluster.append(np.argmin(distMatrix[i]) + 1)

    numDataPerCluster = collections.Counter(serial_cluster)

    updated_centroid = update_centroid(initial_centroid, serial_cluster, numDataPerCluster)

    if np.all(updated_centroid == initial_centroid):
        serial_time = pc() - start_time
        flag = False
    else:
        initial_centroid = updated_centroid

serial_score = adjusted_rand_score(y, serial_cluster)
