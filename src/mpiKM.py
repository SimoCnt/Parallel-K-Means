from createData import X, y
from serialKM import *

import platform
import configobj
import numpy as np
import collections
from mpi4py import MPI
from time import perf_counter as pc
from sklearn.metrics.cluster import adjusted_rand_score
from matplotlib import pyplot as plt

config = configobj.ConfigObj('env.b')
numDataPoints = int(config['NUM_DATAPOINTS'])
numClusters = int(config['NUM_CLUSTERS'])



def createChunks(size):
    chunks = []
    avg = (numDataPoints / float(size))
    i = 0.0

    while(i < numDataPoints):
        chunks.append(X[int(i):int(i + avg)])
        i += avg

    return chunks



def createDistanceMatrix(chunk, centroids):
    distMatrix =np.zeros((len(chunk), len(centroids)))

    for j in range(len(centroids)):
        for i in range(len(chunk)):
            distMatrix[i][j] = np.linalg.norm(centroids[j] - chunk[i])

    return distMatrix



def addCounter(counter1, counter2, datatype):
    for item in counter2:
        counter1[item] += counter2[item]

    return counter1



def updateCentroids(centroids, clusters, chunk, totCounter):
    updatedCentroids = np.zeros((len(centroids), len(centroids[0])))

    for k in range(1, numClusters + 1):
        indices = [i for i, j in enumerate(clusters) if j == k]
        updatedCentroids[k-1] = np.divide((np.sum([chunk[i] for i in indices], axis=0)).astype(np.float),totCounter[k])

    updatedCentroidsReduced = comm.allreduce(updatedCentroids,MPI.SUM)

    return updatedCentroidsReduced



def results(mpi_cluster):
    plt.title("Dataset iniziale")
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    mpi_score = adjusted_rand_score(y, mpi_cluster)

    fig, axes = plt.subplots(1, 3, figsize=(30,10))
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=150)
    axes[1].scatter(X[:, 0], X[:, 1], c=serial_cluster, cmap='jet', edgecolor='k', s=150)
    axes[2].scatter(X[:, 0], X[:, 1], c=mpi_cluster, cmap='jet', edgecolor='k', s=150)
    axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[2].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
    axes[0].set_title('Effettivo', fontsize=18)
    axes[1].set_title('Previsto nel seriale, Score: {0}'.format(round(serial_score, 2)), fontsize=18)
    axes[2].set_title('Previsto in MPI, Score: {0}'.format(round(mpi_score, 2)), fontsize=18)
    plt.show()


    print("Serial Time: {0}".format(serial_time))
    print("Serial Score: {0}".format(serial_score))
    print("\nMPI Time: {0}".format(mpi_time))
    print("MPI Score: {0}".format(mpi_score))



if __name__=='__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    start_time = pc()

    if rank == 0:
        initial_centroid = []
        for i in range(numClusters):
            initial_centroid.append(X[i])

        initial_centroid = np.vstack(initial_centroid)
        chunks = createChunks(size)
    else:
        chunks = []
        initial_centroid = []

    chunk = comm.scatter(chunks, root=0)

    initial_centroid = comm.bcast(initial_centroid, root = 0)

    flag = True
    while flag == True:
        clusters = []
        mpi_cluster = []

        distMatrix = createDistanceMatrix(chunk, initial_centroid)

        for i in range (len(distMatrix)):
            clusters.append(np.argmin(distMatrix[i])+1)

        numClusterPoints = collections.Counter(clusters)

        counter = MPI.Op.Create(addCounter, commute=True)
        totCounter = comm.allreduce(numClusterPoints, op=counter)

        mpi_cluster = comm.gather(clusters, root=0)

        if rank==0:
            mpi_cluster = [item for sublist in mpi_cluster for item in sublist]

        updated_centroid = updateCentroids(initial_centroid, clusters, chunk, totCounter)
        comm.Barrier()

        if np.all(updated_centroid == initial_centroid):
            mpi_time = pc() - start_time
            flag=False
        else:
            initial_centroid = updated_centroid
        comm.Barrier()

    if rank == 0:
        results(mpi_cluster)
