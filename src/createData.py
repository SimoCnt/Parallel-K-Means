import configobj
from sklearn.datasets import make_blobs

config = configobj.ConfigObj('env.b')
numDataPoints = int(config['NUM_DATAPOINTS'])
numClusters = int(config['NUM_CLUSTERS'])
devStandard = float(config['DEV_STANDARD'])


X, y = make_blobs(n_samples=numDataPoints, centers=numClusters, cluster_std=devStandard, random_state=0)
