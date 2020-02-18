import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.datasets import make_blobs
plt.rcParams['figure.figsize'] = 14, 8

# Make some data
def make_cluster_data():
    np.random.seed(46)
    X, Y = make_blobs(n_samples=10000, n_features=2, centers=5, cluster_std=3)

    return X, Y

def feature_printer(data, name):
    return str(list(data.columns[data.loc[name] == 1]))[1:-1]

def colorizer(Y):
    pal = ["#50514f", "#f25f5c", "#ffe066", "#247ba0", "#70c1b3"]
    pal = ["#DC3522", "#4C1B1B", "#FF358B", "#044C29", "#1E1E20", "#225378", "#1695A3", "#EB7F00"] * 3
    colors = np.array(Y, dtype="object")

    for i, color in enumerate(np.unique(Y)):
        colors[colors == color] = pal[i]

    return list(colors)

def Decision_Surface(X, target, model, cell_size=.01, surface=True, points=True):
    # Get bounds
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    # Create a mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_size), np.arange(y_min, y_max, cell_size))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    
    # Add interactions
    for i in range(X.shape[1]):
        if i <= 1:
            continue
        
        meshed_data = np.c_[meshed_data, np.power(xx.ravel(), i)]

    Z_flat = model.predict(meshed_data)
    Z = Z_flat.reshape(xx.shape)

    if surface:
        cs = plt.contourf(xx, yy, Z, color=colorizer(Z_flat))
    
    if points:
        plt.scatter(X[:, 0], X[:, 1], color=colorizer(target), linewidth=0, s=20)
    
    plt.xlabel("Feature 1",fontsize=20)
    plt.ylabel("Feature 2",fontsize=20)
    plt.show()
