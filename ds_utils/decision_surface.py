import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


def Decision_Surface(data, col1, col2, target, model, probabilities=False, gridsize=100, sample=1, seed=0):
    # Select subsample
    np.random.seed(seed)
    indices = np.random.permutation(range(len(target)))[:int(sample*len(target))].tolist()
    data = data.iloc[indices]
    target = target.iloc[indices]
    # Get bounds
    x_min, x_max = data[col1].min(), data[col1].max()
    y_min, y_max = data[col2].min(), data[col2].max()
    # Create a mesh
    x_gridsize = (x_max - x_min) / gridsize
    y_gridsize = (y_max - y_min) / gridsize
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_gridsize), np.arange(y_min, y_max, y_gridsize))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    # Select solumns
    tdf = data[[col1, col2]]
    plt.ylabel(col2)
    plt.xlabel(col1)
    # Fit model
    if None != model:
        model.fit(tdf, target)
        if probabilities:
            # Color-scale on the contour (surface = separator)
            Z = model.predict_proba(meshed_data)[:, 1].reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
        else:
            # Only a curve/line on the contour (surface = separator)
            Z = model.predict(meshed_data).reshape(xx.shape) > 0.5
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    # Make scatter plot with data
    cmap = {0: 'blue', 1: 'red'}
    colors = [cmap[c] for c in target]
    plt.scatter(data[col1], data[col2], color=colors)
    # Build legend
    plt.legend(handles=[mpatches.Patch(color=cmap[k], label=k) for k in cmap], loc="best", title="Target", frameon=True)

    
def Regression_Surface(data, col1, col2, target, model, gridsize=0.5, sample=1):
    # Get bounds
    x_min, x_max = data[col1].min(), data[col1].max()
    y_min, y_max = data[col2].min(), data[col2].max()
    
    # Create a mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, gridsize), np.arange(y_min, y_max, gridsize))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    
    tdf = data[[col1, col2]]

    if model:
        model.fit(tdf, target)
        Z = model.predict(meshed_data).reshape(xx.shape)

    plt.ylabel(col2)
    plt.xlabel(col1)
    
    cs = plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)

    length = target.count()
    indices = np.random.permutation(range(length))[:int(sample*length)]

    plt.scatter(data[col1][indices], data[col2][indices], c=target[indices], cmap=plt.cm.coolwarm )
