import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from mpl_toolkits import mplot3d


def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k'):
    """
        Plot seismic events using Mollweide projection.
        Arguments are the cluster labels and the longitude and latitude
        vectors of the events
    """
    img = imread("./images/Mollweide_projection_SW.jpg")
    plt.figure(figsize = (10,5), frameon = False)
    x = lon / 180* np.pi
    y = lat / 180 * np.pi
    ax = plt.subplot(111, projection = "mollweide")

    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis=0))
    print(np.min(t,axis=0))

    clims = np.array([(-np.pi, 0), (np.pi, 0), (0, -np.pi/2), (0, np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize = (10, 5), frameon = False)
    plt.subplot(111)
    plt.imshow(img, zorder = 0, extent = [lims[0, 0], lims[1, 0], lims[2, 1], lims[3, 1]], aspect = 1)

    x = t[:, 0]
    y= t[:, 1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)
    ix = 0
    for lab in diffs[diffs >= 0]:
        mask = labels == lab
        nots = np.logical_or(nots,mask)
        plt.plot(x[mask], y[mask], 'o', markersize = 4, mew = 1, zorder = 1, alpha = alpha, markeredgecolor = edge)
        ix = ix + 1
    mask = np.logical_not(nots)
    if np.sum(mask) > 0:
        plt.plot(x[mask], y[mask], '.', markersize = 1, mew = 1, markerfacecolor = 'w', markeredgecolor = edge)
    plt.show()
    plt.axis('off')




def plot_3D (x ,y, z):
    """
        Function that plots the data into a 3D plane
        Params:
            x - x vector values
            y - y vector values
            z - z vector values
    """
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, s = 10)
    ax.axis("equal")    #Garantee that axis have the same distance
    plt.show()