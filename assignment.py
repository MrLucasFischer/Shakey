import pandas as pd
import numpy as np
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#TODO MELHORAR ISTOOOOOOOOOOOOOOOOOOOOOOOOOO E POR NOUTRO FICHEIROOOOOOOOOOOOOO

def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5),frameon=False)    
    x = lon/180*np.pi
    y = lat/180*np.pi
    ax = plt.subplot(111, projection="mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis=0))
    print(np.min(t,axis=0))
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize=(10,5),frameon=False)    
    plt.subplot(111)
    plt.imshow(img,zorder=0,extent=[lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=alpha, markeredgecolor=edge)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1,markerfacecolor='w', markeredgecolor=edge)
    plt.show()
    plt.axis('off')


#TODO MELHORAR ISTO POR NOUTRO FICHEIRO

def plot_3D(x,y,z):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y,z,s=10)
    ax.axis("equal")
    plt.show()


class Assignment:

    EARTH_RADIUS = 6371

    def __init__(self, filename):
        data = pd.read_csv(filename).loc[: ,["latitude", "longitude", "fault"]] #read and select select just latitude longitude and fault

        #Converting coordinates to Earth-centered, Earth-fixed (x,y,z)
        data["x"] = self.EARTH_RADIUS * np.cos(data["latitude"] * (np.pi/180.0)) * np.cos(data["longitude"] * (np.pi/180.0))
        data["y"] = self.EARTH_RADIUS * np.cos(data["latitude"] * (np.pi/180.0)) * np.sin(data["longitude"] * (np.pi/180.0))
        data["z"] = self.EARTH_RADIUS * np.sin(data["latitude"] * (np.pi/180.0))
        
        plot_3D(data["x"],data["y"],data["z"])
        # plot_classes(data["fault"], data["longitude"], data["latitude"])