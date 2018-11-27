import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from mpl_toolkits import mplot3d
from sklearn.neighbors import KNeighborsClassifier

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




def plot_3D (x, y, z):
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




def plot_3D_with_centroids (x, y, z, x_centroids, y_centroids, z_centroids):
    """
        Function that plots the data into a 3D plane
        Params:
            x - x vector values
            y - y vector values
            z - z vector values
    """
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, s = 10, c = "blue")
    ax.scatter(x_centroids, y_centroids, z_centroids, s = 100, c = "red")
    ax.axis("equal")    #Garantee that axis have the same distance
    plt.show()


def select_epsilon(coords):
    num_lines = len(coords)
    y_column = np.zeros(num_lines)

    knn = KNeighborsClassifier(4).fit(coords, y_column)

    distances = knn.kneighbors(n_neighbors = 4)[0]
    fourth_distance = distances[:, 3]
    fourth_distance[::-1].sort() #Ordering
    get_user_epsilon(fourth_distance)
    

def get_user_epsilon(distances):
    accepted = False
    epsilon_value = -1

    while(not accepted):
        user_noise_percentage = input("\nInsert a noise percentage estimate (0 - 100): ")
        try:
            user_percentage = float(user_noise_percentage)  #TODO ISTO ESTA A DAR MAL QUANDO METO 100 DA FUCK ??????????????????????????????????????
            if(user_percentage < 0.0 or user_percentage > 100.0):
                raise Exception
            else:
                noise_percentage = user_percentage / 100
                user_answer, epsilon_value = plot_distances(distances, noise_percentage)
                epsilon_value = epsilon_value
                accepted = user_answer == "yes" or user_answer == "y"
        except:
            #The user didn't sent us a number
            print("\nPlease insert a number between 0 and 100")

    return epsilon_value
        
    

def plot_distances(distances, noise_percentage):
    num_points = len(distances)
    points = list(range(1,  num_points + 1))
    
    epsilon_value = distances[int(num_points * noise_percentage) + 1]

    plt.figure(figsize = (11, 8))
    font = {
    'weight' : 'regular',
    'size'   : 24}
    plt.rc('font', **font)
    plt.plot(distances, points)
    plt.axvline(x = epsilon_value, color = "red", label = f"Epsilon Value: {round(epsilon_value, 3)}")
    plt.xlabel("Distance to the 4th neighbor")
    plt.ylabel("Points")
    plt.show()

    valid_answer = False
    while(not valid_answer):
        user_opinion = input("\nIs this epsilon estimate suitable? (y/n): ")
        user_opinion = user_opinion.lower()
        if(user_opinion == "y" or user_opinion == "n" or user_opinion == "yes" or user_opinion == "no"):
            valid_answer = True
        else:
            print("\nPlease insert an answer (y/n)")

    return user_opinion, epsilon_value