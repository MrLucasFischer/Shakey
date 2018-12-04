import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from mpl_toolkits import mplot3d
from sklearn.neighbors import KNeighborsClassifier
import itertools

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
    """
        Function that selects the estimated epsilon value from the user

        Params:
            coords - Coordinates of seismic events

        Returns 
            The user's estimated epsilon value
    """

    num_lines = len(coords) #Get the number of rows from the coordinates
    y_column = np.zeros(num_lines)  #Create a zero filed vector. This parameter is ignored

    #KneighborsClassifier is not used as a classifier
    #It's simply used in order to obtain the distance to the 4th nearest neighbor of each poin
    knn = KNeighborsClassifier(4).fit(coords, y_column) #Fit the KNeighboursClassifier to our data

    distances = knn.kneighbors(n_neighbors = 4)[0] 
    fourth_distance = distances[:, 3]   #Get the distances to the 4th neighbors of each point
    fourth_distance[::-1].sort()    #Sorting our distances list in descending order
    return get_user_epsilon(fourth_distance)
    

def get_user_epsilon(distances):
    """
        Function that will ask the user for a noise percentage in order to choose the epsilon value

        Param:
            distances - A list of distances to the 4th nearest neighbor of each point

        Returns
            The user's estimated epsilon value
    """

    accepted = False
    epsilon_value = -1

    while(not accepted):
        user_noise_percentage = input("\nInsert a noise percentage estimate (0 - 100): ")
        try:
            user_percentage = float(user_noise_percentage)
            if(user_percentage < 0.0 or user_percentage >= 100.0):
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
    """
        Function that will plot the distances vs points and will ask the user for its opinion

        Params:
            distances - A list of distances to the 4th nearest neighbor of each point
            noise_percentage - A float value corresponding to an estimated percentage of noise in our data set

        Rerturns
            user_opinion - A string with information about if the user accepted this epsilon value or not
            epsilon_value - The user's estimated epsilon value
    """

    num_points = len(distances)
    points = list(range(1,  num_points + 1))
    
    epsilon_value = distances[int((num_points - 1) * noise_percentage) + 1]

    plt.figure(figsize = (11, 8))
    font = {
        'weight' : 'regular',
        'size'   : 24
    }
    plt.rc('font', **font)
    plt.plot(distances, points)
    plt.axvline(x = epsilon_value, color = "red", label = f"Epsilon Value: {round(epsilon_value, 3)}")
    plt.xlabel("Distance to the 4th neighbor")
    plt.ylabel("Points")
    plt.legend()
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



def rand_index(faults, labels):
    """
        Function that calculates the Rand index metrics (precision, reacall, rand index, F1)

        Params:
            faults - number of the fault the point belongs to (-1 if it does not belong to any fault)
            labels - label of cluster the point was assigned to

        Returns:
            The metrics calculated from the Rand index (precision, recall, rand index and F1)
    """
    mapping_function = lambda pair: int(pair[0] == pair[1])

    true_positives = 0
    false_positives_partial = 0
    false_negatives = 0
    num_combs = 0

    for i in range(0 , len(faults) - 1):
        same_label = labels[i] == labels[i + 1:]
        same_cluster = faults[i] == faults[i + 1:]

        true_positives += np.sum(np.logical_and(same_label, same_cluster))  #Summing the total number of true positives found up until this point
        false_positives_partial += np.sum(np.logical_or(same_label, same_cluster))  #To have the actual number of false positives we need to subtract the total number of true positives outside the for loop
        false_negatives += np.sum(np.logical_and(np.logical_not(same_label), same_cluster)) #Summing the total number of false negatives found up until this point
        num_combs += len(same_cluster) #Summing the number of combinations up until this point

    false_positives = false_positives_partial - true_positives
    true_negatives = num_combs - (true_positives + false_positives + false_negatives)

    #Examples
    # predictions  = [1, 0, 1, 1, 0]
    # ground_truth = [0, 0, 1, 1, 1]

    # predictions AND ground_truth     = [0, 0, 1, 1, 0] -> Gives us the true positives
    # predictions OR  ground_truth     = [1, 0, 1, 1, 0] -> This minus the true positives gives us the false positive (the first 1)
    # NOT predictions AND ground_truth = [0, 0, 0, 0, 1] -> Gives us the false negative

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    rand = (true_positives + true_negatives) / num_combs
    f1 = 2 * (precision * recall / (precision + recall))

    return precision, recall, rand, f1


def plot_params(params_silhouette, algorithm = "kmean"):
    """
        Plots the different params (K or Number of Components) vs the silhouette score obtained for them
    """
    x_label = "K" if(algorithm == "kmean") else "Number of Components"

    plt.figure(figsize = (11, 8))
    font = {
    'weight' : 'regular',
    'size'   : 24}
    plt.rc('font', **font)

    plt.plot(params_silhouette[:, 0], params_silhouette[:, 1])  #TODO when we only have one entry on the list it shows nothing

    plt.xlabel(x_label)
    plt.ylabel("Silhouette Score")
    plt.show()