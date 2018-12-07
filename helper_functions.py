import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from mpl_toolkits import mplot3d
from sklearn.neighbors import KNeighborsClassifier
import itertools

__image_dir = "./images"

def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k', algorithm = "kmeans", param = -1):
    """
        Plot seismic events using Mollweide projection.
        Arguments are the cluster labels and the longitude and latitude
        vectors of the events
    """

    img = imread(__image_dir + "/Mollweide_projection_SW.jpg")
    plt.figure(figsize = (13,8), frameon = False)
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
    plt.figure(figsize = (13, 8), frameon = False)
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
        plt.plot(x[mask], y[mask], 'o', markersize = 10, mew = 1, zorder = 1, alpha = alpha, markeredgecolor = edge)
        ix = ix + 1
    mask = np.logical_not(nots)
    if np.sum(mask) > 0:
        plt.plot(x[mask], y[mask], '.', markersize = 2, mew = 1, markerfacecolor = 'w', markeredgecolor = edge)

    file_name = f"projection_algo={algorithm}_parm={round(param, 1)}.png"
    plt.savefig(__image_dir + "/" + file_name, dpi=300)
    plt.savefig(__image_dir + "/" + file_name[0:-3]+"eps", dpi=300)
    plt.show()
    plt.close()
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




def plot_3D_with_centroids (x, y, z, x_centroids, y_centroids, z_centroids, param = -1):
    """
        Function that plots the data into a 3D plane
        Params:
            x - x vector values
            y - y vector values
            z - z vector values
    """
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, s = 3, c = "blue", label = "Points")
    ax.scatter(x_centroids, y_centroids, z_centroids, s = 100, marker = "x", c = "red", label = "Centroids")
    plt.legend()
    ax.axis("equal")    #Garantee that axis have the same distance

    file_name = f"3d_plot_param={param}.png"
    plt.savefig(__image_dir + "/" + file_name, dpi=300)
    plt.savefig(__image_dir + "/" + file_name[0:-3]+"eps", dpi=300)
    plt.show()
    plt.close()


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

        Returns
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
    plt.plot(points, distances)
    plt.axhline(y = epsilon_value, color = "red", label = f"Epsilon Value: {round(epsilon_value, 3)}")
    plt.ylabel("Distance to the 4th neighbor")
    plt.xlabel("Points")
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
        Function that calculates the Rand index metrics (precision, recall, rand index, F1)

        Params:
            faults - number of the fault the point belongs to (-1 if it does not belong to any fault)
            labels - label of cluster the point was assigned to

        Returns:
            The metrics calculated from the Rand index (precision, recall, rand index and F1)
    """

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


def plot_params(param_metrics, algorithm = "kmean", file_name = "k_vs_metrics.png"):
    """
        Plots the different params (K, Number of Components or Epsilons) vs the silhouette score, 
        the precision, the recall, the rand and the f1 metrics obtained for them
    """

    x_axis = param_metrics[:, 0]
    silhouette_score = param_metrics[:, 1]
    precision = param_metrics[:, 2]
    recall = param_metrics[:, 3]
    rand = param_metrics[:, 4]
    f1 = param_metrics[:, 5]

    if (algorithm == "kmean"):
        x_label = "K"
    elif (algorithm == "gmm"):
        x_label = "Number of Components"
    else:
        x_label = "epsilon"

    plt.close()
    plt.figure(figsize = (11, 8))
    font = {
    'weight' : 'regular',
    'size'   : 24}
    plt.rc('font', **font)

    plt.plot(x_axis, silhouette_score, "-", linewidth = 3 ,label = "silhouette score")
    plt.plot(x_axis, precision, "-", linewidth = 3 ,label = "precision")
    plt.plot(x_axis, recall, "-", linewidth = 3 ,label = "recall")
    plt.plot(x_axis, rand, "-", linewidth = 3 ,label = "rand")
    plt.plot(x_axis, f1, "-", linewidth = 3 ,label = "f1")

    plt.xlabel(x_label)
    plt.ylabel("Metrics")
    plt.legend()

    plt.savefig(__image_dir + "/" + file_name, dpi=300)
    plt.savefig(__image_dir + "/" + file_name[0:-3]+"eps", dpi=300)
    plt.show()
    plt.close()


def plot_number_of_points_in_clusters(labels, epsilon):
    """
        Barplot of the total number of points inside a cluster for each cluster
    """

    number_of_clusters = len(set(labels)) - 1   #We subtract 1 to remove the "-1" label (which is the label for noise)
    x_axis = []
    y_axis = []

    for cluster in range(0, number_of_clusters):
        x_axis.append(cluster)
        y_axis.append(np.sum(labels == cluster))

    plt.close()
    plt.figure(figsize = (15, 8))
    plt.title("# Points per cluster")
    plt.bar(x = x_axis, height = y_axis)
    plt.axhline(y = 4, linestyle = "--", color='r', label = "4 point clusters")
    plt.legend()
    plt.xlabel("Cluster Labels")
    plt.ylabel("# Points in Cluster")
    plt.xticks(x_axis[::5], x_axis[::5])
    plt.yticks(range(int(np.max(y_axis)))[::10], range(int(np.max(y_axis)))[::10])

    file_name = f"num_points_in_cluster_eps={round(epsilon, 1)}.png"
    plt.savefig(__image_dir + "/" + file_name, dpi=300)
    plt.savefig(__image_dir + "/" + file_name[0:-3]+"eps", dpi=300)
    plt.show()
    plt.close()

    # Plot histogram
    plt.figure(figsize = (15, 8))
    plt.title("Histogram of # Points per cluster")
    bins = np.linspace(0, max(y_axis), num=1+(max(y_axis))/4)
    plt.hist(y_axis, bins=bins)
    plt.grid(True)
    plt.xlabel("# Points in Cluster")
    plt.ylabel("Frequency of clusters")
    file_name = f"hist_num_points_in_cluster_eps={round(epsilon, 1)}.png"
    plt.savefig(__image_dir + "/" + file_name, dpi=300)
    plt.savefig(__image_dir + "/" + file_name[0:-3]+"eps", dpi=300)
    plt.show()
    plt.close()


def plot_mean_distances_in_clusters(coords, labels, epsilon):
    """
        Barplot of the mean distances of points inside the same cluster for each cluster
    """

    number_of_clusters = len(set(labels)) - 1

    x_axis = [] # cluster label
    y_axis = [] # mean distances

    for cluster in range(0, number_of_clusters):
        # Get coordinates of points in this cluster
        coords_cluster = coords[labels == cluster]
        n_points = len(coords_cluster)
        
        # Set up knn and get distances
        knn = KNeighborsClassifier(n_points - 1).fit(coords_cluster, np.zeros(n_points)) #Fit the KNeighboursClassifier to this cluster points
        distances = knn.kneighbors()[0] #Obtain the distances of the points inside this cluster with KNeighborsClassifier

        x_axis.append(cluster)  #Add the current cluster to the x_axis
        y_axis.append(np.mean(distances)) # Add mean to list
        
    plt.close()
    plt.figure(figsize = (15, 8))
    plt.title('Mean intra-cluster distance')
    plt.bar(x = x_axis, height = y_axis)
    plt.xlabel("Cluster Labels")
    plt.ylabel("Mean distance within cluster")
    plt.xticks(x_axis[::5], x_axis[::5])
    plt.yticks(range(int(np.max(y_axis)))[::10], range(int(np.max(y_axis)))[::10] )

    file_name = f"dist_within_cluster_eps={round(epsilon, 1)}.png"
    plt.savefig(__image_dir + "/" + file_name, dpi=300)
    plt.savefig(__image_dir + "/" + file_name[0:-3]+"eps", dpi=300)
    plt.show()
    plt.close()

    # Plot histogram
    plt.figure(figsize = (15, 8))
    font = {
    'weight' : 'regular',
    'size'   : 24}
    plt.rc('font', **font)
    plt.title("Histogram of Mean intra-cluster distance")
    bins2 = np.linspace(0, max(y_axis), num=1+(max(y_axis))/10)
    plt.hist(y_axis, bins=bins2)
    plt.grid(True)
    plt.xlabel("Mean intra-cluster distance")
    plt.ylabel("Frequency of clusters")
    file_name = f"hist_dist_within_cluster_eps={round(epsilon, 1)}.png"
    plt.savefig(__image_dir + "/" + file_name, dpi=300)
    plt.savefig(__image_dir + "/" + file_name[0:-3]+"eps", dpi=300)
    plt.show()
    plt.close()