import pandas as pd
import numpy as np
from helper_functions import *
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class Assignment:
    """
        Class responsable for the core implementation of the assignmet

        Params:
            filename - Name of the data file to read
    """


    EARTH_RADIUS = 6371 #Earth radius in km


    def __init__(self, filename):
        """
            Initializer for the Assignment class.
            Obtains the data from a given file and shuffles that data.

            Params:
                filename - name of the file to obtain the data from
        """

        self.data = pd.read_csv(filename).loc[:, ["latitude", "longitude", "fault"]] #read and select select just latitude longitude and fault

        #Converting coordinates to Earth-centered, Earth-fixed (x, y, z)
        self.data["x"] = self.EARTH_RADIUS * np.cos(self.data["latitude"] * (np.pi/180.0)) * np.cos(self.data["longitude"] * (np.pi/180.0))
        self.data["y"] = self.EARTH_RADIUS * np.cos(self.data["latitude"] * (np.pi/180.0)) * np.sin(self.data["longitude"] * (np.pi/180.0))
        self.data["z"] = self.EARTH_RADIUS * np.sin(self.data["latitude"] * (np.pi/180.0))


    def k_means(self, ks):
        """
            Computes the K-Means algorithm for this objects data and for the given K values
        """

        coords = self.data[["x", "y", "z"]].values  #Get the coordinate values from our data
        k_metrics = []

        if(not type(ks) is list):
            ks = [ks]   #We can pass in a number or a list of numbers, if it's not a list then convert it to one

        for k in ks:
            kmeans = KMeans(n_clusters = k).fit(coords)
            labels = kmeans.predict(coords)
            centroids = kmeans.cluster_centers_

            silh_score = silhouette_score(coords, labels)
            r_index = rand_index(self.data["fault"].values, labels)

            k_metrics.append((k, silh_score, r_index[0], r_index[1], r_index[2], r_index[3]))

            plot_classes(labels, self.data["longitude"].values, self.data["latitude"].values, algorithm = "kmeans", param = k)   #Plots the seismic events with relation to the "predicted" labels
            
            plot_3D_with_centroids(self.data["x"],self.data["y"],self.data["z"], centroids[:, 0], centroids[:, 1], centroids[:, 2], param = k)
        plot_params(np.array(k_metrics)) #Plot the different k values vs their silhouette scores



    def gaussian_mix(self, num_components):
        """
            Computes the Gaussian Mixture Model algorithm for this objects data and for the given number of components
        """

        coords = self.data[["x", "y", "z"]].values
        ncomponents_metrics = []

        if(not type(num_components) is list):
            num_components = [num_components]   #We can pass in a number or a list of numbers, if it's not a list then convert it to one

        for num in num_components:
            gmm = GaussianMixture(n_components =  num).fit(coords)
            labels = gmm.predict(coords)

            silh_score = silhouette_score(coords, labels)
            r_index = rand_index(self.data["fault"].values, labels)

            ncomponents_metrics.append((num, silh_score, r_index[0], r_index[1], r_index[2], r_index[3]))

            plot_classes(labels, self.data["longitude"].values, self.data["latitude"].values, algorithm = "gmm", param = num)   #Plots the seismic events with relation to the "predicted" labels
            #o gmm.predict_proba(coords) da-nos o grau de pertenca de cada ponto as diferentes gaussianas
        
        plot_params(np.array(ncomponents_metrics), algorithm = "gmm", file_name = "numcomponents_vs_metrics.png") #Plot the different number of components vs their silhouette scores



    def dbscan(self, epsilons = None):
        """
            Computes the DBSCAN algorithm for this objects data and with a Epsilon choosen in the select_epsilon method
        """
        coords = self.data[["x", "y", "z"]].values

        if(epsilons is None):   #If the users did not pass in a Epsilon value
            estimated_epsilon = select_epsilon(coords)  #Epsilon selection method as described in "A density base algorithm for discovering clusters"

            dbscan = DBSCAN(eps = estimated_epsilon, min_samples = 4).fit(coords)   #Fitting DBSCAN to our points
            labels = dbscan.labels_ #DBSCAN "prediction" of the labelling of our dataset
            r_index = rand_index(self.data["fault"].values, labels) #Obtaining evalutation metrics from our external index

            plot_classes(labels, self.data["longitude"].values, self.data["latitude"].values, algorithm = "dbscan", param = estimated_epsilon)   #Plots the seismic events with relation to the "predicted" labels

            print(f"\nSilhouete Score: {round(silhouette_score(coords, labels), 3)}")
            print(f"Precision: {round(r_index[0], 3)} \nRecall: {round(r_index[1], 3)}\nRand: {round(r_index[2], 3)}\nF1 Score: {round(r_index[3], 3)}")

        else:   #In case the user passed in a fixed number of epsilon distances to observe

            param_metric = []

            for epsilon in epsilons:

                dbscan = DBSCAN(eps = epsilon, min_samples = 4).fit(coords) #Fitting DBSCAN to our points
                labels = dbscan.labels_ #DBSCAN "prediction" of the labelling of our dataset

                plot_classes(labels, self.data["longitude"].values, self.data["latitude"].values, algorithm = "dbscan",param = epsilon)   #Plots the seismic events with relation to the "predicted" labels

                plot_number_of_points_in_clusters(labels, epsilon)    #Obtaining the number of points present in each cluster
                plot_mean_distances_in_clusters(coords, labels, epsilon)   #Obtaining the mean distance between points in the same cluster

                silh_score = silhouette_score(coords, labels)
                r_index = rand_index(self.data["fault"].values, labels)

                param_metric.append((epsilon, silh_score, r_index[0], r_index[1], r_index[2], r_index[3]))
            
            plot_params(np.array(param_metric), algorithm = "dbscan", file_name = "epsilon_vs_metrics.png")