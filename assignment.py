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
        #plot_3D(self.data["x"],self.data["y"],self.data["z"])
        # plot_classes(self.data["fault"], self.data["longitude"], self.data["latitude"])


    def k_means(self, ks):
        """
            Computes the K-Means algorithm for this objects data and for the given K values
        """

        coords = self.data[["x", "y", "z"]].values  #Get the coordinate values from our data
        k_silh = []

        if(not type(ks) is list):
            ks = [ks]   #We can pass in a number or a list of numbers, if it's not a list then convert it to one

        for k in ks:
            kmeans = KMeans(n_clusters = k).fit(coords)
            labels = kmeans.predict(coords)
            centroids = kmeans.cluster_centers_

            silh_score = silhouette_score(coords, labels)
            print(silh_score)
            print(rand_index(self.data["fault"].values, labels))

            k_silh.append((k, silh_score))
            # plot_3D_with_centroids(self.data["x"],self.data["y"],self.data["z"], centroids[:, 0], centroids[:, 1], centroids[:, 2])

        plot_params(np.array(k_silh)) #Plot the different k values vs their silhouette scores



    def gaussian_mix(self, num_components):
        """
            Computes the Gaussian Mixture Model algorithm for this objects data and for the given number of components
        """

        coords = self.data[["x", "y", "z"]].values
        ncomponents_silh = []

        if(not type(num_components) is list):
            num_components = [num_components]   #We can pass in a number or a list of numbers, if it's not a list then convert it to one

        for num in num_components:
            gmm = GaussianMixture(n_components =  num).fit(coords)
            labels = gmm.predict(coords)

            silh_score = silhouette_score(coords, labels)
            print(silh_score)
            print(rand_index(self.data["fault"].values, labels))
            
            ncomponents_silh.append((num, silh_score))
            #o gmm.predict_proba(coords) da-nos o grau de pertenca de cada ponto as diferentes gaussianas

        plot_params(np.array(ncomponents_silh), algorithm = "gmm") #Plot the different number of components vs their silhouette scores



    def dbscan(self, epsilons = None):
        """
            Computes the DBSCAN algorithm for this objects data and with a Epsilon choosen in the select_epsilon method
        """
        coords = self.data[["x", "y", "z"]].values

        if(epsilons is None):
            estimated_epsilon = select_epsilon(coords)
            dbscan = DBSCAN(eps = estimated_epsilon, min_samples = 4).fit(coords)
            labels = dbscan.labels_
            print(silhouette_score(coords, labels))
            print(rand_index(self.data["fault"].values, labels))
        else:
            for epsilon in epsilons:
                dbscan = DBSCAN(eps = epsilon, min_samples = 4).fit(coords)
                labels = dbscan.labels_
                get_number_of_points_in_clusters(labels)
                print(silhouette_score(coords, labels))
                print(rand_index(self.data["fault"].values, labels))
                plot_classes(labels, self.data["longitude"].values, self.data["latitude"].values)