import pandas as pd
import numpy as np


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
        
        # plot_3D(self.data["x"],self.data["y"],self.data["z"])
        # plot_classes(self.data["fault"], self.data["longitude"], self.data["latitude"])