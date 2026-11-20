import csv
import gzip
import pickle
import os
import codecs

import numpy as np

from urllib.request import urlopen

import matplotlib.pyplot as plt


class TravelingSalesmanProblem:
    """This class encapsulates the Traveling Salesman Problem.
    City coordinates are read from an online file and distance matrix is calculated.
    The data is serialized to disk.
    The total distance can be calculated for a path represented by a list of city indices.
    A plot can be created for a path represented by a list of city indices.

    :param name: The name of the corresponding TSPLIB problem, e.g. 'burma14' or 'bayg29'.
    """

    def __init__(self, name):
        """
        Creates an instance of a TSP

        :param name: name of the TSP problem
        """

        # initialize instance variables:
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0

        self.data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tsp-data")

        # initialize the data:
        self.__initData()

    def __len__(self):
        """
        returns the length of the underlying TSP
        :return: the length of the underlying TSP (number of cities)
        """
        return self.tspSize

    def __initData(self):
        """Reads the serialized data, and if not available - calls __create_data() to prepare it
        """

        # attempt to read serialized data:
        try:
            self.locations = pickle.load(open(os.path.join(self.data_path, f"{self.name}-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join(self.data_path, f"{self.name}-dist.pickle"), "rb"))
        except (OSError, IOError):
            pass

        # serailized data not found - create the data from scratch:
        if not self.locations or not self.distances:
            self.__createData()

        # set the problem 'size':
        self.tspSize = len(self.locations)

    def __createData(self):
        """Reads the desired TSP file from the Internet, extracts the city coordinates, calculates the distances
        between every two cities and uses them to populate a distance matrix (two-dimensional array).
        It then serializes the city locations and the calculated distances to disk using the pickle utility.
        """
        self.locations = []

        # open whitespace-delimited file from url and read lines from it:
        with open(os.path.join(self.data_path, f"{self.name}.tsp")) as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)

            # skip lines until one of these lines is found:
            for row in reader:
                if row[0] in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                    break

            # read data lines until 'EOF' found:
            for row in reader:
                if row[0] != 'EOF':
                    # remove index at beginning of line and take x,y coordinates:
                    row = [x for x in row[1:] if x]

                    # convert x,y coordinates to ndarray:
                    self.locations.append(np.asarray(row, dtype=np.float64))
                else:
                    break

            # set the problem 'size':
            self.tspSize = len(self.locations)

            # print data:
            # print(f"length = {self.tspSize}, locations = {self.locations}")

            # initialize distance matrix by filling it with 0's:
            self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]

            # populate the distance matrix with TSPLIB EUC_2D integer-rounded Euclidean distances:
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    # calculate euclidean distance and round to nearest integer (TSPLIB EUC_2D nint):
                    distance = int(float(np.linalg.norm(self.locations[j] - self.locations[i])) + 0.5)
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
                    # print(f"{i}, {j}: location1 = {self.locations[i]}, location2 = {self.locations[j]} => distance = {distance}")

            # serialize locations and distances:
            pickle.dump(self.locations, open(os.path.join(self.data_path, f"{self.name}-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join(self.data_path, f"{self.name}-dist.pickle"), "wb"))

    def getOptimalTour(self):
        """Read the optimal tour from the corresponding .opt.tour file and return it as a list of city indices"""
        opt_tour = []
        tour_file = os.path.join(self.data_path, f"{self.name}.opt.tour")
        with open(tour_file) as f:
            in_tour_section = False
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if tokens[0] == "TOUR_SECTION":
                    in_tour_section = True
                    continue
                if in_tour_section:
                    for token in tokens:
                        if token in ("-1", "EOF"):
                            return opt_tour
                        opt_tour.append(int(token) - 1)  # Convert to 0-based
        return opt_tour

    def getTotalDistance(self, indices):
        """Calculates the total distance of the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: total distance of the path described by the given indices
        """
        # distance between th elast and first city:
        distance = self.distances[indices[-1]][indices[0]]

        # add the distance between each pair of consequtive cities:
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]

        return distance

    def plotData(self, indices):
        """plots the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: the resulting plot
        """

        # plot the dots representing the cities:
        plt.scatter(*zip(*self.locations), marker='.', color='red')

        # create a list of the corresponding city locations:
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # plot a line between each pair of consequtive cities:
        plt.plot(*zip(*locs), linestyle='-', color='blue')

        return plt


# testing the class:
def main():
    problems = ["berlin52", "eil101"
                , "tsp225", "pcb442", "pr1002"
                ]
    for i, problem in enumerate(problems, start=1):
        # create a problem instance:
        tsp = TravelingSalesmanProblem(problem)

        # read optimal solution from .opt.tour file:
        optimalSolution = tsp.getOptimalTour()
        exact_dist = tsp.getTotalDistance(optimalSolution)
        rounded_dist = round(exact_dist)

        print("Problem name: " + tsp.name)
        print("Optimal distance = ", exact_dist)
        print("-" * 50)

        # plot the solution in its own figure window:
        plt.figure(i)
        tsp.plotData(optimalSolution)
        plt.title(f"{tsp.name} (Optimal Tour Length: {rounded_dist})")

    # show all plots simultaneously:
    plt.show()


if __name__ == "__main__":
    main()
