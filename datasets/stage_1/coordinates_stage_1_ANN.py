# import time
import pandas as pd
import numpy as np
import os
import mysql.connector as mysql


# from tqdm.auto import tqdm

class Datasets:
    def __init__(self):
        self.rand_xv = None
        self.rand_yv = None
        self.yv = None
        self.xv = None

        # MySQL
        self.mydb = mysql.connect(
            host="172.17.0.2",
            user="user",
            database="dataset",
            password="userpass", port=3306
        )
        self.mycursor = self.mydb.cursor()
    @staticmethod
    def create_base_dataset():  # Create a dataset using all possible combinations
        # High RAM memory, low time
        print("Creating base dataset...")
        x = np.arange(0, 25, 0.1)  # Creates an array from 0 to 25 for x
        y = np.arange(0, 25, 0.1)  # Creates an array from 0 to 25 for y
        coords = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        coords = np.round(coords, 2)
        n_dataset = pd.DataFrame(coords)
        n_dataset.head(100)
        n_dataset.to_csv("base_dataset.csv", index=False)
        print("Base dataset created. Please run again")

        # Low memory but long time
        # n_dataset = [[10, 10]]
        # x = np.arange(0, 25, 0.001)
        # y = np.arange(0, 25, 0.001)
        # coords_generator = ((xi, yi) for xi in x for yi in y)  # Generator object is returned
        # for coordinate in tqdm(coords_generator, desc="Creating base dataset", ascii=True, total=625000000):
        #     n_dataset = np.vstack((n_dataset, np.asarray(coordinate)))
        # print("Base dataset created.")

    @staticmethod
    def create_dynamic_range(coord):
        # Algorithm to create ranges and conditions
        if coord >= 0.1: lxd = coord - 0.1; dr = 0
        else: lxd = 0; dr = 0.1-coord;
        if coord <= 24.9: lxu = coord + 0.1; ur = 0
        else: lxu = 25; ur = (coord + 0.1) - 25;
        lxd = lxd - ur
        lxu = lxu + dr
        # Creating array of 50
        xv = np.round(np.linspace(lxd, lxu, 50), 3)
        return xv

    @staticmethod
    def create_random_range(coord):
        # create an array of random numbers that are not the same that input coords
        while True:
            rand_c = np.round(np.random.uniform(low=2, high=24, size=2), 2)  # New array of 2 random numbers
            if coord[0] == rand_c[0] or coord[1] == rand_c[1]:  # Conditional if the random number is the same that the
                # real one
                print('Same coordinates, trying again')
                rand_c = np.round(np.random.uniform(low=2, high=24, size=2), 2)  # Create another random number
                # to be checked by the conditional
            else: break
        rand_x_v = Datasets.create_dynamic_range(rand_c[0])  # Create a dynamic array of 50 by 1 using the random number
        rand_y_v = Datasets.create_dynamic_range(rand_c[1])  # The same for y
        return rand_x_v, rand_y_v

    def concatenate_coordinates(self, x, y):
        self.xv = self.create_dynamic_range(x)  # Create a dynamic array of 50 by 1 using the winning coordinate
        self.yv = self.create_dynamic_range(y)  # Same for y

        self.rand_xv, self.rand_yv = Datasets.create_random_range(np.array([x, y]))

        # Mach both x and y arrays for winning coordinates and insert them in MySQL table
        for coordinates in zip(self.xv, self.yv):
            # print("Pair of coordinates: {}, {}".format(round(coordinates[0], 2), round(coordinates[1], 2)))
            query = "INSERT INTO dynamic (X, Y, RESULT) VALUES (%s, %s, %s) ;"
            val = (coordinates[0], coordinates[1], 1)
            self.mycursor.execute(query, val)
            self.mydb.commit()

        # Add original coordinates if an adjustment in the ranges was made and insert them in MySQL table
        if x <= 0.1 or y <= 0.1 or x >= 24.9 or y >= 24.9:
            # print("Pair of real coordinates: {}, {}".format(round(x, 2), round(y, 2)))
            query = "INSERT INTO dynamic (X, Y, RESULT) VALUES (%s, %s, %s) ;"
            val = (round(x, 2), round(y, 2), 1)
            self.mycursor.execute(query, val)
            self.mydb.commit()

        # Also match the x and y arrays for random coordinates and insert them in MySQL table
        for rand_coordinates in zip(self.rand_xv, self.rand_yv):
            # print("Pair of coordinates: {}, {}".format(round(rand_coordinates[0], 2), round(rand_coordinates[1], 2)))
            query = "INSERT INTO dynamic (X, Y, RESULT) VALUES (%s, %s, %s) ;"
            val = (rand_coordinates[0], rand_coordinates[1], 0)
            self.mycursor.execute(query, val)
            self.mydb.commit()
        self.mydb.close()


if __name__ == '__main__':
    for i in range(50):
        newDataset = Datasets()
        newDataset.concatenate_coordinates(12.616, 24.265)
    # dataset = ReadDataset()
