"""
Created on Tuesday, January 22 of 2024 by Greg
"""
import sys
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keyboard
import numpy as np
import PySpin
from PIL import Image as im
import mysql.connector

# Control algorithms
# ANN for stage 1
from Algorithms.stage_1.ANN import Stage2ANN
# CNN
from Algorithms.stage_2.xy_cnn.IA import BackPropagation

# image acquisition
from acquire_image import FLIR


class WinnerMove:
    # @staticmethod
    # def BDWL22():
    #     return -0.035, 0.04, True
    #
    # @staticmethod
    # def BDWL23():
    #     return -0.035, 0.012, True
    #
    # @staticmethod
    # def BDWL32():
    #     return -0.02, 0.06, True
    #
    # @staticmethod
    # def BDWL33():
    #     return -0.03, 0.013, True

    @staticmethod
    def dl0():
        return 0.0035, 0.004, True

    # @staticmethod
    # def BDWR22():
    #     return 0.035, 0.04, True
    #
    # @staticmethod
    # def BDWR23():
    #     return 0.035, 0.012, True
    #
    # @staticmethod
    # def BDWR32():
    #     return 0.02, 0.04, True
    #
    # @staticmethod
    # def BDWR33():
    #     return 0.03, 0.013, True

    @staticmethod
    def dr0():
        return -0.0035, 0.004, True

    # @staticmethod
    # def BUPL22():
    #     return -0.035, -0.04, True
    #
    # @staticmethod
    # def BUPL23():
    #     return -0.035, -0.012, True
    #
    # @staticmethod
    # def BUPL32():
    #     return -0.02, -0.04, True
    #
    # @staticmethod
    # def BUPL33():
    #     return -0.03, -0.013, True

    @staticmethod
    def ul0():
        return 0.0035, -0.004, True

    # @staticmethod
    # def BUPR22():
    #     return 0.035, -0.04, True
    #
    # @staticmethod
    # def BUPR23():
    #     return 0.035, -0.012, True
    #
    # @staticmethod
    # def BUPR32():
    #     return 0.02, -0.04, True

    # @staticmethod
    # def BUPR33():
    #     return 0.03, -0.013, True

    @staticmethod
    def ur0():
        return -0.0035, -0.004, True

    @staticmethod
    def d0():
        return 0, 0.0045, True

    @staticmethod
    def c():
        qry = SqlQuery()
        qry.lab_stop()
        return 0, 0, False

    @staticmethod
    def l0():
        return 0.008, 0, True

    @staticmethod
    def r0():
        return -0.008, 0, True

    @staticmethod
    def u0():
        return 0, -0.0045, True

    # @staticmethod
    # def ncup():
    #     return 0, -0.02, True

    # @staticmethod
    # def ncr():
    #     return 0.04, 0, True

    # @staticmethod
    # def ncl():
    #     return -0.04, 0, True

    # @staticmethod
    # def ncdw():
    #     return 0, 0.02, True

    @staticmethod
    def default():
        return 0, 0, False


# ======================================================================================================================
# plt.imshow(data)
# plt.show()
# data.save('ERROR.png')
# time2 = time.time()
# print('Time = ' + str(time2 - time1))
# ttime = time2 - time1
# qry.sqltime(ttime)
# ======================================================================================================================

class CNNController:
    def __init__(self, data):
        # Incoming image
        self.im_data = data
        # Set-points
        self.XSetpoint = 640
        self.YSetpoint = 512
        # instances
        self.bp = BackPropagation()
        self.qry = SqlQuery()
        self.model = tf.keras.models.load_model("model_10um.keras")
        # C:\Users\grego\OneDrive - Universidad de Guadalajara\GitHub\SpatialF_controller\model_CNN.keras

    def predict(self):
        winner_class = self.bp.predict(self.im_data, self.model)
        switcher = WinnerMove()
        case = getattr(switcher, winner_class, switcher.default)
        X, Y, stop_h = case()
        X = X
        Y = Y
        self.qry.qy(X, Y)
        stop_h = self.qry.next_step()
        return stop_h


class SqlQuery:
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="greg",
            password="contpass01",
            database="airy"
        )

        self.mycursor = self.mydb.cursor()

    def sqltime(self, t_time):
        print(type(t_time))
        sql = "INSERT INTO times (ID, time_seg) VALUES (%s, %s)"
        val = (None, t_time)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def map_sql(self, X, Y):
        print("--------------")
        print(X)
        print(Y)
        print("--------------")
        sql = "INSERT INTO MAPING (X, Y) VALUES (%s, %s)"
        val = (X, Y)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def lab_stop(self):
        sql = "UPDATE DATA SET STOP = %s WHERE ID = %s"
        val = (1, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def qy(self, X, Y):
        # def for py
        sql = "UPDATE DATA SET X = %s, Y = %s, SIGNALS = %s WHERE ID = %s"
        val = (-X, -Y, 1, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    @staticmethod
    def next_step():
        # def for py
        while True:
            mydb0 = mysql.connector.connect(
                host="localhost",
                user="greg",
                password="contpass01",
                database="airy"
            )

            mycursor0 = mydb0.cursor()
            mycursor0.execute("SELECT SIGNALS FROM DATA WHERE ID = 1")
            myresult = mycursor0.fetchall()
            myresult = myresult[0]
            myresult = myresult[0]
            mycursor0.close()
            mydb0.close()
            time.sleep(0.01)
            # print(myresult)
            if myresult == 0:
                return True
            if keyboard.is_pressed('ENTER'):
                print('Program is closing...')
                # input('Done! Press Enter to exit...')
                return False


if __name__ == "__main__":
    FLIR_instance = FLIR()
    stop_handle = True
    # Start CNN controller, a time independent controller
    CNNController = CNNController(None)
    while stop_handle:  # inverted boolean logic in this while condition
        exit_code = FLIR_instance.main()

        plt.imshow(FLIR_instance.image, cmap='gray')
        plt.show()
        time.sleep(2)
        # Preprocessing of the image
        image_data = FLIR_instance.image
        A = image_data
        B = image_data
        C = np.dstack((A, B))
        image = np.dstack((C, B))
        data = im.fromarray(image)
        data = data.resize((500, 400))
        CNNController.im_data = data
        stop_handle = CNNController.predict()

    FLIR_instance.stop_recording()
    if exit_code:
        sys.exit(0)
    else:
        sys.exit(1)
