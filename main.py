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
# control algorithms
from Algorithms.stage_1.ANN import Stage2ANN
from Algorithms.stage_2.xy_cnn.IA import BackPropagation
# image acquisition
from acquire_image import FLIR


#

class WinnerMove:
    def BDWL22(self):
        return -0.035, 0.04, True

    def BDWL23(self):
        return -0.035, 0.012, True

    def BDWL32(self):
        return -0.02, 0.06, True

    def BDWL33(self):
        return -0.03, 0.013, True

    def BDWLS(self):
        return -0.035, 0.04, True

    def BDWR22(self):
        return 0.035, 0.04, True

    def BDWR23(self):
        return 0.035, 0.012, True

    def BDWR32(self):
        return 0.02, 0.04, True

    def BDWR33(self):
        return 0.03, 0.013, True

    def BDWRS(self):
        return 0.035, 0.04, True

    def BUPL22(self):
        return -0.035, -0.04, True

    def BUPL23(self):
        return -0.035, -0.012, True

    def BUPL32(self):
        return -0.02, -0.04, True

    def BUPL33(self):
        return -0.03, -0.013, True

    def BUPLS(self):
        return -0.035, -0.04, True

    def BUPR22(self):
        return 0.035, -0.04, True

    def BUPR23(self):
        return 0.035, -0.012, True

    def BUPR32(self):
        return 0.02, -0.04, True

    def BUPR33(self):
        return 0.03, -0.013, True

    def BUPRS(self):
        return 0.035, -0.04, True

    def CDW(self):
        return 0, 0.045, True

    def CENTER(self):
        qry = SqlQuery()
        qry.lab_stop()
        return 0, 0, False

    def CL(self):
        return -0.08, 0, True

    def CR(self):
        return 0.08, 0, True

    def CUP(self):
        return 0, -0.045, True

    def ncup(self):
        return 0, -0.02, True

    def ncr(self):
        return 0.04, 0, True

    def ncl(self):
        return -0.04, 0, True

    def ncdw(self):
        return 0, 0.02, True

    def default(self):
        return 0, 0, False


class Camera:
    def __init__(self):

        self.image = None
        self.continue_recording = True

    def acquire_images(self, cam, nodemap, nodemap_tldevice):
        """
        This function continuously acquires images from a device and display them in a GUI.

        :param cam: CameraDataSet to acquire images from.
        :param nodemap: Device nodemap.
        :param nodemap_tldevice: Transport layer device nodemap.
        :type cam: CameraPtr
        :type nodemap: INodeMap
        :type nodemap_tldevice: INodeMap
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        # global continue_recording #Check uses
        XSetpoint = 640
        YSetpoint = 512

        print('*** IMAGE ACQUISITION ***\n')
        print('Press enter to close the program..')
        bp = BackPropagation()
        qry = SqlQuery()
        # Retrieve and display images
        time1 = time.time()
        aux = 0
        model = tf.keras.models.load_model("model.keras")
        # Testing first move (stage 1 algorithm)
        stage1 = Stage2ANN
        loop_aux = 0
        from_0 = input("Starting from 0? [Y/N]: ")
        if from_0 == 'Y' or from_0 == 'y':
            while True:
                loop_aux = loop_aux + 1
                points, midd_point = stage1.prepare_data()
                X = points[loop_aux][0]
                Y = points[loop_aux][1]
                # qry.map_sql(X, Y)
                qry.qy(X, Y)
                qry.next_step()
                image_result = cam.GetNextImage(100)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:
                    image_data = image_result.GetNDArray()
                    if np.mean(image_data) > 100:
                        break
        # This part is for cnn Controller
        while self.continue_recording:
            aux = aux + 1
            try:

                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.
                # time.sleep(2)
                image_result = cam.GetNextImage(100)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    # Getting the image data as a numpy array
                    image_data = image_result.GetNDArray()
                    image_data = np.uint8(image_data)
                    # 3D array for superpixel job
                    A = image_data
                    B = image_data
                    C = np.dstack((A, B))
                    image = np.dstack((C, B))
                    data = im.fromarray(image)
                    data = data.resize((375, 300))
                    imname = "img_" + str(aux) + ".png"
                    # imsave = data.save(imname)
                    # plt.imshow(data)
                    # plt.show()
                    winner_class = bp.predict(data, model)
                    switcher = WinnerMove()
                    case = getattr(switcher, winner_class, switcher.default)
                    X, Y, self.continue_recording = case()
                    # qry.map_sql(X, Y)
                    qry.qy(X, Y)
                    qry.next_step()
                    if keyboard.is_pressed('ENTER'):
                        # print('Program is closing...')

                        # Close figure
                        # plt.close('all')
                        # input('Done! Press Enter to exit...')
                        self.continue_recording = False
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False
        plt.imshow(data)
        plt.show()
        data.save('ERROR.png')
        time2 = time.time()
        print('Time = ' + str(time2 - time1))
        ttime = time2 - time1
        qry.sqltime(ttime)


class Controller:
    def __init__(self, instance):
        self.instance = instance

    def main(self):
        while True:
            time.sleep(2)
            plt.imshow(self.instance.image)
            plt.show()


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
        val = (X, Y, 1, 1)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def next_step(self):
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
                break


if __name__ == "__main__":
    print("Enter to debugger")
    # caminstance = Camera()
    # caminstance.capture()
    FLIR_instance = FLIR()
    while True:
        FLIR_instance.main()
        plt.imshow(FLIR_instance.image, cmap='gray')
        plt.show()
        time.sleep(2)
        image_data = FLIR_instance.image
        A = image_data
        B = image_data
        C = np.dstack((A, B))
        image = np.dstack((C, B))
        print(image.shape)
        data = im.fromarray(image)
        data = data.resize((375, 300))
        print(data.size)
        if keyboard.is_pressed('ENTER'):
            print('Program is closing...')

            # Close figure
            plt.close('all')
            input('Done! Press Enter to exit...')
            FLIR_instance.stop_recording()
            break
    # controller_instance = Controller(FLIR_instance)
    # if FLIR.result():
    #     sys.exit(0)
    # else:
    #     sys.exit(1)
