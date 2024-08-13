import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.optimizers.schedules import ExponentialDecay
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector as mysql
from tqdm import tqdm
import concurrent.futures


class Stage1ANN:  # Classification stage
    def __init__(self):
        # Instance attributes for availability for any transformation in data and model structure
        self.stage1_model_s = None # Instance attributes for one unified model structure. (Methods that can
        # modify the structure under some conditions)
        self.x = None  # Input variables for ANN. Instance attribute for dataset transformations
        self.y = None  # Output variables for ANN
        self.loaded_dataset = None
        # Instance attributes for querying in any method
        self.myresult = None
        self.mycursor = None
        self.mydb = None


    def load_data(self):
        # Load from MySQL
        self.mydb = mysql.connect(
            host="172.17.0.2",  # Use docker inspect <docker id > for ip
            user="user",
            database="dataset",
            password="userpass", port=3306
        )

        self.mycursor = self.mydb.cursor()
        self.mycursor.execute("SELECT * FROM dynamic")
        self.myresult = self.mycursor.fetchall()
        self.mydb.close()
        self.loaded_dataset = pd.DataFrame(self.myresult, columns=['ID', 'X', 'Y', 'RESULT'])
        # print(loaded_dataset.head())

    def prepare_data(self):
        self.x = self.loaded_dataset[['X', 'Y']].values
        self.y = self.loaded_dataset[['RESULT']].values

    def set_model(self):
        lr_schedule = ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,  # Number of steps before decay
            decay_rate=0.9  # Factor by which the learning rate is reduced
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_dim=self.x.shape[1]))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.000001),
                           metrics=['accuracy'])
        self.stage1_model_s = model

    def train(self):
        epochs = 20
        # Create a callback that saves the model's weights
        cp_callback = [keras.callbacks.ModelCheckpoint(filepath='model.{epoch:03d}-{val_accuracy:.2f}.keras',
                                                       monitor='val_accuracy', verbose=1, save_freq='epoch'),
                       # Path of the callback file, monitor is for monitoring a variable and use in conjunction with
                       # mode to choose the best epoch qhe use save_best_only
                       keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', verbose=1,
                                                     min_delta=0.00001, mode='min')
                       # Epoch to wait without improvement is in patience argument, min_delta is minimum improvement
                       # and mode is to stop when the quantity monitored has stopped increasing (max)
                       ]

        history = self.stage1_model_s.fit(self.x, self.y, epochs=epochs, initial_epoch=0, batch_size=1, verbose=1,
                                 validation_split=0.2, callbacks=cp_callback)
        self.stage1_model_s.save('model.keras')
        accuracy = self.stage1_model_s.evaluate(self.x, self.y)
        # acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs_range = range(epochs)
        # plt.figure(figsize=(8, 8))
        # plt.subplot(1, 2, 1)
        # plt.plot(epochs_range, acc, label='Training Accuracy')
        # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        # plt.legend(loc='lower right')
        # plt.title('Training and Validation Accuracy')
        # plt.subplot(1, 2, 2)
        # plt.plot(epochs_range, loss, label='Training Loss')
        # plt.plot(epochs_range, val_loss, label='Validation Loss')
        # plt.legend(loc='upper right')
        # plt.title('Training and Validation Loss')
        # plt.show()

class Stage2ANN:
    def __init__(self):
        # Less permissive class in instance attributes, this is the last, modifications must be made in stage1
        self.y = None
        self.stage1_model = keras.models.load_model("model.keras")
        self.stage2_model_s = None

    def generate_data(self):
        prediction = [0, ]
        result = np.empty(shape=(0, 2))
        for i in tqdm(np.arange(12, 14, 0.01), desc='Predicting'):
            for j in np.arange(24, 25, 0.01):
                # print(str(round(i, 2)) + ',' + str(round(j, 2)))
                data = np.round(np.array([[i, j]]), 3)
                prediction = self.stage1_model.predict(data, verbose=0)
                # print(self.prediction[0])
                if prediction[0] > 0.95:
                    result = np.vstack((result, data))
        return result

    def parallelize(self):
        pass

    def prepare_data(self):
        points = self.generate_data()
        middle_index = points.shape[0] // 2
        target_point = points[middle_index]
        return points, target_point



def available_gpu():  # Available nvidia gpu function
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    print('----------------------------------------------------------------')
    if gpus:
        # Optionally, print details about each GPU
        for gpu in gpus:
            print('----------------------------------------------------------------')
            print(gpu)

    print('----------------------------------------------------------------')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('----------------------------------------------------------------')
def stage1_main():
    stage1 = Stage1ANN()
    stage1.load_data()
    stage1.prepare_data()
    stage1.set_model()
    stage1.train()
def stage2_main():
    stage2 = Stage2ANN()


if __name__ == "__main__":
    # Testing methods
    available_gpu()
    # stage1_main()
    stage2_main()