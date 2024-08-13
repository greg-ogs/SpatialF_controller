import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class BackPropagation:
    def __init__(self):
        self.val_ds = None
        self.train_ds = None
        self.class_names = ['BDWL22', 'BDWL23', 'BDWL32', 'BDWL33', 'BDWLS', 'BDWR22', 'BDWR23',
                            'BDWR32', 'BDWR33', 'BDWRS', 'BUPL22', 'BUPL23', 'BUPL32', 'BUPL33', 'BUPLS',
                            'BUPR22', 'BUPR23', 'BUPR32', 'BUPR33', 'BUPRS', 'CDW', 'CENTER', 'CL', 'CR', 'CUP',
                            'ncdw', 'ncl', 'ncr', 'ncup']

        # Batch * 2 and image from 180 by 180 to 700 * 875

        self.batch_size = 128
        self.img_height = 400
        self.img_width = 500
        self.image_size = (self.img_height, self.img_width)
        self.data_dir = "C:/Users/grego/Downloads/GitHub/DATASET"

        # image_count = len(list(data_dir.glob('*/*.jpg')))
        # print(image_count)

    def train_model(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=350,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=350,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.train_ds.class_names
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.Rescaling(1. / 255)

        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixel values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))
        print(self.class_names)
        num_classes = len(self.class_names)
        model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(64, 9, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 9, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 9, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 9, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 7, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 7, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(512, 7, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(4608, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        epochs = 20
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        model.save('model.keras')
        print("keras done")
        model.save('model.h5')
        print("h5 done")
        input("Enter to continue")

    def predict(self, img_data, model):
        # img = keras.preprocessing.image.load_img(
        #     img_dir, target_size=self.image_size)
        img_array = tf.keras.utils.img_to_array(img_data)
        img_array = tf.expand_dims(img_array, 0)
        # model = tf.keras.models.load_model("model.keras")
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Change class_names for a static array
        # Identify class index in this array

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )
        # del model
        # del img_array
        # del predictions
        # tf.keras.backend.clear_session()
        return self.class_names[np.argmax(score)]


if __name__ == "__main__":
    mod = BackPropagation()
    # for i in range(3):
    # mod.predict("C:/Users/grego/Downloads/Drives/Figure 2022-07-18 134302 (1).jpeg")
    mod.train_model()
