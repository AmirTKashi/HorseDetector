# %% Imports
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

# %% Load / Train  the model

# Check if model exists
if not os.path.isfile("model/horse_or_human.h5"):

    # check if the data has been downloaded:
    if not os.path.isdir("data/horse-or-human"):
        print("\n"*2)
        print("---- Downloading the Source Images ----\n")
        os.system(
            "wget --no-check-certificate \
                  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
                  -O data/horse-or-human.zip"
        )
        print("\n")
        print("---- Download Complete ----\n\n")
        print("---- Unzipping the images ----\n")
        local_zip = "data/horse-or-human.zip"
        zip_ref = zipfile.ZipFile(local_zip, "r")
        zip_ref.extractall("data/horse-or-human")
        zip_ref.close()
        print("---- Unzip complete ----")


    # Define Model
    horse_or_human = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(300, 300, 3)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fifth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation="relu"),
            # Only 1 output neuron. It will contain a value from 0-1
            # where 0 for 'horses' and 1 for the other ('humans')
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    horse_or_human.compile(
        optimizer=RMSprop(lr=0.001),  # 'adam',
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train Data Generator
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        "data/horse-or-human/",  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300*300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode="binary",
    )

    # Train the model
    # define callbacks:
    class StopTrainIfAccuracy(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoh, logs={}):
            print("\n", logs)
            if logs.get("accuracy") > 0.999:
                print("\nThe training has reached the accuracy above 99.9%")
                self.model.stop_training = True

    train_callback = StopTrainIfAccuracy()

    # Fit the model
    history = horse_or_human.fit_generator(
        train_generator,
        steps_per_epoch=8,  # 8 * 128 batch_size
        epochs=15,
        verbose=1,
        callbacks=[train_callback],
    )

    # Save model
    horse_or_human.save("model/horse_or_human.h5")

else:
    horse_or_human = load_model("model/horse_or_human.h5")


horse_or_human.summary()


# %% test the model on the test images
test_dir = "data/test_images/"
test_files = os.listdir(os.path.join(test_dir))

for f in test_files:
    img = load_img(test_dir + f, target_size=(300, 300))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    img_class = horse_or_human.predict(x)

    if img_class[0] < 0.5:
        text = f + " is a horse"
    else:
        text = f + " is NOT a horse"

    print(text, "\n")
    plt.imshow(img)
    plt.title(text)
    plt.show()
    plt.pause(0.05)

    c = input("Press any key to continue\n")
