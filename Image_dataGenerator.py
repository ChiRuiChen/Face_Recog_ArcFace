import tensorflow as tf
from preprocessing import normalize_numpy
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = '/home/mifon/Documents/SilenceChen/Datasets/CasiaWebfaces/casia-112x112/casia-112x112'
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=normalize_numpy
    )

train_generator = img_gen.flow_from_directory(
        dataset_dir,
        target_size=(112,112),
        batch_size=1
       )


train_generator_iter = iter(train_generator)

image, label = next(train_generator_iter)
plt.imshow(image[0])
print('class: ', np.argmax(label[0]))
