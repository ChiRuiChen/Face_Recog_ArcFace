import tensorflow as tf


dataset = tf.data.TFRecordDataset("Dataset/faces_webface_112x112/train.rec"
                                  )

dataset = dataset.apply(tf.data.experimental.ignore_errors())
image_numpy = None
label_numpy = None

for image, label in dataset.take(1):
    image_numpy = image.numpy()
    label_numpy = label.numpy
