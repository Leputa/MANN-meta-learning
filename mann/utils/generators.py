import os
import numpy as np
import random
from PIL import Image

from .images import get_images_labels


class OmniglotGenerator(object):
    def __init__(self, data_folder, nb_classes=5, nb_samples_per_class=10, img_size = (20, 20)):
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.img_size = img_size
        self.images = []
        for dirname, subdirname, filelist in os.walk(data_folder):
            if filelist:
                self.images.append(
                    [Image.open(os.path.join(dirname, filename)).copy() for filename in filelist]
                )
        num_train = 1200
        self.train_images = self.images[:num_train]
        self.test_images = self.images[num_train:]

    def sample(self, batch_type, batch_size, sample_strategy="random"):
        if batch_type == "train":
            data = self.train_images
        elif batch_type == "test":
            data = self.test_images

        sampled_inputs = np.zeros((batch_size, self.nb_classes * self.nb_samples_per_class, np.prod(self.img_size)), dtype=np.float32)
        sampled_outputs = np.zeros((batch_size, self.nb_classes * self.nb_samples_per_class), dtype=np.int32)

        for i in range(batch_size):
            images, labels = get_images_labels(data, self.nb_classes, self.nb_samples_per_class, self.img_size, sample_strategy)
            sampled_inputs[i] = np.asarray(images, dtype=np.float32)
            sampled_outputs[i] = np.asarray(labels, dtype=np.int32)
        return sampled_inputs, sampled_outputs






