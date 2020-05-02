import os
import random
import numpy as np

import scipy
from scipy.misc import imread
from scipy.ndimage import rotate, shift

from PIL import Image, ImageOps


def get_images_labels(all_images, nb_classes, nb_samples_per_class, image_size, sample_stategy = "uniform"):
    sample_classes = np.random.choice(range(len(all_images)), replace=True, size=nb_classes)
    if sample_stategy == "random":
        labels = np.random.randint(0, nb_classes, nb_classes * nb_samples_per_class)
    elif sample_stategy == "uniform":
        labels = np.concatenate([[i] * nb_samples_per_class for i in range(nb_classes)])
        np.random.shuffle(labels)
    angles = np.random.randint(0, 4, nb_classes) * 90
    images = [image_transform(all_images[sample_classes[i]][np.random.randint(0, len(all_images[sample_classes[i]]))],
                              angle=angles[i]+(np.random.rand()-0.5)*22.5, trans=np.random.randint(-10, 11, size=2).tolist(), size=image_size)
              for i in labels]
    return images, labels

def image_transform(image, angle=0., trans=(0.,0.), size=(20, 20)):
    image = ImageOps.invert(image.convert("L")).rotate(angle, translate=trans).resize(size)
    np_image= np.reshape(np.array(image, dtype=np.float32), newshape=(np.prod(size)))
    max_value = np.max(np_image)
    if max_value > 0.:
        np_image = np_image / max_value
    return np_image


# def get_images(character_folders, nb_classes, nb_samples_per_class, sample_stategy = "uniform"):
#     sampled_characters = random.sample(character_folders, nb_classes)
#     if sample_stategy == "random":
#         images_labels = [(label, os.path.join(character, image_path)) \
#                         for label, character in zip(np.arange(nb_classes), sampled_characters) \
#                         for image_path in os.listdir(character)]
#         images_labels = random.sample(images_labels, nb_classes * nb_samples_per_class)
#     elif sample_stategy == "uniform":
#         sampler = lambda x: random.sample(x, nb_samples_per_class)
#         images_labels = [(i, os.path.join(path, image))
#                          for i, path in zip(np.arange(nb_classes), sampled_characters)
#                          for image in sampler(os.listdir(path))]
#     random.shuffle(images_labels)
#     return images_labels

# def load_transform(image_path, angle=0., size=(20, 20)):
#     # Load the image
#     original = imread(image_path, flatten=True)
#     # Rotate the image
#     rotated = np.maximum(np.minimum(rotate(original, angle=angle * 90 + (np.random.rand() - 0.5) * 22.5, cval=1.), 1.), 0.)
#     # Resize the image
#     resized = np.asarray(scipy.misc.imresize(rotated, size=size), dtype=np.float32) / 255.
#     # Invert the image
#     inverted = 1. - resized
#     max_value = np.max(inverted)
#     if max_value > 0.:
#         inverted /= max_value
#     return inverted




