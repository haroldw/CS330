import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    images = {}
    for path in paths:
        images[path] = sampler(os.listdir(path))

    images_labels = []
    counter = 0
    while counter < nb_samples:
      for label, path in zip(labels, paths):
        images_labels.append((label, os.path.join(path,images[path][counter])))
      counter += 1

    train_data = images_labels[:-len(paths)]
    test_data = images_labels[-len(paths):]
    if shuffle:
      random.shuffle(train_data)
      random.shuffle(test_data)
    return train_data + test_data


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)  # misc.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size, shuffle=False):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders
        
        #############################
        #### YOUR CODE GOES HERE ####
        all_image_batches = np.zeros((batch_size, self.num_samples_per_class,
                                      self.num_classes, self.dim_input))
        all_label_batches = np.zeros((batch_size, self.num_samples_per_class,
                                      self.num_classes, self.num_classes))

        one_hot_labels = np.identity(self.num_classes)        
        for i in range(batch_size):
          # Random sample classes in the target folder
          tar_folders = random.sample(folders, self.num_classes)
          # get samples from each chosen folder
          data = get_images(tar_folders,
                            one_hot_labels,
                            nb_samples=self.num_samples_per_class,
                            shuffle=shuffle)

          # reshape data into the desired shape
          images = [image_file_to_array(image_file, self.dim_input)
                      for _, image_file in data]
          images = np.vstack(images).reshape(self.num_samples_per_class,
                                             self.num_classes,
                                             -1)
          
          labels = [label for label, _ in data]
          labels = np.vstack(labels).reshape(self.num_samples_per_class,
                                             self.num_classes,
                                             -1)
          all_image_batches[i]  = images
          all_label_batches[i]  = labels
          #############################

        return all_image_batches.astype(np.float32), all_label_batches.astype(np.float32)