import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2

from numpy.linalg import svd, inv, norm

class ImageStitcher:
    def __init__(self, ratio, num_sample, num_iteration, tolerance, require):
        self.ratio = ratio      # store it to the ImageStitcher object
        self.num_sample = num_sample
        self.num_iteration = num_iteration
        self.tolerance = tolerance
        self.require = require
        
    
        
    """ load multiple pairs of images """
    def load_images(self, path):
        """ load_image and convert BGR to RGB """
        def load_image(path, name, index):
            return cv2.imread(os.path.join(path, f'{name}{index}.jpg'), cv2.IMREAD_COLOR)[:, :, ::-1]   # in OpenCV, default channel is BGR -> reverse it
        
        image_list = []
        names = {i[:-5] for i in os.listdir(path)}  # extract image name from directory (delete the last 5 character of image)
        
        for name in names:
            left_image = load_image(path, name, 1)  # index: 1  ex: hill1.jpg
            right_image = load_image(path, name, 2)
            image_list.append((name, left_image, right_image))
        return image_list
    
    """ use SIFT to extract keypoints and descriptors from the image """
    def extract_sift_features(self, image):
        return cv2.SIFT_create().detectAndCompute(image, None)
    
    """ match keypoints and return the match points """
    def match_features(self, image1, keypoints1, descriptors1, image2, keypoints2, descriptors2):
        matches = []
        for i in range(descriptors1.shape[0]):
            # find the two closest descriptors in the other image
            distances = [(norm(descriptors1[i] - descriptors2[j]).astype(float), j) for j in range(descriptors2.shape[0])]
            distances.sort(key = lambda x: x[0])    # sort it by distance
            closest, second_closest = distances[:2]
            
            d1, j1 = closest    # d1: closest distance, j1: closest descriptor
            d2, j2 = second_closest
    