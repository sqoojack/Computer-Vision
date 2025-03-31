
""" Hybrid Image: merge two photos, one is processed by low-pass filter, another is processed by high-pass filter.
    low-pass filter: Is used for smoothing the image. Attenuates the high-frequency components and preserves the low-frequency components.
    high-pass filter: Is used for sharpening the image. Attenuates the low-frequency components and preserves the high-frequency components."""
    
import os
import re   # used to process string
import math
import cv2  # used to process image
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from skimage.transform import resize   # adjust image size
from numpy.fft import fft2, ifft2, fftshift, ifftshift  # used to process Fourier Transform

""" calculate image's center position """
def decide_size(image, div):    # div: dividion ratio
    center_x = image.shape[0] / div + (image.shape[0] % 2)
    center_y = image.shape[1] / div + (image.shape[1] % 2)
    return center_x, center_y

""" Generate filter matrix (Gaussion or ideal filter) """
def create_filter(image, sigma, filter_type='gaussian', highlow=True):  
    # sigma: filter's parameter, determine frequency influence scope, highlow: determine high-pass or low-pass filter (True -> low-pass filter)
    center_x, center_y = decide_size(image, 2)
    
    if filter_type == 'gaussian':
        def filter_func(i, j):
            entry = math.exp(-1.0 * ((i - center_x) ** 2 + (j - center_y) ** 2) / (2 * sigma ** 2))
            return entry if highlow else 1 - entry
    elif filter_type == 'ideal':
        def filter_func(i, j):
            entry = 1 if math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) <= sigma else 0
            return entry if highlow else 1 - entry
    else:
        raise ValueError("Unknown filter type")    
    
    filter_array = np.array([ [filter_func(i, j) for j in range(image.shape[1])] for i in range(image.shape[0]) ])
    return filter_array

""" apply filter to the frequency domain of the image """
def apply_filter(image, sigma, filter_type='gaussian', highlow=True):
    # DFT: Discrete Fourier Transform
    channels = []
    for i in range(image.shape[2]):     # apply filter to every channels
        shift_DFT = fftshift(fft2(image[:, :, i]))
        filter_DFT = shift_DFT * create_filter(image, sigma, filter_type, highlow)
        channels.append(np.real(ifft2(ifftshift(filter_DFT))))
    return np.stack(channels, axis=-1)

""" generate hybrid image (integrate high-frequency and low-frequency between two images) """
def hybrid_image(image1, image2, high_sigma, low_sigma, filter_type):
    return apply_filter(image1, high_sigma, filter_type, highlow=False) + apply_filter(image2, low_sigma, filter_type, highlow=True)

""" adjust two images to same size """
def resize_images(img1, img2):
    h1, w1 = img1.shape[:2]     # img1's height and width
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    return resize(img1, (h, w)), resize(img2, (h, w))

""" constraint image pixel value to 0~255, and cast datatype to 'uint8' """
def clip_and_cast(array):
    return array.clip(0, 255).astype(np.uint8)

def plot_and_save(image, title, save_path):
    plt.imshow(image, cmap='gray')  # cmap='gray': display image in grayscale
    plt.title(title, fontsize=12)
    plt.savefig(save_path, dpi=300)
    plt.clf()    # clean current figure
    
def process_and_save_hybrid(img1, img2, high_sigma, low_sigma, filter_type, result_name, data_choose):
    filter_image = hybrid_image(img1, img2, high_sigma, low_sigma, filter_type)
    filter_image = filter_image / np.max(filter_image)
    filter_image = np.clip(filter_image, 0.0, 1.0)
    
    title = f'{filter_type.capitalize()} Filter -> High_sigma: {high_sigma}, Low_sigma: {low_sigma}'
    save_path = os.path.join('output/task1_result', f'{data_choose}_{result_name}_{filter_type}_hybrid.png')
    plot_and_save(filter_image, title, save_path)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_choose', type=int, default=0, help='Input a number between 0 and 6. Each number corresponds to a different pair of images.')   # dataset has many pairs of images
    parser.add_argument('--low_sigma', type=float, default=5.0)
    parser.add_argument('--high_sigma', type=float, default=20.0)
    parser.add_argument('--filter_type', type=str, default='gaussian')
    parser.add_argument('--result_name', type=str, default='result')
    args = parser.parse_args()
    
    """ To choose which pairs of images """
    data_paths = {
        0: ('./data/task1and2_hybrid_pyramid/0_Afghan_girl_before.jpg', './data/task1and2_hybrid_pyramid/0_Afghan_girl_after.jpg'),
        1: ('./data/task1and2_hybrid_pyramid/1_bicycle.bmp', './data/task1and2_hybrid_pyramid/1_motorcycle.bmp'),
        2: ('./data/task1and2_hybrid_pyramid/2_bird.bmp', './data/task1and2_hybrid_pyramid/2_plane.bmp'),
        3: ('./data/task1and2_hybrid_pyramid/3_cat.bmp', './data/task1and2_hybrid_pyramid/3_dog.bmp'),
        4: ('./data/task1and2_hybrid_pyramid/4_einstein.bmp', './data/task1and2_hybrid_pyramid/4_marilyn.bmp'),
        5: ('./data/task1and2_hybrid_pyramid/5_fish.bmp', './data/task1and2_hybrid_pyramid/5_submarine.bmp'),
        6: ('./data/task1and2_hybrid_pyramid/6_makeup_before.jpg', './data/task1and2_hybrid_pyramid/6_makeup_after.jpg'),
    }
    
    if args.data_choose in data_paths:
        img1_path, img2_path = data_paths[args.data_choose] 
    else:
        raise ValueError("Invalid value for data_choose. Must be between 0 and 6.")
    
    os.makedirs("output/task1_result", exist_ok=True)
    
    """ Load images """
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)     # read gray scale image
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both input images not found.")
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize image to be the same size
    img1, img2 = resize_images(img1, img2)
    
    # process and save hybrid images
    process_and_save_hybrid(img1, img2, args.high_sigma, args.low_sigma, args.filter_type, args.result_name, args.data_choose)
    
    print("Hybrid image generation finished.")
    
""" Command Line: python3 task1.py --data_choose 2 """
        