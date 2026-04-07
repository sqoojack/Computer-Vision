
""" subsampling: Reducing the size of image. ex: img = img[::2, ::2] -> shrink size to half of image """

import os
import re   # used to regular expression
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy.fft import fft2, fftshift

class ImagePyramid:
    def __init__(self, filter_size, sigma):
        self.kernel = self.gaussian_filter(filter_size, sigma)
        
    """ construct gaussian filter """
    def gaussian_filter(self, filter_size, sigma):     # sigma: Gaussian standard
        kernel = np.zeros((filter_size, filter_size))
        center = filter_size // 2
        
        for i in range(filter_size):
            for j in range(filter_size):
                g = np.exp(
                    -((center - i) ** 2 + (center - j) ** 2) / (2 * (sigma ** 2))
                )   # compute Gaussain value
                g /= 2 * np.pi * (sigma ** 2)
                kernel[i, j] = g    # assign g to correct kernel position
        kernel /= kernel.sum()
        return kernel

    """ use Gaussian filter to smooth image """
    def smooth(self, img, kernel):
        return cv2.filter2D(img, -1, kernel)

    """ Create Gaussian pyramid """
    def gaussian_pyramid(self, img, num_layers, kernel):
        res = [np.array(img)]
        for i in range(num_layers - 1):
            img = self.smooth(img, kernel)
            img = img[::2, ::2]     # downsample
            res.append(np.array(img))
        return res

    """ Caculate image's magnitude spectrum """
    @ staticmethod
    def magnitude_spectrum(img):
        fshift = fftshift(fft2(img))    
        return np.log(np.abs(fshift) + 1e-7)
        
    """ Create Laplacian pyramid """
    def laplacian_pyramid(self, g_pyramid, num_layers):   # g_pyramid: Gaussian pyramid
        res = [g_pyramid[-1]]   # initialize result list, start at peak of Gaussian pyramid
        for i in range(1, num_layers):
            upsample = cv2.pyrUp(g_pyramid[num_layers - i])
            if g_pyramid[num_layers - i - 1].shape != upsample.shape:
                upsample = cv2.resize(upsample, (g_pyramid[num_layers - i - 1].shape[1], g_pyramid[num_layers - i - 1].shape[0]))
                
            laplacian = cv2.subtract(g_pyramid[num_layers - i - 1], upsample)   # calculate laplacian layer
            res.append(laplacian)
        return res[::-1]    # reverse it. (start at bottom layer)
            
def plt_result(g_pyramid, l_pyramid, num_layers, filter_size, filter_sigma, img_name):
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"filter_size = {filter_size} x {filter_size}, sigma = {filter_sigma}", fontsize=32)
    
    text_kwargs = {     # to control appearance text annotations
        'size': 18,
        'ha': 'right',  # horizon alignment to the right
        'va': 'center', # vertical alignment to the center
        'rotation': 'vertical'    # display text by vertical 
    }
    for i in range(num_layers):
        plt.subplot(4, num_layers, i + 1)    # 4: total has 4 column subgraph, num_layers: each column has num_layers subgraph, i+1: current subgraph's position
        plt.imshow(g_pyramid[i], cmap='gray'), plt.xticks([], []), plt.yticks([], [])   # xticks([], []): used to hide x axis tick label
        plt.title(f"level {i}", fontsize=20)
        
        if i == num_layers - 1:     # When arrive final layer, annotate particular text to make it easy to understand 
            plt.gca().text(1.3, 0.5, 'Gaussian (final layer)', transform=plt.gca().transAxes, **text_kwargs)    # 1.3, 0.5: x, y coordinate
        
        plt.subplot(4, num_layers, num_layers + i + 1)
        plt.imshow(l_pyramid[i], cmap='gray'), plt.xticks([], []), plt.yticks([], [])
        
        if i == num_layers - 1:
            plt.gca().text(1.3, 0.5, 'Laplacian (final layer)', transform=plt.gca().transAxes, **text_kwargs)
        
        plt.subplot(4, num_layers, num_layers * 2 + i + 1)
        plt.imshow(ImagePyramid.magnitude_spectrum(g_pyramid[i])), plt.xticks([], []), plt.yticks([], [])
        
        if i == num_layers - 1:
            plt.gca().text(1.3, 0.5, 'Gaussian Sqectrum (final layer)', transform=plt.gca().transAxes, **text_kwargs)
            
        plt.subplot(4, num_layers, num_layers * 3 + i + 1)
        plt.imshow(ImagePyramid.magnitude_spectrum(l_pyramid[i])), plt.xticks([], []), plt.yticks([], [])
        
        if i == num_layers - 1:
            plt.gca().text(1.3, 0.5, 'Laplacian Sqectrum (final layer)', transform=plt.gca().transAxes, **text_kwargs)
        
        plt.tight_layout()  # automatically adjust the spaces between subplots
        plt.savefig(os.path.join('output/task2_result', f'{img_name}_pyd_{num_layers}_size_{filter_size}_sigma_{filter_sigma}.png'), dpi=300)  # dpi: Dots Per Inch
        
def main():
    parser = ArgumentParser()
    parser.add_argument('--data_choose', type=int, default=0, help='Input a number between 0 and 6. Each number corresponds to a different pair of images.')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--filter_sigma', type=float, default=1.)
    args = parser.parse_args()
    
    """ To choose which pairs of images """
    data_paths = {
        0: ('./data/task1and2_hybrid_pyramid/0_Afghan_girl_before.jpg'),
        1: ('./data/task1and2_hybrid_pyramid/1_bicycle.bmp'),
        2: ('./data/task1and2_hybrid_pyramid/2_bird.bmp'),
        3: ('./data/task1and2_hybrid_pyramid/3_cat.bmp'),
        4: ('./data/task1and2_hybrid_pyramid/4_einstein.bmp'),
        5: ('./data/task1and2_hybrid_pyramid/5_fish.bmp'),
        6: ('./data/task1and2_hybrid_pyramid/6_makeup_before.jpg'),
        7: ("./our_data/car.jpg"),
        8: ("./our_data/sushi.jpg"),
    }
    
    if args.data_choose in data_paths:
        img_path = data_paths[args.data_choose]
    else:
        raise ValueError("Invalid value for data_choose. Must between 0 and 6.")
    
    img_name = re.sub(r'\..+', '', img_path.split('/')[-1])    # ex: img_name = '1_bicycle.bmp'
    os.makedirs('output/task2_result', exist_ok=True)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # read it by grayscale
    
    pyramid = ImagePyramid(args.filter_size, args.filter_sigma)
    
    kernel = pyramid.gaussian_filter(args.filter_size, args.filter_sigma)
    g_pyramid = pyramid.gaussian_pyramid(img, args.num_layers, kernel)
    l_pyramid = pyramid.laplacian_pyramid(g_pyramid, args.num_layers)
    plt_result(g_pyramid, l_pyramid, args.num_layers, args.filter_size, args.filter_sigma, img_name)
    
    print(f"Image pyramid has finished.")
    
if __name__ == '__main__':
    main()
    
""" 
    Result image: 
    First column: is Gaussian Pyramid. Through applying Gaussian filters and repeatedly downsampling, it can be seen that as the level increases, the resolution decreases.
    Second column: is Laplacian Pyramid. Calculated through the differences between each level of the Gaussian Pyramid. As the level increases, the edge features gradually decrease.
    Third column:   display each layer of Gaussian Pyramid's magnitude spectrum. The brighter part indicate stronger frequency components. 
                    As the level increases, image center is brighter. Maybe the reason is lower frequency more and more prominent.
"""
    