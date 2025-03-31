
import os
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from task2 import ImagePyramid

""" remove white border from image """
def remove_white_border(img):
    white = 215     # because 215 is close to white value
    while np.all(img[0, :] >= white):   # check the image's first column. If all pixels > 215, remove that.
        img = img[1:, :]    # remove the first column 
        
    while np.all(img[-1, :] >= white):  # check the image's last column.
        img = img[:-1, :]
        
    while np.all(img[:, 0] >= white):   # check the image's first row.
        img = img[:, 1:]
        
    while np.all(img[:, -1] >= white):  
        img = img[:, :-1]
    return img

def remove_black_border(img):
    h, w = img.shape
    new_h, new_w = int(0.92 * h), int(0.92 * w)
    return img[h - new_h : new_h, w - new_w : new_w]

""" use x, y translation model to align color channels. """
def align_channels(base, to_align, search_range=15):    # base: base channel, to_align: channel to be aligned
    best_offset = (0, 0)
    min_diff = float('inf')
    base = (base - np.mean(base)) / np.std(base)    # standard base channel
    to_align = (to_align - np.mean(to_align)) / np.std(to_align)     # Normalize the pixel value of the channel to the range between 0 and 1
    
    for x in range(-search_range, search_range + 1):    # search best bias to smallest the difference bewteen two channels.
        for y in range(-search_range, search_range + 1):
            shifted = np.roll(np.roll(to_align, y, axis=0), x, axis=1)     # shift along the y-axis by best_offset[1] pixels, then shift along the x-axis by best_offset[0] pixels
            
            min_height = min(shifted.shape[0], base.shape[0]) # to crop base or shifted so that they can remain the same size
            min_width = min(shifted.shape[1], base.shape[1])
            shifted_cropped = shifted[:min_height, :min_width]
            base_cropped = base[:min_height, :min_width]
            
            diff = np.sum((base_cropped - shifted_cropped) ** 2)
            if diff < min_diff:
                min_diff = diff
                best_offset = (x, y)
    aligned = np.roll(np.roll(to_align, best_offset[1], axis=0), best_offset[0], axis=1)
    aligned = aligned[:base.shape[0], :base.shape[1]]   # crop to match the size of the base channel
    return aligned

""" Process the glass plate image, split it to R, G, B channels, and align them, merge them into an RGB image """
def process_glass_plate(img_path, pyramid_layer=5, filter_size=5, filter_sigma=1.):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = remove_white_border(img)
    height = img.shape[0] // 3      # img.shape[0]: the height of image, divided by 3 to get each colors height (R, G, B)
    b_channel = remove_black_border(img[:height, :])     # The first 1/3 of the image is used as the blue channel    
    g_channel = remove_black_border(img[height:2 * height, :])   # The middle 1/3 of the image is used as the green channel
    r_channel = remove_black_border(img[2 * height:, :])     # , is used to distinguish x, y axis. -> form 2/3 of the image height to the end, selecting all rows.
    
    kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
    kernel = kernel * kernel.T
    
    pyramid = ImagePyramid(filter_size, filter_sigma)
    
    g_pyramid_b = pyramid.gaussian_pyramid(b_channel, pyramid_layer, kernel)[::-1]
    g_pyramid_g = pyramid.gaussian_pyramid(g_channel, pyramid_layer, kernel)[::-1]
    g_pyramid_r = pyramid.gaussian_pyramid(r_channel, pyramid_layer, kernel)[::-1]
    
    """ align G, R channels to B channels """
    for i in range(1, pyramid_layer):    
        g_aligned = align_channels(g_pyramid_b[i], g_pyramid_g[i])
        r_aligned = align_channels(g_pyramid_b[i], g_pyramid_r[i])
    
    """ Ensure all channels have the same size """
    min_height = min(b_channel.shape[0], g_aligned.shape[0], r_aligned.shape[0])
    min_width = min(b_channel.shape[1], g_aligned.shape[1], g_aligned.shape[1])
    b_channel = b_channel[:min_height, :min_width]
    g_aligned = g_aligned[:min_height, :min_width]
    r_aligned = r_aligned[:min_height, :min_width]
    
    """ transform channels to uint8 """
    b_channel = b_channel.astype(np.uint8)
    g_aligned = ((g_aligned - np.min(g_aligned)) / (np.max(g_aligned) - np.min(g_aligned)) * 255).astype(np.uint8)   # Rescale these data to the range of 0 to 255
    r_aligned = ((r_aligned - np.min(r_aligned)) / (np.max(r_aligned) - np.min(r_aligned)) * 255).astype(np.uint8)
    
    color_image = cv2.merge((b_channel, g_aligned, r_aligned))
    return color_image

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_choose', type=int, default=0)
    args = parser.parse_args()
    
    data_dict = {
        0: './data/task3_colorizing/cathedral.jpg',
        1: './data/task3_colorizing/emir.tif',
        2: './data/task3_colorizing/icon.tif',
        3: './data/task3_colorizing/lady.tif',
        4: './data/task3_colorizing/melons.tif',
        5: './data/task3_colorizing/monastery.jpg',
        6: './data/task3_colorizing/nativity.jpg',
        7: './data/task3_colorizing/onion_church.tif',
        8: './data/task3_colorizing/three_generations.tif',
        9: './data/task3_colorizing/tobolsk.jpg',
    }
    
    if args.data_choose in data_dict:
        img_path = data_dict[args.data_choose]
    else:
        raise ValueError("Invalid value of choose_data. Must between 0 and 9.")
    
    img_name = re.sub(r'\..+', '', img_path.split('/')[-1])    # Take the last part of img_path as the img_name
    os.makedirs("output/task3_result", exist_ok=True)
    
    color_img = process_glass_plate(img_path)
    
    output_path = os.path.join("output/task3_result", f'{img_name}_colorize.png')
    cv2.imwrite(output_path, color_img)
    print(f"Successfully stored result image to {output_path}")
    
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title("Colorized Image")
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    main()
    