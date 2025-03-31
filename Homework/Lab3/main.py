import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm, svd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--num_sample', type=int, default=10)
    parser.add_argument('--num_iteration', type=int, default=3000)
    parser.add_argument('--tolerance', type=float, default=5)
    parser.add_argument('--require', type=float, default=0.9)
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--my_data', type=str, default='my_data')
    parser.add_argument('--output', type=str, default='output')
    return parser.parse_args()


def load_data(path):
    def load_image(path, name, index):
        return cv2.imread(os.path.join(path, f'{name}{index}.jpg'), cv2.IMREAD_COLOR)[:, :, ::-1]

    names = {i[:-5] for i in os.listdir(path)}
    return tuple(zip(names, (load_image(path, name, 1) for name in names), (load_image(path, name, 2) for name in names)))


def stitch(left, right, ratio, num_sample, num_iteration, tolerance, require):
    def sift(image):
        return cv2.SIFT_create().detectAndCompute(image, None)

    def match(image1, key_points1, descriptions1, image2, key_points2, descriptions2, ratio):
        matches = []
        for i in range(descriptions1.shape[0]):
            first, second = (cv2.DMatch(_distance=d, _trainIdx=j, _queryIdx=i)
                             for d, j, i in sorted((norm(descriptions1[i] - descriptions2[j]).astype(float), j, i)
                                                   for j in range(descriptions2.shape[0]))[:2])
            if not ratio or first.distance < ratio * second.distance:
                matches.append(first)

        matching = cv2.drawMatches(image1, key_points1, image2, key_points2, matches, None, flags=2)
        points1 = np.array([key_points1[m.queryIdx].pt for m in matches])
        points2 = np.array([key_points2[m.trainIdx].pt for m in matches])

        return points1, points2, matching

    def ransac(source_points, destination_points, num_sample, num_iteration, tolerance, require):
        def homomat(source_points, destination_points):
            P = np.zeros((source_points.shape[0] * 2, 9))
            for i, (s, d) in enumerate(zip(source_points, destination_points)):
                PT_i = np.array((s[0], s[1], 1))
                P[i * 2, :] = [*(-1 * PT_i), 0, 0, 0, *(d[0] * PT_i)]
                P[i * 2 + 1, :] = [0, 0, 0, *(-1 * PT_i), *(d[1] * PT_i)]

            _, _, VT = svd(P, full_matrices=False)
            h = VT.T[:, -1]
            H = (h / h[-1]).reshape(3, 3)

            return H

        max_num_inliers, optimal = 0, None
        require, index, ones = require * len(source_points), np.arange(len(source_points)), np.ones((len(source_points), 1))
        for _ in range(num_iteration):
            np.random.shuffle(index)
            sample = index[:num_sample]

            sample_source_points, sample_destination_points = source_points[sample], destination_points[sample]
            H = homomat(sample_source_points, sample_destination_points)

            sample_source_points, sample_destination_points = np.hstack((source_points, ones)), np.hstack((destination_points, ones))
            estimate = H @ sample_source_points.T
            estimate = (estimate / estimate[-1, :]).T

            if max_num_inliers < (num_inliers := sum(1 for e, d in zip(estimate, sample_destination_points) if norm(e - d) < tolerance)):
                max_num_inliers, optimal = num_inliers, H

            if require <= max_num_inliers:
                break

        return optimal

    def warp(image, T):
        def boundaries(image, T):
            corners = np.array((
                (0, 0, 1),
                (image.shape[1], 0, 1),
                (0, image.shape[0], 1),
                (image.shape[1], image.shape[0], 1)
            ))
            corners = T @ corners.T
            corners = corners / corners[-1, :]

            right_width_min = int(min(corners[0, 1], corners[0, 3]))
            boundaries = corners[0, :].min(), corners[0, :].max(), corners[1, :].min(), corners[1, :].max()
            width_min, width_max, height_min, height_max = map(lambda x: x.astype(np.int32), boundaries)
            width = width_max - width_min + 1 + np.abs(width_min)
            height = height_max - height_min + 1

            return height_min, right_width_min, width, height

        def default_coordinates(height, width, height_min):
            coordinates = np.empty((height, width, 2), dtype=np.float32)
            for i, t in enumerate(np.ix_(np.arange(height), np.arange(width))):
                coordinates[..., i] = t
            coordinates[..., 1] += height_min
            return coordinates.reshape(-1, 2)

        def mapping(image, coordinates):
            x, y = coordinates[..., 1], coordinates[..., 0]
            x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
            x2, y2 = x1 + 1, y1 + 1

            invalid = (x1 < 0) | (x2 >= image.shape[0]) | (y1 < 0) | (y2 >= image.shape[1])
            x[invalid], y[invalid], x1[invalid], x2[invalid], y1[invalid], y2[invalid] = 0, 0, 0, 0, 0, 0
            dx1, dx2, dy1, dy2 = (d[..., None] for d in (x - x1, x2 - x, y - y1, y2 - y))

            interpolation = (image[x1, y1] * dx2 * dy2 + image[x1, y2] * dx2 * dy1 + image[x2, y1] * dx1 * dy2 + image[x2, y2] * dx1 * dy1)

            return interpolation.astype(np.uint8)

        def transform(coordinates, T):
            temp = T @ np.hstack((coordinates, np.ones((coordinates.shape[0], 1)))).T
            return (temp / temp[-1, :]).T

        height_min, right_width_min, height, width = boundaries(image, T)
        coordinates, height_min = default_coordinates(height, width, height_min), np.abs(height_min)
        warped = mapping(image, transform(coordinates, inv(T)).reshape(height, width, -1)[..., :2]).transpose(1, 0, 2)
        return height_min, right_width_min, warped

    def blend(left, right, height_min):
        left_extend, height_max = np.zeros_like(right), min(height_min + left.shape[0], right.shape[0])
        left_extend[height_min:height_max, 0:left.shape[1]] = left[:height_max - height_min,]
        mask_left, mask_right = left_extend.sum(axis=2) > 0, right.sum(axis=2) > 0

        overlap = np.logical_and(mask_left, mask_right).astype(np.int32)
        x_min, x_max = np.where(overlap)[1].min().astype(np.int32), np.where(overlap)[1].max().astype(np.int32)
        overlap = overlap[:, :, np.newaxis]

        mask_alpha = np.zeros_like(right).astype(np.float32)
        mask_alpha[:, x_min:x_max + 1, :] = np.linspace(0, 1, num=x_max - x_min + 1).reshape(1, -1, 1).repeat(mask_alpha.shape[0], axis=0)
        mask_alpha *= overlap

        overlap_left, overlap_right = overlap * left_extend, overlap * right
        overlap_blended = ((1 - mask_alpha) * overlap_left + mask_alpha * overlap_right).astype(np.int32)

        blended = right.copy()
        blended[height_min:height_max, 0:left.shape[1]] = left[:height_max - height_min]
        blended = (1 - overlap) * blended + overlap * overlap_blended

        return blended

    (key_points_l, descriptions_l), (key_points_r, descriptions_r) = sift(left), sift(right)
    source_points, destination_points, matching = match(right, key_points_r, descriptions_r, left, key_points_l, descriptions_l, ratio)
    H = ransac(source_points, destination_points, num_sample, num_iteration, tolerance, require)
    height_min, right_width_min, warped = warp(right, H)
    blended = blend(left, warped, height_min)
    stitched = blended[height_min:min(height_min + left.shape[0], blended.shape[0]), :right_width_min]
    return map(lambda x: x.astype(np.uint8), (matching, warped, blended, stitched))


def output(matching, warped, blended, stitched, name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subname, image in {'matching': matching, 'warped': warped, 'blended': blended, 'stitched': stitched}.items():
        plt.imsave(f'{output_dir}/{name}_{subname}.png', image)


if __name__ == '__main__':
    args = parse_args()
    for name, left, right in load_data(args.data) + load_data(args.my_data):
        matching, warped, blended, stitched = stitch(left, right, args.ratio, args.num_sample, args.num_iteration, args.tolerance, args.require)
        output(matching, warped, blended, stitched, name, args.output)
