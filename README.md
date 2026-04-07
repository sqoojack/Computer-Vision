# NYCU Computer Vision Course

This repository contains five major projects developed for the Computer Vision course at National Yang Ming Chiao Tung University (NYCU). Each project explores different facets of image processing, geometric vision, and deep generative models.

## Lab 1: Camera Calibration
The goal of this lab was to implement a complete camera calibration pipeline from scratch using Zhang’s method to determine the internal characteristics and spatial orientation of a camera. 
* **Technical Highlights**: The implementation involved finding chessboard corners using OpenCV, then manually deriving the homography matrix ($H$) for each image using Singular Value Decomposition (SVD). Camera intrinsics ($K$) were retrieved through Cholesky decomposition of the $B$ matrix, and extrinsic parameters (rotation $R$ and translation $t$) were calculated for each view.
* **Task Completion**: Successfully implemented the calibration process without relying on high-level `cv2.calibrateCamera` functions.
* **Performance Result**: The custom implementation yielded extrinsic parameter visualizations that were almost identical to the results provided by OpenCV's built-in functions, demonstrating the accuracy of the mathematical derivation.

## Lab 2: Hybrid Images & Colorizing the Russian Empire
This project focused on frequency-domain image processing and multi-resolution analysis through three distinct tasks: Hybrid Images, Image Pyramids, and historical photo colorization.
* **Technical Highlights**:
    * **Hybrid Imaging**: Created visual effects that change with viewing distance by combining a low-pass filtered image (Gaussian/Ideal) with a high-pass filtered one using 2D Fast Fourier Transform (FFT).
    * **Image Pyramids**: Developed Gaussian pyramids for smoothing/downsampling and Laplacian pyramids for capturing edge and residual information.
    * **Colorization**: Aligned R, G, and B channels from glass plate images using normalized cross-correlation to minimize squared differences between channels.
* **Task Completion**: Successfully generated hybrid photos and colorized the "Russian Empire" collection with high alignment precision.
* **Performance Result**: Experimental observations showed that the Gaussian filter produced significantly smoother transitions compared to the Ideal filter, while the number of pyramid layers was found to be critical for alignment accuracy.

## Lab 3: Automatic Panoramic Image Stitching
This project implemented a complete pipeline for stitching multiple overlapping images into a seamless wide-angle panorama.
* **Technical Highlights**: The process utilized SIFT (Scale-Invariant Feature Transform) for interest point detection and feature description. Feature matching was conducted between image pairs, followed by RANSAC (Random Sample Consensus) to robustly estimate the homography matrix ($H$). The final stage involved warping the images into a shared coordinate system.
* **Task Completion**: Finished the automatic stitching process from scratch, transforming independent images into a unified panoramic view.

## Lab 4: Structure from Motion (SfM)
The objective was to reconstruct 3D structures from sequences of 2D images by estimating camera poses and triangulating points.
* **Technical Highlights**:
    * **Matching**: Used SIFT and KNN matching with a ratio distance filter to find correspondence across images.
    * **Estimation**: Implemented a RANSAC-optimized normalized 8-point algorithm to calculate the fundamental matrix ($F$) and compute Sampson error for outlier rejection.
    * **Reconstruction**: Derived the essential matrix ($E$) to obtain four possible $R$ and $t$ solutions, then applied 3D triangulation to find the correct pose where points lie in front of both cameras.
* **Task Completion**: Developed a full pipeline from 2D point matching to 3D mesh triangulation visualized in Blender.
* **Performance Result**: The implementation successfully underscored the importance of the distance ratio in matching, with colorful and textured objects yielding superior 3D reconstruction quality compared to monotone objects.

## Final Project: SG-I2V Trajectory Control and DiT Experimentation
The final project involved analyzing and attempting to modify SG-I2V, a zero-shot trajectory control framework for image-to-video generation.
* **Technical Highlights**: The project analyzed diffusion features and explored modifying the standard U-Net structure in SG-I2V to a DiT (Diffusion Transformer) model, incorporating Self-Attention, Cross-Attention, and Layer Normalization blocks. It utilized high-frequency preserved post-processing (FFT/IFFT) to update latents while maintaining detail.
* **Task Completion**: Analyzed trajectory control effectiveness across people, animals, and vehicle categories and evaluated the feasibility of DiT-based video diffusion.
* **Performance Result**: 
    * **Direction Accuracy**: The framework achieved 100% movement direction correctness for vehicles in short-frame (7-frame) generation.
    * **Generation Quality**: Quantitative analysis showed that 65% (13/20) of generated vehicle videos were rated as "Good," significantly outperforming people (30%) and animal (45%) categories.