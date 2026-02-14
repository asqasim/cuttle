# **AI Building Footprint Extraction from UAV Imagery**

**Looking for setup and usage instructions?**  
Check out the [**Usage & Installation Guide (GUIDE.md)**](https://www.google.com/search?q=./GUIDE.md) for step-by-step commands to run the training and inference scripts.

## **1\. Project Overview**

This project automates the digitization of building footprints from high-resolution drone (UAV) orthomosaics. Using a deep learning approach, we convert raw .tif imagery into geospatial vector data (.shp), reducing manual digitization time by approximately 90%.

## **2\. Technical Architecture**

The system utilizes a **U-Net** architecture, a convolutional neural network (CNN) designed specifically for fast and precise image segmentation.

* **Model Backbone:** MobileNetV2 (Pre-trained on ImageNet).  
  * *Why?* It provides a lightweight, efficient "encoder" capable of high performance on CPU-based systems without sacrificing significant accuracy.  
* **Framework:** PyTorch with the Segmentation Models Pytorch (SMP) library.  
* **Input Resolution:** $256 \\times 256$ pixel tiles.  
* **Total Tiles:** 490 (Derived from a 150-house sample site).

## **3\. The Training Process**

Instead of training from scratch, we employed **Transfer Learning** to leverage pre-existing visual patterns:

1. **Data Preparation:** Geo-referenced imagery was "tiled" into 256px windows. To handle class imbalance, a **Balanced Sampling** strategy ensured the model prioritized tiles containing roof structures.  
2. **Augmentation:** We used the Albumentations library to perform random flips, rotations, and color shifts. This effectively "multiplied" our dataset, teaching the model to recognize roofs regardless of orientation or lighting.  
3. **Loss Function:** We used **Dice Loss** rather than standard cross-entropy.  
   * *Technical Reason:* Dice Loss is mathematically optimized to handle "spatial overlap," making it superior for finding small objects (roofs) in large backgrounds (fields).  
4. **Optimization:** Adam Optimizer with a learning rate of $10^{-4}$.

## **4\. Inference & Post-Processing**

AI predictions are "pixel-based" (probabilistic). To convert these into GIS-ready vectors, we implemented a three-stage pipeline:

1. **Probability Thresholding:** Pixels with $\>0.5$ confidence are classified as "Building."  
2. **Morphological Cleaning:** Using OpenCV to perform Opening (noise removal) and Closing (filling holes within roofs).  
3. **Vectorization:** Converting the raster mask into OGC-compliant geometries (Polygons) using Rasterio and GeoPandas, maintaining the original Coordinate Reference System (CRS).

## **5\. Performance & Results**

* **Hardware:** Trained on CPU (Dell Latitude 3380\) in \~20 hours.  
* **Accuracy:** Successfully generalized to out-of-distribution test sites (images not seen during training).
