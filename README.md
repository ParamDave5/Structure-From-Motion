# SFM
This repository implements reconstruction of a 3D scene and simulataneously obtaining the camera poses of a monocular camera w.r.t the scene. 
The algorithm of the same is:

* Feature Matching and Outlier rejection using RANSAC

*  Estimating Fundamental Matrix using the 8-point method
* Estimating Essential Matrix from Fundamental Matrix 
* Estimate Camera Pose from Essential Matrix. Decompose Essential Matrix 
* Check for Cheirality Condition using Linear Triangulation
* Perform Non Linear Traingulation 
* Linear and Non-Linear Perspective-n-Point
 * Bundle Adjustment

<img src ="BundleAdjustment56.png" width=400/>

## Run Instructions

```
python Wrapper.py --Path PATH_TO_DATA --Filtered False
```
