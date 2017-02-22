# Self-Driving Car Engineer Nanodegree
## Computer Vision: Advanced Lane Finding

### Overview

The objective of this project is to identify lane lines using traditional computer vision techniques. In order to achieve this, we have designed a computer vision pipeline as depicted below.

<p align="center">
 <img src="./images/pipeline.png">
</p>

The input of our pipeline would be images or video clips. Input images or videos will go through various pipeline stages (will be discussed in the latter part of this document) and produce annotated video and images as given below.

Image | Video
------------|---------------
![image_output](./images/image_output.png) | ![video_output](./images/video_output.gif)

Next, we are going to describe our pipeline stages starting from the Camera Calibrator. The sample input and output of the Camera Calibrator pipeline stage is given below.

### Camera Calibrator

Camera calibration logic is encapsulated in **`CameraCalibrator`** class in the **`advanced_lane_finding.py`** module. This class's constructor takes following arguments.

1. A list of camera images which we are going to use for camera calibration. (Usually, we use chessboard images)
2. Number of corners in X direction
3. Number of corners in Y direction
4. A boolean flag, if it is True, we do camera calibration and store those calibration data. 

The public method of this **`CameraCalibrator`** class is **`undistort`** and it takes a distorted image as the input and produces an undistorted image.

<p align="center">
 <img src="./images/camera_calibrator.png">
</p>

Following image shows before and after applying distortion correction to a typical road image.

<p align="center">
 <img src="./images/undistorted.png">
</p>

### Warp Transformer

The second step of the lane line finding pipeline is "perspective transformation" step. In computer vision,  an image perspective is a phenomenon where objects appear smaller the further away they are from a viewpoint.   

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. In the project we are going to use bird’s-eye view transform that allows us to view a lane from above; this will be useful for calculating the lane curvature in step 4.

Warped operation is encapsulated in **`PerspectiveTransformer`** class of the **`advanced_lane_finding.py`** package located in **`$PROJECT_HOME/src`** folder. In order to create an instance of **`PerspectiveTransformer`**  class, we need to provide four source and destination points. In order to clearly visible lane lines, we have selected following source and destination points. 

|Source Points | Destination Points|
|--------------|-------------------|
|(253, 697)    |   (303, 697)      |
|(585, 456)    |   (303, 0)        |
|(700, 456)    |   (1011, 0)       |
|(1061, 690)   |   (1011, 690)     |


I verified the performance of my perspective transformation by transforming an image (**`../output_images/undistorted_test_images/straight_lines2.jpg`**) using above source and destination points as given below.

<p align="center">
 <img src="./images/warp.png">
</p>

### Binarizer

Correctly identifying lane line pixels is one of the main tasks of this project. In order to identify lane line, we have used three main techniques namely:

1. Sobel operation in X direction
2. Color thresholding in S component of the HLS color space.
3. Color thresholding in L component of the HLS color space.

These three operations are encapsulated in the method called **`binarize`** in **`advanced_lane_finding.py`** module located in **`$PROJECT_HOME/src`** folder.

Also, below shows the `binarize` operation applied to a sample image.

<p align="center">
 <img src="./images/binarizer.png">
</p>

### Lane Line Extractor

Now we have extracted lane line pixels. So next step would be calculating the road curvature and other necessary quantities (such as how much the vehicle off from the center of the lane)

In order to calculate road curvature, we have used two methods as given below.

1. **`naive_lane_extractor(self, binary_warped)`** (inside the **Line** class in advanced_line_finding module)
2. **`smart_lane_extractor(self, binary_warped)`** (inside the **Line** class in advanced_line_finding module

Both methods take a binary warped image (similar to one shown above) and produce X coordinates of both left and right lane lines. `naive_lane_extractor(self, binary_warped)` method uses **sliding window** to identify lane lines from the binary warped image and then uses a second order polynomial estimation technique to calculate road curvature.

The complete description of our algorithm in given in [Advanced Lane Line Finding](https://github.com/upul/CarND-Advanced-Lane-Lines/blob/master/src/Advanced_Lane_Line_Finding.ipynb) notebook.

The output of lane line extractor algorithm is visualize in following figure.

<p align="center">
 <img src="./images/lane_pixels.png">
</p>

When it comes to video processing we start (with the very first image in the video) with **``naive_lane_extractor(self, binary_warped``** method. Once we have identified lane lines we moves to the **``smart_lane_extractor(self, binary_warped)``** which doesn't blindly search entire image but uses lane lines identify in the previous image in the video.

### Lane Line Curvature Calculator

We have created a utility mehtod called **``def calculate_road_info(self, image_size, left_x, right_x)``** inside of the **``Line``** class. It takes  size of the image (**``image_size``**), left lane line pixels (**``left_x``**) and right lane line pixels (**``right_x``**) as arguments returns following information.

1. **``left_curverad``** : Curvature of the left road line in meters.
2. **``right_curverad``** : Curvature of the right road line in meters.
3. **``lane_deviation``** : Deviation of the vehicle from the center of the line in meters.

### Highlighted Lane Line and Lane Line Information

In order to easy work with images as well as videos, we have created a Python class called **`Line`** inside the **`advanced_lane_finding`** module. It encapsulates all the methods we described above and few more helper methods as well.  

The key method of **`Line`** class is **`process(self, image)`** method. It takes a single image as the input. That image goes through the image process pipeline as described above and finally produces another image which contains highlighted lane line, lane line curvature information and the content of the original image.

The following section shows how we can use it with road images and videos.

```python
src_image = mpimg.imread('../test_images/test5.jpg')

line = advanced_lane_finding.Line()
output_image = line.process(src_image)

plt.figure(figsize=(10, 4))
plt.axis('off')
plt.imshow(output_image)
plt.show()
```
<p align="center">
 <img src="./images/image_output.png">
</p>

```python
output_file = '../processed_project_video.mp4'
input_file = '../project_video.mp4'
line = advanced_lane_finding.Line()

clip = VideoFileClip(input_file)
out_clip = clip.fl_image(line.process) 
out_clip.write_videofile(output_file, audio=False)
```

<p align="center">
    <a href="https://www.youtube.com/watch?v=ZNmvFZJRKWA">
        <img src="https://img.youtube.com/vi/ZNmvFZJRKWA/0.jpg" alt="video output">
    </a>
</p>

### Conclusions and Future Directions

This was my very first computer vision problem.  This project took relatively a large amount of time when it compares to other (deep learning) projects. The hyper-parameter tuning process in my computer vision pipeline was tedious and time-consuming. Unfortunately, our pipeline didn't generalize across diffrent road conditions and I think it is one of the main drawbacks of traditional computer vision approach to self-driving cars (and mobile robotics in general)

When it comes to extensions and future directions, I would like to highlight followings.

1. As the very first step, in the future, I would like to improve my computer vision pipeline. Presently, it works with project video only. But, I would like to improve it in order to work with other videos as well.
2. Secondly,  I would like to explore machine learning (both traditional and new deep learning) approaches suitable to address lane finding problem. 

