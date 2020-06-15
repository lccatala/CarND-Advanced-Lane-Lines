## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distorted_chessboard_img]: ./camera_cal/calibration1.jpg "Distorted"
[undistorted_chessboard_img]: ./undistorted_images/calibration1.jpg "Undistorted"
[pipeline_input_img]: ./test_images/straight_lines1.jpg "Straight Lines 1 distorted"
[undistorted_road_img]: ./output_images/undistorted.jpg "Straight Lines 1 undistorted"
[sobel_img]: ./output_images/sobel_thresholded.jpg "Color and Sobel thresholding"
[blurred_img]: ./output_images/blurred_sobel_thresholded.jpg "Blurred color and Sobel thresholding"
[unwarped_points]: ./output_images/unwarped_points.jpg "src_points in original image"
[warped_points]: ./output_images/warped_points.jpg "src_points in warped image"
[sliding_windows_img]: ./output_images/sliding_windows.jpg "Sliding windows"
[separated_lines_img]: ./output_images/separated_lines.jpg "Separated lines"
[plotted_lane_img]: ./output_images/plotted_lane.jpg "Plotted lane"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. All the code is in the file `pipeline.py` 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In this step we correct the project camera's distortion using a set of images provided in the folder `camera_cal`.

We start by preparing 2 lists, `objpoints` and `imgpoints`, which represent 3D points in a real-world space and 2D points in each image's plane respectively. Next, we prepare template data `objp` to be appended to objpoints using NumPy's `mgrid()` function. We then iterate through each calibration image and perform the following steps:

1. We convert the image to grayscale and use it to detect the positions of chessboard corners using OpenCV's `cvtColor()` and `findChessboardCorners()` functions.
2. We then append the detected corners to `imgpoints` and our template `objp` to `objpoints`.
3. With both `objpoints` and `imgpoints`, we use the function `calibrateCamera()` from OpenCV to obtain the camera's transformation matrix `M`

We then applied this distortion correction to a test image and obtained the following output

Distorted | Undistorted
-|-
![alt text][distorted_chessboard_img] | ![alt text][undistorted_chessboard_img]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Distorted | Undistorted
-|-
![alt text][pipeline_input_img] | ![alt text][undistorted_road_img]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

We used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 209 through 211 in `pipeline.py`).  Here's an example of my output for this step. We applied this thresholding to a perspective-transformed image (see subsection 3).

We performed the following operations:
- For yellow-lane filtering, HSV in range `(200, 100, 100)` and `30, 255, 255` for Hue, Saturation and Value respectively.
- For white-lane filtering, HSV in range `(0, 0, 223)` and `((255, 32, 255)` for Hue, Saturation and Value respectively.
- Sobel gradient with `kernel_size=5` and `threshold=[50, 255]` for x gradient in channels L and S.

![alt text][sobel_img]

Then we blurred it to reduce noise

![alt text][blurred_img]

(note: selected pixels are shown in yellow and discarded ones in purple)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_perspective()`, which appears in lines 25 through 32 in the file `pipeline.py`  .  The `warp_perspective()` function takes as inputs an image (`img`), as well as source (`src_points`) and destination (`dst_points`) points.  I chose to hardcode the source and destination points in the following manner:

```python
src_points = [[200, 720], # Bottom left
              [570, 470], # Top left
              [720, 470], # Top right
              [1130, 720]] # Bottom right
dst_points = [[320, 720], # Bottom left
              [320, 0], # Top left
              [980, 0], # Top right
              [980, 720]] # Bottom right
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720        | 
| 570, 470      | 320, 0          |
| 720, 470      | 980, 0          |
| 1130, 720     | 980, 720        |

I verified that my perspective transform was working as expected by drawing the `src_points` and `dst_points` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Original image | Warped image
-|-
![alt text][unwarped_points] | ![alt text][warped_points]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

We identified each lane-line's pixels with the sliding-windows method. We started by finding the most likely `x` values for each lane to start at with a histogram of the lower part of the image (`find_ends()` function at lines 174 through 181). Then we used those `x` values as starting points for the function `find_lines()` (lines 115 through 172). This last one will iterate through the image creating of windows around the line centers and use them to select each lane's pixels based on their position.

We can see this process represented here, where the sliding windows are represented by clear blue squares.
![alt text][sliding_windows_img]

Then we used those pixel positions to fit a quadratic polynomial to each line (lines 226 and 233).

![alt text][separated_lines_img]

It's important to note that our pipeline will keep the line from the previous frame unless new pixels are detected, preventing some issues like `None` values.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is computed in lines 245 through 247, making use of function `measure_curvature_radius()` (lines 183 through 192).

We convert the lines from pixel space to real-world space (meters) and then calculate the radious of the curve using the expression `curvature = (1 + (2 A*y*y_mpp + B^2)^1.5) / |2A|`, where `y_mpp` is a constant defining the ammount of meters per pixel in the `y` axis. We calculate each lane's curvature separately and then average them to get an approximation to the real lane's curvature.

To calculate the deviation of the vehicle from the lane center, we first find the lower ends of the detected lane lines (value of `x` for the highest `y` of each polynomial) and average them to get the center of the lane. We assume the camera is centered in the car, so the car is at the center of the image. With these values we can obtain the distance between both of them, then applying the convenient correction factor `1/2.81362` to obtain a measurement in centimeters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

We implemented this step in lines 266 through 281 in my code in `pipeline.py` . We also plotted the left and right lane pixels in red and blue respectively. For the final image to look like a regular camera image we undid the perspective transformation. This was achieved with OpenCV's `warpPerspective()` and the inverse transformation matrix `M` we got when calibrating the camera.

Here is an example of my result on a test image:

![alt text][plotted_lane_img]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of the project was finding an adequate thresholding system that would get enough white and yellow pixels from the lane lines without adding in too much noise. This was achieved by applying Sobel after transforming the image perspective, since it made detecting lines based on their `x` gradient, which added some details that were missed with "regular" color masking.

While this assignment was designed to be as simple as possible, it has some obvious shortcommings. In order to make it perform correctly in more challenging images, our pipeline should be enhanced. Here are some ways to do this:
- Fully convolutional networks for lane detection
- Line tracking through multiple frames to make them more stable.
- A way to filter out reflections in other cars, since these can be captured by our thresholds and are especially tricky to deal with
