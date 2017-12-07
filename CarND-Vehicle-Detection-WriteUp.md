
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/new/hog.png
[image2]: ./examples/new/sliding_windows.jpg
[image3]: ./examples/new/heatmap.jpg
[image4]: ./examples/new/final_output.jpg
[image4]: ./examples/new/false_pos.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Project Writeup - Vehicle Detection


**I structured the write up document the same way as the jupyter notebook to make it easier to reference code cells to certain tasks.

## Pre-work

### Load training images

First thing was to import the training images and store them to lists.

## Functions for feature extraction, frame searching and box finding

### 1. Extract features from training images

#### 1.1 Color features

First thing to implement was the function to extract color features from images. Therefore histograms for each color channel were created and concatenated to a single feature vector.

#### 1.2 HOG features

The hog features from each image were extracted by using the parameters `orientation`, `pix_per_cell` and `cell_per_block`.
I implemented the code in a way that an easy switching between different color channels or all color channels is feasible.


The following image shows an example of a car image and the corresponding HOG image with the parameters `orientations = 8`, `pix_per_cell = 8`, `cell_per_block = 2`:

![hog features][image1]

I tried extracting the HOG features from each image channel and from all channels simultaneously. At the end I stuck with `all` channels, because it showed the best performance from my point of view.

As color space `YCrCB` showed the best performance.


#### 1.3 Combine and normalize

To extract features from the images a function was created to combine each feature extraction step. The possibility to quickly change to other color spaces is implemented as well. 
Each feature vector is appended to a list and finally concatenated to a single feature vector containing all extracted features.

### 2. Search frames/images and find boxes

#### 2.1 Sliding windows

This function creates windows and stores them to a list. The window size is adjustable. This functon is explained in more depth later on.

#### 2.2 Draw boxes

This function takes in the image and a list of boxes to draw in the image. Color and thickness can be adjusted. To insert the boxes in the image the rectangle() function of openCV is used.

#### 2.3 Single image extraction function

This function is used for testin porpuses. The functionality is the same as described in section 1 of this write up. It just takes in a single image and extracts the features. Having a function for that is helpful to explore different color spaces and HOG-parameters.

#### 2.4 Search windows

To search in the input image or frame of a video for regions that match with extracted features from training images a sliding windows approach is used. Therefore a function was created that takes the image, the trained classifier, the color space to work with, the input parameters for feature extraction, and the hog_channel.
If one of the window's features match the features of the LinearSVM for a car the found window is append to a list and returned.

### 3. Training the classifier

As classifier I chose a linear support vector machine. I thought as wellabout implementing a deep learning approach like YOLO (because everyone is talking about it at the moment) but kept with the SVM due to the easy handling. Nevertheless, I want to try to implement YOLO as well after the project submission.


- First I extracted the features for each training image of cars and non-cars and stored them to variables
- Next step was arranging the training features with the corresponding classes
- Normalize them with help of `StandardScaler()` by `sklearn.preprocessing`
- Split into training and test data with a 0.2 ratio
- Fit the training data to the classifier `LinearSVC()` object

I tested the accuracy of the classifier with the `.score` function and achieved 98.95%.



### 4. Implementation of Sliding Window Search

To cover almost all possible driving distances of other cars I implemented a scale factor in the slinding window search function.
The function can be called multiple times to run the sliding window search over the same image or frame of a video multiple times with different scales.
Instead of a overlapping of the windows I implemented a variable to define the cells per step.
`cells_per_step = 2`
`nxsteps = (nxblocks - nblocks_per_window) // cells_per_step`
`nysteps = (nyblocks - nblocks_per_window) // cells_per_step`


#### 4.1 Find cars function

The function is coded in the cell of the section 4.1 of the jupyter notebook.

The output with the input parameters:
- y start value: 400
- x start value: 656
- scale: 1.5
can be seen in the following image.

![find cars function][image2]


#### 4.2 Add heatmaps to cut out false-positives

To avoid false-positives affect the performance of the pipeline I implemented the heatmap approach that was shown in the lesson.
Therefore each box is transformed to a heatmap and by applying a variable `threshold` false positives can be cut out.
It is likely that false-positives are single events on the image. These single events shall be cut out by applying a `threshold` of `1`.
That means that single boxes are cut out of the further evaluation.
A detected false-positive is shown on the following image.

![false-positive][image5]

A visualization of the heatmap of the image from section 4.1 is shown below.

![heatmaps][image3]


The final output is shown below:

![final output][image4]


However, my first implementation of the pipeline showed not the best results on the project video.
The found boxes were kind of wobbling around. That is why I further implemented a smoothin' function.

Therefore I created a Class object to easy handle each frame of the video.

### 5. Vehicle Detector Class Object

The `VehicleDetector()` is initialized with its parameters for the features extract functions. As well variables for heatmap, frame count (to smooth the found boxes), kernel for dilation (cv2 function) and for the Linear Support Vector Machine are defined.
The `heat images` are stored by using the `deque` function by `collections` library. A deque can be seen as a queue with 2 ends. (I quite liked that explanation :-) )
The Class object has the following class functions:
1. Find cars with smoothing over the last `n` frames
2. heatmap evaluation
3. draw labeled boxes as `@staticmethod` to decouple it from the actual states of the class variables

This approach increased the performance enormous!

## Video Implementation

The video I created with my `VehicleDetector` class can accessed by the link below.
Here's a [link to my video result for the project](./output_videos/project_video_output_smooth.mp4)

#### Project video

Here's a [link to my video result](./output_videos/project_video_output_smooth.mp4)


#### Combining it with the advanced lane finding project

I thought that a combination of both project would be a great overall result of Term 1 so I decided to combine it.
Therefore I copied the functions I coded for the last project and stored the camera calibration matrixes with pickle and loaded them into the new project.
I came up with the `process_image_comb` function which takes in an image (frame of the video) and
1. Search for the lane lines and subsequent
2. Detect vehicles

The output video was quite good, one or two false positives were identified but I was satisfied with it!

Here's a [link to a video with combined lane line search and vehicle detections](./output_videos/project_video_output_comb.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

I took the approach to implement a linear support vector machine because this classifier provides a good performance a is as well fast.
The accuracy that was reached was sufficient for the project video but might fail in more advanced situations. In my opinion a (fast) deep learning approach like YOLO will provide enhanced performance in vehicle detection. In my opinion the usage of deep learning approaches is very straight forward and has still a huge space for optimization as a ongoing field of research. 
I guess my pipeline in its actual state will need some more robust filter for false-positives instead of the heatmap approach, maybe a redundant one, and as well of course a lot more training data to distinguish even better between cars and non-cars.

Nevertheless I will continue to work on this project beyond the submission. I still want to implement a deep learning approach on this and hope I find some time for it during the upcoming Christmas holidays.