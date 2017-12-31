## Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/HOG_example.png
[image3]: ./images/sliding_windows.png
[video1]: ./output_video/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This is it ;)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell of the IPython notebook labeled "Define functions for features extraction".  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The final parameters I decided to use are: 

```color_space = 'YUV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' 
spatial_size = (32, 32) 
hist_bins = 16 
spatial_feat = True
hist_feat = True
hog_feat = True
```
I've already done some exprimentation during the quizes and then performed more experiments when working with the project. 
With those parameters I achive 99% accuracy with some cost on performance. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and colour features if you used them).

I trained a linear SVM in the code cell labeled "Train LinearSVC". I use StandardScaler to zero mean the features to combat overfitting

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I define the functions used for slide windows searching in the code cell labeled "Define function for sliding windows". I've decided to search for far object using a smaller windows starting from the middle of the image. I then use bigger windows taking into account the prespective of the view. 
Here is a list of search patterns representing start and stop for x and y in the image and a scale for determinging the size of the window. 

```
search_pattern = [
    [[128, 1152], [400, 528], 1],
    [[112, 1168], [400, 400 + 96], 1.5],
    ##[[None, None], [405, 405 + 128], 2],
    [[None, None], [410, 410 + 160], 2.5],
    [[None, None], [420, None], 3],
    ##[[None, None], [430, None], 3.5],
    [[None, None], [440, None], 4]   
]
```

Here is how it looks like:

![alt text][image3]

However in the end I don't use this technique, but instead get the hog features for the how image at once and then go through small windows throughout the bottom half of the image. 
This can be seen in the `find_cat` function defined in the code cell with label "Define Hog Sub-sampling windows search function"

The Hog Sub-sampling process and image more than 2 times faster than a simple windows search

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The result of the search on all test images can be found in `./output_images` directory. I consider this very good results with little false possitives

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./videos_output/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the `process_image` function defined in the code cell labeled "Process videos" I record the positions of positive detections in each frame of the video. I save the last 5 frames detections in a buffer variable. 
From the combined positive detections of the last 5 frames I create a heatmap and then thresholded that map to identify vehicle positions with a threshold of 3. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.
By combining the last 5 frame detections and using a threshold of 3 I filter false positive. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I developed this on a Mac Book Pro laptop. I consider this to be a decent computer, but it is clearlly uncapable of processing vehicle detectoin in real time. 

I think much better result will be achived with a larger dataset. For example if big truck is in the next lane I doupt the current implementation will handle it. A big truck would also cover more than a half of the image, so it would be interesting to see how a SVC detection will handle that. 

With a larger dataset I think a bigger search window can be used witch would lower the processing time. 

An additional improvement can be made with a bigger frame buffer and larger filter threshold that will filter even more false detections. However that might hide some positive, witch would be much worse. 
