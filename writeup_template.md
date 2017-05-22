#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[class_dist]: ./examples/class_dist.png "data class distribution"
[lenet]: ./examples/modifiedLeNet.jpeg "Modified LeNet Arch"
[x1]: ./new_images/1x.png "new Image 1"
[x2]: ./new_images/2x.png "new Image 2"
[x3]: ./new_images/3x.png "new Image 3"
[x4]: ./new_images/5x.png "new Image 4"
[x5]: ./new_images/6x.png "new Image 5"
[aug]: ./examples/aug.png "augmented image"
[ori]: ./examples/aug.png "original image"
[gray]: ./examples/gray.png "original image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rbcorx/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set without augmentation is 80% of 39209 ~ 31367
* The size of the validation set is without augmentation is 20% of 39209 ~ 7842
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the various classes.

![alt text][class_dist]

As we can see the images are highly skewed towards certain classes (by more than 10x!) implying the dataset is highly unbalanced.
To get optimum results, we need to balance the examples of each class.

We can do so by two methods:

- Trim down excess data for over-represented classes
- Augment the data to include more examples of under represented classes

I have chosen to augment the data with new images as explained in the upcoming section.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale due to two reasons:

1. Accuracy doesn't seem to be affected as expected as colors don't act as a major differentiator between the signs.
2. Changing to grayscale speeds up training.

Here is an example of a traffic sign image after grayscaling.

![alt text][gray]

As a last step, I normalized the image data because of two reasons:

1. Normalized features result in faster and better gradient descent optimization to the minima.
2. Any differences in brightness of images will be diminished as that is not what we want the model to focus on.

I decided to generate additional data to augment the dataset by randomly introducing some jitter to images and adding them back to the dataset so classes could be more well balanced.
I ensure atleast 800 examples of each class are present and I add jitter by randomly translating, scaling, warping, and changing the brightness of the image by a little amount only so that the images are still well recongnizable.

I have drawn inspiration from a blogpost by a Udacian for the same.

After this, the data distribution looks more balanced:

![alt text][class_dist]

Here is an example of an original image and an augmented image:

![alt text][ori]
![alt text][aug]

The difference between the original data set and the augmented data set is that it has random jittering done to it in the form as mentioned before.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I began by reusing the same architecture from the LeNet Lab. This model worked quite well and gave around 93% accuracy after image preprocessing like grayscaling and normalization.

Then, after experimentation, I used a modifield LeNet architecture specially designed for traffic sign classification problems.
I referenced the Sermanet/LeCunn traffic sign classification journal article. The improvement in the accuracy was apparent and it was well worth the effort.

The paper doesn't go into great detail about how the model is implemented so I discussed with and referenced some implementations by udacity students before finalizing my own. The layers are as follows:

![alt text][lenet]



1. 5x5 conv (32x32x1 -> 28x28x6)
2. ReLU
3. 2x2 max pool (28x28x6 -> 14x14x6)
4. 5x5 conv (14x14x6 -> 10x10x16)
5. ReLU
6. 2x2 max pool (10x10x16 -> 5x5x16)
7. 5x5 convolution (5x5x6 -> 1x1x400)
8. ReLu
9. Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
10. Concatenate flattened layers to a single size-800 layer
11. Dropout layer
12. Fully connected layer (800 -> 43)



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer with following configurations:

mu: 0
sigma: 0.1
dropout probability: 0.5
batch_size: 128
epochs: 15
learning rate: 0.0009

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I experimented in this project by implementing different network architectures to get a feel of what works and tried to enhance them somehow by tinkering with well known implementations to get better insight. I started with LeNet and moved on to a modified LeNet architecture specially desined for traffic sign classification.

Adding the dropout layers really made a differnce when it came to the test set accuracy as the model was able to generalize better.

I also experimented with data augmentation techniques to balance the skewed data distribution between the classes and they have had a remarkable improvement on the accuracy of the model.

I tuned the learning rate between 0.001 - 0.0005 to see how that effected the training process and got the best results with a bit lower learning rate but on a higher epoch which is expected behaviour in general.

My final model results were:
* training set accuracy of ~ 99%
* validation set accuracy of 98.7%
* test set accuracy of 93.5%

The model's results on test set accuracy is a clear indication that the modified LeNet is working as expected.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][x1] ![alt text][x2] ![alt text][x3]
![alt text][x4] ![alt text][x5]

The first image might be difficult to classify because

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy was found to be 20% on the images with 1 correct out of 5.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
h
I noticed that my images are a bit brighter and will take a different range in the color channels, maybe a range that the model is not trained on. The GTSRB dataset states that the images "contain a border of 10 % around the actual traffic sign (at least 5 pixels) to allow for edge-based approaches" and the images that I used do not all include such a border. This could have led to the model's confusion.

The top 5 softmax probabilities are as follows:

[ 0.03666489  0.0403071   0.04249405  0.04303512  0.04337623]
[ 0.0481916   0.05685914  0.05723968  0.05844232  0.09018107]
[ 0.04342517  0.04447183  0.05767543  0.07445375  0.10666957]
[ 0.03556685  0.03615285  0.03903935  0.04429515  0.04737494]
[ 0.0530977   0.07001691  0.08208198  0.10873241  0.11040591]



The code for making predictions on my final model is located in the cell with the headline "Analyze Performance and Output Top 5 Softmax Probabilities For Each Image Found on the Web" of the Ipython notebook.

For the first image, the model is not sure at all what it's seeing as there is no peak in the probabilities
[ 0.03666489  0.0403071   0.04249405  0.04303512  0.04337623]

For the third image, the model correctly predicts the sign but the probability is only 10.6% for that class
[ 0.04342517  0.04447183  0.05767543  0.07445375  0.10666957]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


