# Road-Lane-Semantic-segmentation
Automatic and accurate detection of road/lane boundaries is very important for autonomous driving vehicles. After applying several heuristic based methods, the next step is to apply machine learning for this task. Semantic segmentation is the task of individually classfying each pixel in the scene to fit into predefined road categories.  
A common approach to train a fully convolutional network is to leverage an existing classification model. For that purpose we suggest to use the [popular VGG16 classifier](https://neurohive.io/en/popular-networks/vgg16/).  
We implement the model introduced by [Long et Al](https://arxiv.org/pdf/1605.06211.pdf)  
Below some samples of the predicted results  
![example1](https://github.com/mmarouen/Road-Lane-Semantic-segmentation/blob/master/images/pred1.png)  
![example1](https://github.com/mmarouen/Road-Lane-Semantic-segmentation/blob/master/images/pred2.png)  
## Project setup
* Google colab 
## Prepare data for training
## Calling VGG16 weights
## Calling built FCN models
