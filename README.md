# Road-Lane-Semantic-segmentation
Automatic and accurate detection of road/lane boundaries is very important for autonomous driving vehicles. After applying several heuristic based methods, the next step is to apply machine learning for this task. Semantic segmentation is the task of individually classfying each pixel in the scene to fit into predefined road categories.  
A common approach to train a fully convolutional network is to leverage an existing classification model. For that purpose we suggest to use the [popular VGG16 classifier](https://neurohive.io/en/popular-networks/vgg16/).  
We implement the model introduced by [Long et Al](https://arxiv.org/pdf/1605.06211.pdf). I have trained the model on the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).  
[semantic_segmentation_fcn8.ipynb](https://github.com/mmarouen/Road-Lane-Semantic-segmentation/blob/master/semantic_segmentation_fcn8.ipynb) notebook can be used both for training FCN networks and for inference on road images.
## Inference
For inference `h5` model files must be used along with the predictor function provided in `ss_utils.py`:  
```python
import ss_utils
import cv2
from keras.models import load_model
fcn32model=load_model('fcn32.h5')
some_image = cv2.imread(image_path, 1)
bgd_img,segmented_img=ss_utils.fcn_predict(fcn8model,some_image,nClasses,InputHeight,InputWidth)
```
Below some samples of the predicted results  
![example1](https://github.com/mmarouen/Road-Lane-Semantic-segmentation/blob/master/images/pred1.png)
![example1](https://github.com/mmarouen/Road-Lane-Semantic-segmentation/blob/master/images/pred2.png)
## Project setup
* **Google colab environment**
Access to a gpu resource is mandatory to train the networks. For that purpose I suggest using [google colab](https://colab.research.google.com/notebooks/welcome.ipynb) which provides a free and high quality development environment for deep learning project. Of course it's possible to use other deep learning platforms but it's not covered for this project.  
* **Versions**
keras version 2.2.4
Python version 3.6.7
tensorflow version 1.13.1
* **Data and weights access**
Necessary data to build the model are VGG16 weights and Kitti dataset. There's no need to download any data in advance since I provide gdrive access to the necessary files.
## Training
The output of the training is a `h5` which can be called to perform inference on random road images. It's important to mention that FCN models does not expect a fixed image size but a minimum image size should be around 300x500. I set the model to resize every input image into a 176x576 image. The segmented image will also be in the same size.  
## Next steps
* Train on larger datasets
* For now inference runs on a per image basis, so I prepare batch predictions
* The end purpose is to run the algorithm in a 20 fps context. For that end we need to switch to faster algorithms during inference
### Contact
For any additional questions or remarks, please let me know @ <azzouz.marouen@gmail.com>
