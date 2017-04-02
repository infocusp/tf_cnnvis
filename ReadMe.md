# tf_cnnvis

tf_cnnvis is a CNN visualization library based on the paper [Visualizing and Understanding Convolutional Networks](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) by Matthew D. Zeiler and Rob Fergus. We use the [TensorFlow](https://www.tensorflow.org/) library to reconstruct the input images from different layers of the convolutional neural network. The generated images are displayed in [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

<img src="https://bitbucket.org/repo/Lyk4Mq/images/4115906191-images.jpg" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/780477117-reconstructed_1.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/1721308804-reconstructed_2.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/856735034-reconstructed_3.png" width="196" height="196">


<img src="https://bitbucket.org/repo/Lyk4Mq/images/1446219986-Lena.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3167061216-lena_reconstructed_1.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3614692134-lena_reconstructed_2.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/124519901-lena_reconstructed_3.png" width="196" height="196">


<img src="https://bitbucket.org/repo/Lyk4Mq/images/995747735-mancoffee.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/755140547-man_reconstructed_1.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3249576627-man_reconstructed_2.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/744562091-man_reconstructed_3.png" width="196" height="196">

Figure 1: Original image and the reconstructed versions from maxpool layer 1,2 and 3 of Alexnet generated using tf_cnnvis. 

![tensorboard.png](https://bitbucket.org/repo/Lyk4Mq/images/2741459243-tensorboard.png)

[View Full size](https://bitbucket.org/repo/Lyk4Mq/images/2005224096-tensorboard.png)




## Requirements:
* Tensorflow (>= 1.0)
* numpy
* scipy
* h5py
* wget
* Pillow
* six

If you are using pip you can install these with

```pip install tensorflow numpy scipy h5py wget Pillow six```

## Setup script
Clone the repository

```
#!bash

git clone https://github.com/InFoCusp/tf_cnnvis.git
```

And run 

```
#!bash
sudo pip install setuptools
sudo python setup.py install
sudo python setup.py clean
```


## API
**tf_cnnvis.get_visualization(graph_or_path, value_feed_dict, input_tensor=None, layers='r', path_logdir='./Log', path_outdir='./Output', force=False, n=8)** 

The function to generate the visualizations of the input image reconstructed from the feature maps of a given layer.
#### Parameters
* graph_or_path (tf.Graph object or String) – TF graph or [Path-to-saved-graph] as String containing the CNN.
* value_feed_dict (dict or list) – Values of placeholders to feed while evaluating the graph
    * dict : {placeholder1 : value1, ...}
    * list : [value1, value2, ...]
* input_tensor (tf.tensor object (Default = None)) – tf.tensor where we pass the input images to the TF graph
* layers (list or String (Default = 'r')) – 
    * ‘r’ : Reconstruction from all the relu layers 
    * ‘p’ : Reconstruction from all the pooling layers 
    * ‘c’ : Reconstruction from all the convolutional layers
* path_outdir (String (Default = "./Output")) – [path-to-dir] to save results into disk as images
* path_logdir (String (Default = "./Log")) – [path-to-log-dir] to make log file for TensorBoard visualization
* force (boolean (Default = False)) – True to took of limit for number of featuremaps in a layer
* n (int (Default = 8)) – Number of gradient ops computed in parallel. Increasing this number increases amount of parallelization (and hence reduces computation time) at the cost of higher RAM usage. 

#### Returns
* is_success (boolean) – True if the function ran successfully. False otherwise

## To visualize in TensorBoard
To start Tensorflow, run the following command in console

```
#!bash

tensorboard --logdir=./Log
```

and under tensorboard homepage look under the *Images* tab

## Additional helper functions
### image_normalization(image, ubound=255.0, epsilon=1e-07)
Performs Min-Max image normalization. Transforms the pixel values to range [0, ubound]
#### Parameters
* image (3-D numpy array) – A numpy array to normalize
* ubound (float (Default = 255.0)) – upperbound for a image pixel value
* epsilon (float (Default = 1e-7)) – for computational stability

#### Returns
* norm_image (3-D numpy array) – The normalized image

### convert_into_grid(Xs, padding=1, ubound=255.0)
Convert 4-D numpy array into a grid of images
#### Parameters
* Xs (4-D numpy array (first axis contations an image)) – The 4D array of images to put onto grid
* padding (int (Default = 1)) – Spacing between grid cells
* ubound (float (Default = 255.0)) – upperbound for a image pixel value


#### Returns
* (3-D numpy array) – A grid of input images
