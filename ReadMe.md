# tf_cnnvis

tf_cnnvis is a CNN visualization library which you can to better understand your own CNNs. We use the [TensorFlow](https://www.tensorflow.org/) library at the backend and the generated images are displayed in [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). We have implemented 2 CNN visualization techniques so far:

1) Based on the paper [Visualizing and Understanding Convolutional Networks](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) by Matthew D. Zeiler and Rob Fergus. The goal here is to reconstruct the input image from the information contained in any given layers of the convolutional neural network. Here are a few examples

<img src="https://bitbucket.org/repo/Lyk4Mq/images/4115906191-images.jpg" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/780477117-reconstructed_1.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/1721308804-reconstructed_2.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/856735034-reconstructed_3.png" width="196" height="196">


<img src="https://bitbucket.org/repo/Lyk4Mq/images/1446219986-Lena.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3167061216-lena_reconstructed_1.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3614692134-lena_reconstructed_2.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/124519901-lena_reconstructed_3.png" width="196" height="196">


<img src="https://bitbucket.org/repo/Lyk4Mq/images/995747735-mancoffee.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/755140547-man_reconstructed_1.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3249576627-man_reconstructed_2.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/744562091-man_reconstructed_3.png" width="196" height="196">

Figure 1: Original image and the reconstructed versions from maxpool layer 1,2 and 3 of Alexnet generated using tf_cnnvis. 




2) CNN visualization based on [Deep dream](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb). Here's the relevant [blog post](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) explaining the technique. In essence, it attempts to construct an input image that maximizes the activation for a given output. We present some samples below:  

<img src="https://bitbucket.org/repo/Lyk4Mq/images/302562334-deep_999.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/586427523-deep_9.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3658923710-deep_24.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/2708104439-deep_385.png" width="196" height="196">

Carbonara, Ibex, Elephant, Ostrich


<img src="https://bitbucket.org/repo/Lyk4Mq/images/3678905801-deep_993.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/1049045916-deep_970.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/2473296459-deep_934.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/1596202689-deep_933.png" width="196" height="196">

Cheese burger, Tennis ball, Fountain pen, Clock tower


<img src="https://bitbucket.org/repo/Lyk4Mq/images/3869149957-deep_738.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/4283505926-deep_915.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/2629248471-deep_14.png" width="196" height="196"> <img src="https://bitbucket.org/repo/Lyk4Mq/images/3971745292-deep_22.png" width="196" height="196">

Cauliflower, Baby Milk bottle, Sea lion, Dolphin


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
**tf_cnnvis.activation_visualization(graph_or_path, value_feed_dict, input_tensor=None, layers='r', path_logdir='./Log', path_outdir='./Output')** 

The function to generate the activation visualizations of the input image at the given layer.
#### Parameters
* graph_or_path (tf.Graph object or String) – TF graph or [Path-to-saved-graph] as String containing the CNN.
* value_feed_dict (dict) – Values of placeholders to feed while evaluating the graph
    * dict : {placeholder1 : value1, ...}

* input_tensor (tf.tensor object (Default = None)) – tf.tensor where we pass the input images to the TF graph
* layers (list or String (Default = 'r')) – 
    * ‘r’ : Reconstruction from all the relu layers 
    * ‘p’ : Reconstruction from all the pooling layers 
    * ‘c’ : Reconstruction from all the convolutional layers
* path_outdir (String (Default = "./Output")) – [path-to-dir] to save results into disk as images
* path_logdir (String (Default = "./Log")) – [path-to-log-dir] to make log file for TensorBoard visualization

#### Returns
* is_success (boolean) – True if the function ran successfully. False otherwise

**tf_cnnvis.deconv_visualization(graph_or_path, value_feed_dict, input_tensor=None, layers='r', path_logdir='./Log', path_outdir='./Output')** 

The function to generate the visualizations of the input image reconstructed from the feature maps of a given layer.
#### Parameters
* graph_or_path (tf.Graph object or String) – TF graph or [Path-to-saved-graph] as String containing the CNN.
* value_feed_dict (dict) – Values of placeholders to feed while evaluating the graph
    * dict : {placeholder1 : value1, ...}

* input_tensor (tf.tensor object (Default = None)) – tf.tensor where we pass the input images to the TF graph
* layers (list or String (Default = 'r')) – 
    * ‘r’ : Reconstruction from all the relu layers 
    * ‘p’ : Reconstruction from all the pooling layers 
    * ‘c’ : Reconstruction from all the convolutional layers
* path_outdir (String (Default = "./Output")) – [path-to-dir] to save results into disk as images
* path_logdir (String (Default = "./Log")) – [path-to-log-dir] to make log file for TensorBoard visualization

#### Returns
* is_success (boolean) – True if the function ran successfully. False otherwise

**tf_cnnvis.deepdream_visualization(graph_or_path, value_feed_dict, layer, classes, input_tensor=None, path_logdir='./Log', path_outdir='./Output')** 

The function to generate the visualizations of the input image reconstructed from the feature maps of a given layer.
#### Parameters
* graph_or_path (tf.Graph object or String) – TF graph or [Path-to-saved-graph] as String containing the CNN.
* value_feed_dict (dict) – Values of placeholders to feed while evaluating the graph
    * dict : {placeholder1 : value1, ...}

* layer (String) - name of a layer in TF graph
* classes (List) - list featuremap index for the class classification layer
* input_tensor (tf.tensor object (Default = None)) – tf.tensor where we pass the input images to the TF graph 
* path_outdir (String (Default = "./Output")) – [path-to-dir] to save results into disk as images
* path_logdir (String (Default = "./Log")) – [path-to-log-dir] to make log file for TensorBoard visualization

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
### tf_cnnvis.utils.image_normalization(image, ubound=255.0, epsilon=1e-07)
Performs Min-Max image normalization. Transforms the pixel values to range [0, ubound]
#### Parameters
* image (3-D numpy array) – A numpy array to normalize
* ubound (float (Default = 255.0)) – upperbound for a image pixel value

#### Returns
* norm_image (3-D numpy array) – The normalized image

### tf_cnnvis.utils.convert_into_grid(Xs, padding=1, ubound=255.0)
Convert 4-D numpy array into a grid of images
#### Parameters
* Xs (4-D numpy array (first axis contations an image)) – The 4D array of images to put onto grid
* padding (int (Default = 1)) – Spacing between grid cells
* ubound (float (Default = 255.0)) – upperbound for a image pixel value


#### Returns
* (3-D numpy array) – A grid of input images

