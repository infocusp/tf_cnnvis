# tf_cnnvis

tf_cnnvis is a CNN visualization library based on the paper [Visualizing and Understanding Convolutional Networks](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) by Matthew D. Zeiler and Rob Fergus. We use the [TensorFlow](https://www.tensorflow.org/) library to reconstruct the input images from different layers of the convolutional neural network. The generated images are displayed in [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

## Requirements:
* Tensorflow (>= 1.0)
* Numpy
* Scipy
* h5py

## Setup script
Clone the repository

```
#!bash

git clone https://bitbucket.org/infocusp/tf_cnnvis.git
```

And run 

```
#!bash

python setup.py
```


## API
### get_visualization(graph_or_path, value_feed_dict, input_tensor=None, layers='r', path_logdir='./Log', path_outdir='./Output', force=False, n=8) 
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
* n (int (Default = 8)) – Number of gradient ops will be added to the graph to avoid redundent forward pass

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
Min-Max image normalization. Convert pixle values in range [0, ubound]
#### Parameters
* image (3-D numpy array) – A numpy array to normalize
* ubound (float (Default = 255.0)) – upperbound for a image pixel value
* epsilon (float (Default = 1e-7)) – for computational stability

#### Return
* (3-D numpy array) – A normalized image

### convert_into_grid(Xs, ubound=255.0, padding=1)
Convert 4-D numpy array into a grid image
#### Parameters
* Xs (4-D numpy array (first axis contations an image)) – A numpy array of images to make grid out of it
* ubound (float (Default = 255.0)) – upperbound for a image pixel value
* padding (int (Default = 1)) – padding size between grid cells

#### Return
* (3-D numpy array) – A grid of input images