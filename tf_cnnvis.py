# imports
import os
import numpy as np
import tensorflow as tf
from math import ceil, sqrt
from scipy.misc import imsave
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

is_Registered = False # prevente duplicate gradient registration

MAX_FEATUREMAP = 1024 # max number of feature
is_force = False # force computation for featuremaps > MAX_FEATUREMAP

MAX_IMAGES = 1 # max number of images


# register custom gradients
def _register_custom_gradients():
    """
    Register Custom Gradients.
    """

    global is_Registered

    if not is_Registered:
        # register LRN gradients
        @ops.RegisterGradient("Customlrn")
        def _CustomlrnGrad(op, grad):
            return grad

        # register Relu gradients
        @ops.RegisterGradient("GuidedRelu")
        def _GuidedReluGrad(op, grad):
            return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

        is_Registered = True


# save given graph object as meta file
def _save_model(graph):
    """
    Save the given TF graph at PATH = "./model/tmp-model"

    :param graph: 
        TF graph
    :type graph:  tf.Graph object

    :return: 
        Path to saved graph
    :rtype: String
    """

    PATH = "./model/tmp-model"
    _make_dir(path = os.path.dirname(PATH))

    with graph.as_default():
        with tf.Session() as sess:
            fake_var = tf.Variable([0.0], name = "fake_var")
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, PATH)

    return PATH + ".meta"


# All visualization of convolution happens here
def get_visualization(
    graph_or_path, 
    value_feed_dict, 
    input_tensor = None, 
    layers = 'r', 
    path_logdir = './Log', 
    path_outdir = "./Output", 
    force = False,
    n = 8):
    """
    cnnvis main api function

    :param graph_or_path: 
        TF graph or
        <Path-to-saved-graph> as String
    :type graph_or_path: tf.Graph object or String

    :param value_feed_dict: 
        Values of placeholders to feed while evaluting.
        dict : {placeholder1 : value1, ...}.
        list : [value1, value2, ...].
    :type value_feed_dict: dict or list

    :param input_tensor: 
        tf.tensor object which is an input to TF graph
    :type input_tensor: tf.tensor object (Default = None)

    :param layers: 
        'r' : Reconstruction from all the relu layers
        'p' : Reconstruction from all the pooling layers
        'c' : Reconstruction from all the convolutional layers
    :type layers: list or String (Default = 'r')

    :param path_outdir: 
        <path-to-dir> to save results into disk as images
    :type path_outdir: String (Default = "./Output")

    :param path_logdir: 
        <path-to-log-dir> to make log file for TensorBoard visualization
    :type path_logdir: String (Default = "./Log")

    :param force: 
        True to took of limit for number of featuremaps in a layer
    :type force: boolean (Default = False)

    :param n: 
        Number of gradient ops will be added to the graph to avoid redundent forward pass
    :type n: int (Default = 8)

    :return: 
        True if successful. False otherwise.
    :rtype: boolean
    """

    global is_force
    is_force = force

    is_success = True

    # map from keyword to layer type
    dict_layer = {'r' : "relu", 'p' : 'maxpool', 'c' : 'conv2d'}

    if isinstance(graph_or_path, tf.Graph):
        PATH = _save_model(graph_or_path)
    elif isinstance(graph_or_path, basestring):
        PATH = graph_or_path
    else:
        print("graph_or_path must be a object of graph or string.")
        is_success = False
        return is_success

    _register_custom_gradients() # register custom gradients

    with tf.Graph().as_default() as g:
        with g.gradient_override_map({'Relu': 'GuidedRelu', 'LRN': 'Customlrn'}): # overwrite gradients with custom gradients
            new_saver = tf.train.import_meta_graph(PATH) # Import graph

        if isinstance(layers, list):
            for layer in layers:
                if layer != None and layer.lower() not in dict_layer.keys():
                    is_success = _visualization_by_layer_name(g, value_feed_dict, input_tensor, layer, path_logdir, path_outdir, n)
                elif layer != None and layer.lower() in dict_layer.keys():
                    layer_type = dict_layer[layer.lower()]
                    is_success = _visualization_by_layer_type(g, value_feed_dict, input_tensor, layer_type, path_logdir, path_outdir, n)
                else:
                    print("Skipping %s . %s is not valid layer name or layer type" % (layer, layer))
        else:
            if layers != None and layers.lower() not in dict_layer.keys():
                is_success = _visualization_by_layer_name(g, value_feed_dict, input_tensor, layers, path_logdir, path_outdir, n)
            elif layers != None and layers.lower() in dict_layer.keys():
                layer_type = dict_layer[layers.lower()]
                is_success = _visualization_by_layer_type(g, value_feed_dict, input_tensor, layer_type, path_logdir, path_outdir, n)
            else:
                is_success = False
                print("%s is not a valid layer name or layer type." % (layers))

    return is_success

def _visualization_by_layer_type(
    graph, 
    value_feed_dict, 
    input_tensor, 
    layer_type, 
    path_logdir, 
    path_outdir,
    n):
    """
    Generate filter visullization from the layers which are of type layer_type

    :param graph: 
        TF graph 
    :type graph_or_path: tf.Graph object

    :param value_feed_dict: 
        Values of placeholders to feed while evaluting.
        dict : {placeholder1 : value1, ...}.
        list : [value1, value2, ...].
    :type value_feed_dict: dict or list

    :param input_tensor: 
        Where to reconstruct
    :type input_tensor: tf.tensor object (Default = None)

    :param layer_type: 
        Type of the layer. Supported layer types : 
        'r' : Reconstruction from all the relu layers
        'p' : Reconstruction from all the pooling layers
        'c' : Reconstruction from all the convolutional layers
    :type layer_type: String (Default = 'r')

    :param path_logdir: 
        <path-to-log-dir> to make log file for TensorBoard visualization
    :type path_logdir: String (Default = "./Log")

    :param path_outdir: 
        <path-to-dir> to save results into disk as images
    :type path_outdir: String (Default = "./Output")

    :param n: 
        Number of gradient ops will be added to the graph to avoid redundent forward pass
    :type n: int (Default = 8)

    :return: 
        True if successful. False otherwise.
    :rtype: boolean
    """

    is_success = True

    try:
        y = []
        # Loop through all operations and parse operations
        # for operations of type = layer_type
        for i in graph.get_operations():
            if layer_type.lower() == i.type.lower():
                y.append(i.name)

        for layer in y:
            is_success = _visualization_by_layer_name(graph, value_feed_dict, input_tensor, layer, path_logdir, path_outdir, n)
    except:
        is_success = False
        print("No Layer with layer type = %s" % (layer_type))

    return is_success


def _visualization_by_layer_name(
    graph, 
    value_feed_dict, 
    input_tensor, 
    layer_name, 
    path_logdir, 
    path_outdir,
    n):
    """
    Generate and store filter visullization from the a layer which has name layer_name

    :param graph: 
        TF graph 
    :type graph_or_path: tf.Graph object

    :param value_feed_dict: 
        Values of placeholders to feed while evaluting.
        dict : {placeholder1 : value1, ...}.
        list : [value1, value2, ...].
    :type value_feed_dict: dict or list

    :param input_tensor: 
        Where to reconstruct
    :type input_tensor: tf.tensor object (Default = None)

    :param layer_name: 
        'r' : Reconstruction from all the relu layers
        'p' : Reconstruction from all the pooling layers
        'c' : Reconstruction from all the convolutional layers
    :type layer_name: String (Default = 'r')

    :param path_logdir: 
        <path-to-log-dir> to make log file for TensorBoard visualization
    :type path_logdir: String (Default = "./Log")

    :param path_outdir: 
        <path-to-dir> to save results into disk as images
    :type path_outdir: String (Default = "./Output")

    :param n: 
        Number of gradient ops will be added to the graph to avoid redundent forward pass
    :type n: int (Default = 8)

    :return: 
        True if successful. False otherwise.
    :rtype: boolean
    """

    is_success = True

    x = []
    feed_dict = {}

    try:
        with graph.as_default() as g:
            # get op of name given in method argument layer_name
            op = g.get_operation_by_name(name = layer_name)
            op_tensor = op.outputs[0] # output tensor of the operation

            # create all ones tensor to provide grad_ys
            tensor_shape = op_tensor.get_shape().as_list() # get shape of tensor

            global MAX_FEATUREMAP
            # check for limit on number of feature maps
            if not is_force and tensor_shape[-1] > MAX_FEATUREMAP:
                print("Skipping. Too many featuremap. May cause memory errors.")
                return

            all_ones = tf.ones_like(op_tensor)

            # creating placeholders to pass masks
            mask = [tf.placeholder(tf.float32, [tensor_shape[-1]]) for i in range(n)]
            np_mask = np.identity(n = tensor_shape[-1])

            # creating feed_dict and find input tensors
            # if not provided
            X = None
            is_value_feed_dict = isinstance(value_feed_dict, dict)

            if is_value_feed_dict:
                for key_op in value_feed_dict.keys():
                    tmp = g.get_tensor_by_name(name = key_op.name)
                    feed_dict[tmp] = value_feed_dict[key_op]

                    if input_tensor != None and input_tensor.name == tmp.name:
                        X = tmp

            if X == None:
                for i in g.get_operations():
                    # parsing input placeholders
                    if "Placeholder" in i.name:
                        if not is_value_feed_dict:
                            x.append(i.outputs[0])
                            if input_tensor != None:
                                X = g.get_tensor_by_name(name = input_tensor.name)
                            else:
                                X = x[0]
                        else:
                            X = i.outputs[0]
                            break

            if not is_value_feed_dict:
                value_feed_dict[0] = value_feed_dict[0][:MAX_IMAGES] # only taking first MAX_IMAGES from given images array
                original_images = value_feed_dict[0]
                feed_dict = dict(zip(x, value_feed_dict)) # prepare feed_dict for the reconstruction
            else:
                feed_dict[X] = feed_dict[X][:MAX_IMAGES] # only taking first MAX_IMAGES from given images array
                original_images = feed_dict[X]

            # creating gradient ops
            reconstruct = [tf.gradients(op_tensor, X, grad_ys=tf.multiply(all_ones, mask[i]))[0] for i in range(n)]

            out = [] # list to store reconstructed images
            activations = [] # list to store activations

            # computing reconstruction
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # Execute the gradient operations in batches of 'n'
                for i in xrange(0, tensor_shape[-1], n):
                    c = 0
                    for j in range(n):
                        if (i + j) < tensor_shape[-1]:
                            feed_dict[mask[j]] = np_mask[i + j]
                            c += 1
                    if c > 0:
                        out.extend(sess.run(reconstruct[:c], feed_dict = feed_dict))

                # compute activations
                activations = sess.run(op_tensor, feed_dict = feed_dict)

                sess = None
    except:
        is_success = False
        print("No Layer with layer name = %s" % (layer_name))
        return is_success

    grid_images = _images_to_grid(out)

    act_shape = activations.shape
    activations = [np.expand_dims(im, axis = 3) for im in np.transpose(activations, (3, 0, 1, 2))]
    grid_activations = _images_to_grid(activations)

    # write results into disk
    if path_outdir != None:
        is_success = _write_into_disk(path_outdir, out, grid_images, grid_activations, layer_name)

    # write results into log file of TFBOARD
    if path_logdir != None:
        is_success = _write_into_log(path_logdir, original_images, out, grid_images, activations, grid_activations, layer_name)

    print("Reconstruction Completed for %s layer." % (layer_name))

    return is_success


def _write_into_disk(path_outdir, images, grid_images, grid_activations, layer):
    is_success = True

    path_out = os.path.join(path_outdir, layer.lower())

    for i in range(len(grid_images)):
        grid_image_path = os.path.join(path_out, "image_%d" % (i))
        grid_activation_path = os.path.join(path_out, "image_%d" % (i), "activations")

        is_success = _make_dir(grid_image_path)
        is_success = _make_dir(grid_activation_path)

        imsave(os.path.join(grid_image_path, "grid_image"), grid_images[i][0], format = "png")
        imsave(os.path.join(grid_activation_path, "grid_activation"), grid_activations[i][0,:,:,0], format = "png")

    # for j in range(len(images[i])):
    #     image_path = os.path.join(path_out, "image_%d" % (j))
    #     is_success = _make_dir(image_path)
    #     for i in range(len(images)):
    #         imsave(os.path.join(image_path, "feature_%d" % (i)), images[i][j], format = "png")

    return is_success


def _write_into_log(path_logdir, original_images, images, grid_images, activations, grid_activations, layer):
    is_success = True

    path_log = os.path.join(path_logdir, layer.lower().replace("/", "_"))
    is_success = _make_dir(path_log)

    with tf.Graph().as_default() as g:
        image1 = tf.placeholder(tf.float32, shape = [None, None, None, 3])
        image2 = tf.placeholder(tf.float32, shape = [None, None, None, 1])

        image_summary_t1 = tf.summary.image(name = "One_By_One_Deconv", tensor = image1, max_outputs = MAX_FEATUREMAP)
        image_summary_t2 = tf.summary.image(name = "All_At_Once_Deconv", tensor = image1, max_outputs = MAX_IMAGES)

        # image_summary_t3 = tf.summary.image(name = "One_By_One_Activations", tensor = image2, max_outputs = MAX_FEATUREMAP)
        image_summary_t4 = tf.summary.image(name = "All_At_Once_Activations", tensor = image2, max_outputs = MAX_IMAGES)

        image_summary_t5 = tf.summary.image(name = "Input_Images", tensor = image1, max_outputs = MAX_IMAGES)

        with tf.Session() as sess:
            summary1 = sess.run(image_summary_t1, feed_dict = {image1 : np.concatenate(images, axis = 0)})
            summary2 = sess.run(image_summary_t2, feed_dict = {image1 : np.concatenate(grid_images, axis = 0)})

            # summary3 = sess.run(image_summary_t3, feed_dict = {image2 : np.concatenate(activations, axis = 0)})
            summary4 = sess.run(image_summary_t4, feed_dict = {image2 : np.concatenate(grid_activations, axis = 0)})

            summary5 = sess.run(image_summary_t5, feed_dict = {image1 : original_images})
        try:
            file_writer = tf.summary.FileWriter(path_log, g) # create file writer
            # compute and write the summary

            file_writer.add_summary(summary1)
            file_writer.add_summary(summary2)

            # file_writer.add_summary(summary3)
            file_writer.add_summary(summary4)

            file_writer.add_summary(summary5)
        except:
            is_success = False
            print("Error occured int writting results into log file.")
        finally:
            file_writer.close() # close file writer

    return is_success


# helper function for convert_into_grid
def _images_to_grid(images):
    """
    Convert a list of arrays of images into a list of grid of images

    :param images: 
        a list of 4-D numpy arrays(each containing images)
    :type images: list

    :return: 
        a list of grids which are grid representation of input images
    :rtype: list
    """
    grid_images = []
    # if 'images' is not empty convert
    # list of images into grid of images
    if len(images) > 0:
        N = len(images)
        H, W, C = images[0][0].shape
        for j in range(len(images[0])):
            tmp = np.zeros((N, H, W, C))
            for i in range(N):
                tmp[i] = images[i][j]
            grid_images.append(np.expand_dims(convert_into_grid(tmp), axis = 0))

    return grid_images


# convert given array of images into large gird of images as a single image
def convert_into_grid(
    Xs, 
    ubound=255.0, 
    padding=1):
    """
    Convert 4-D numpy array into a grid image

    :param Xs: 
        A numpy array of images to make grid out of it
    :type Xs: 4-D numpy array (first axis contations an image)

    :param ubound: 
        upperbound for a image pixel value
    :type ubound: float (Default = 255.0)

    :param padding: 
        padding size between grid cells
    :type padding: int (Default = 1)

    :return: 
        A grid of input images 
    :rtype: 3-D numpy array
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                grid[y0:y1, x0:x1] = image_normalization(Xs[next_idx], ubound=ubound)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid.astype('uint8')


# normalize image (min-max normalization)
def image_normalization(image, ubound = 255.0, epsilon = 1e-7):
    """
    Min-Max image normalization. Convert pixle values in range [0, ubound]

    :param image: 
        A numpy array to normalize
    :type image: 3-D numpy array

    :param ubound: 
        upperbound for a image pixel value
    :type ubound: float (Default = 255.0)

    :param epsilon: 
        for computational stability
    :type epsilon: float (Default = 1e-7)

    :return: 
        A normalized image
    :rtype: 3-D numpy array
    """
    img_min = np.min(image)
    img_max = np.max(image)

    return (((image - img_min) * ubound) / (img_max - img_min + epsilon)).astype('uint8')


# dir exists or not
def _is_dir_exist(path):
    return os.path.exists(path)


# make dir at given path
def _make_dir(path):
    is_success = True

    # if dir is not exist make one
    if not _is_dir_exist(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            is_success = False


    return is_success