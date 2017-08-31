# imports
import os
import time

import numpy as np

from six.moves import range
from six import string_types

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from skimage.restoration import denoise_tv_bregman

from .utils import *
from .utils import config


is_Registered = False # prevent duplicate gradient registration
# map from keyword to layer type
dict_layer = {'r' : "relu", 'p' : 'maxpool', 'c' : 'conv2d'}
units = None

configProto = tf.ConfigProto(allow_soft_placement = True)

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
	PATH = os.path.join("model", "tmp-model")
	make_dir(path = os.path.dirname(PATH))

	with graph.as_default():
		with tf.Session(config=configProto) as sess:
			fake_var = tf.Variable([0.0], name = "fake_var")
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			saver.save(sess, PATH)

	return PATH + ".meta"


# All visualization of convolution happens here
def _get_visualization(graph_or_path, value_feed_dict, input_tensor, layers, path_logdir, path_outdir, method = None):
	"""
	cnnvis main api function

	:param graph_or_path:
		TF graph or
		<Path-to-saved-graph> as String
	:type graph_or_path: tf.Graph object or String

	:param value_feed_dict:
		Values of placeholders to feed while evaluting.
		dict : {placeholder1 : value1, ...}.
	:type value_feed_dict: dict or list

	:param input_tensor:
		tf.tensor object which is an input to TF graph
	:type input_tensor: tf.tensor object (Default = None)

	:param layers:
		Name of the layer to visualize or layer type.
		Supported layer types :
		'r' : Reconstruction from all the relu layers
		'p' : Reconstruction from all the pooling layers
		'c' : Reconstruction from all the convolutional layers
	:type layers: list or String (Default = 'r')

	:param path_logdir:
		<path-to-log-dir> to make log file for TensorBoard visualization
	:type path_logdir: String (Default = "./Log")

	:param path_outdir:
		<path-to-dir> to save results into disk as images
	:type path_outdir: String (Default = "./Output")

	:return:
		True if successful. False otherwise.
	:rtype: boolean
	"""
	is_success = True

	if isinstance(graph_or_path, tf.Graph):
		PATH = _save_model(graph_or_path)
	elif isinstance(graph_or_path, string_types):
		PATH = graph_or_path
	else:
		print("graph_or_path must be a object of graph or string.")
		is_success = False
		return is_success

	is_gradient_overwrite = method == "deconv"
	if is_gradient_overwrite:
		_register_custom_gradients() # register custom gradients

	with tf.Graph().as_default() as g:
		if is_gradient_overwrite:
			with g.gradient_override_map({'Relu': 'GuidedRelu', 'LRN': 'Customlrn'}): # overwrite gradients with custom gradients
				sess = _graph_import_function(PATH)
		else:
			sess = _graph_import_function(PATH)

		if not isinstance(layers, list):
			layers =[layers]

		for layer in layers:
			if layer != None and layer.lower() not in dict_layer.keys():
				is_success = _visualization_by_layer_name(g, value_feed_dict, input_tensor, layer, method, path_logdir, path_outdir)
			elif layer != None and layer.lower() in dict_layer.keys():
				layer_type = dict_layer[layer.lower()]
				is_success = _visualization_by_layer_type(g, value_feed_dict, input_tensor, layer_type, method, path_logdir, path_outdir)
			else:
				print("Skipping %s . %s is not valid layer name or layer type" % (layer, layer))

	return is_success


def _graph_import_function(PATH):
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph(PATH) # Import graph
		new_saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(PATH)))
		return sess

def _visualization_by_layer_type(graph, value_feed_dict, input_tensor, layer_type, method, path_logdir, path_outdir):
	"""
	Generate filter visualization from the layers which are of type layer_type

	:param graph:
		TF graph
	:type graph_or_path: tf.Graph object

	:param value_feed_dict:
		Values of placeholders to feed while evaluting.
		dict : {placeholder1 : value1, ...}.
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

	:return:
		True if successful. False otherwise.
	:rtype: boolean
	"""
	is_success = True

	layers = []
	# Loop through all operations and parse operations
	# for operations of type = layer_type
	for i in graph.get_operations():
		if layer_type.lower() == i.type.lower():
			layers.append(i.name)

	for layer in layers:
		is_success = _visualization_by_layer_name(graph, value_feed_dict, input_tensor, layer, method, path_logdir, path_outdir)
	return is_success

def _visualization_by_layer_name(graph, value_feed_dict, input_tensor, layer_name, method, path_logdir, path_outdir):
	"""
	Generate and store filter visualization from the layer which has the name layer_name

	:param graph:
		TF graph
	:type graph_or_path: tf.Graph object

	:param value_feed_dict:
		Values of placeholders to feed while evaluting.
		dict : {placeholder1 : value1, ...}.
	:type value_feed_dict: dict or list

	:param input_tensor:
		Where to reconstruct
	:type input_tensor: tf.tensor object (Default = None)

	:param layer_name:
		Name of the layer to visualize
	:type layer_name: String

	:param path_logdir:
		<path-to-log-dir> to make log file for TensorBoard visualization
	:type path_logdir: String (Default = "./Log")

	:param path_outdir:
		<path-to-dir> to save results into disk as images
	:type path_outdir: String (Default = "./Output")

	:return:
		True if successful. False otherwise.
	:rtype: boolean
	"""
	start = -time.time()
	is_success = True

	# try:
	parsed_tensors = parse_tensors_dict(graph, layer_name, value_feed_dict)
	if parsed_tensors == None:
		return is_success
	op_tensor, x, X_in, feed_dict = parsed_tensors

	is_deep_dream = True
	with graph.as_default() as g:
		# computing reconstruction
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			X = X_in
			if input_tensor != None:
				X = get_tensor(graph = g, name = input_tensor.name)
			# original_images = sess.run(X, feed_dict = feed_dict)

			results = None
			if method == "act":
				# compute activations
				results = _activation(graph, sess, op_tensor, feed_dict)
			elif method == "deconv":
				# deconvolution
				results = _deconvolution(graph, sess, op_tensor, X, feed_dict)
			elif method == "deepdream":
				# deepdream
				is_success = _deepdream(graph, sess, op_tensor, X, feed_dict, layer_name, path_outdir, path_logdir)
				is_deep_dream = False

			sess = None
	# except:
	# 	is_success = False
	# 	print("No Layer with layer name = %s" % (layer_name))
	# 	return is_success

	if is_deep_dream:
		is_success = write_results(results, layer_name, path_outdir, path_logdir, method = method)

	start += time.time()
	print("Reconstruction Completed for %s layer. Time taken = %f s" % (layer_name, start))

	return is_success


# computing visualizations
def _activation(graph, sess, op_tensor, feed_dict):
	with graph.as_default() as g:
		with sess.as_default() as sess:
			act = sess.run(op_tensor, feed_dict = feed_dict)
	return act
def _deconvolution(graph, sess, op_tensor, X, feed_dict):
	out = []
	with graph.as_default() as g:
		# get shape of tensor
		tensor_shape = op_tensor.get_shape().as_list()

		with sess.as_default() as sess:
			# creating placeholders to pass featuremaps and
			# creating gradient ops
			featuremap = [tf.placeholder(tf.int32) for i in range(config["N"])]
			reconstruct = [tf.gradients(tf.transpose(tf.transpose(op_tensor)[featuremap[i]]), X)[0] for i in range(config["N"])]

			# Execute the gradient operations in batches of 'n'
			for i in range(0, tensor_shape[-1], config["N"]):
				c = 0
				for j in range(config["N"]):
					if (i + j) < tensor_shape[-1]:
						feed_dict[featuremap[j]] = i + j
						c += 1
				if c > 0:
					out.extend(sess.run(reconstruct[:c], feed_dict = feed_dict))
	return out
def _deepdream(graph, sess, op_tensor, X, feed_dict, layer, path_outdir, path_logdir):
	tensor_shape = op_tensor.get_shape().as_list()

	with graph.as_default() as g:
		n = (config["N"] + 1) // 2
		feature_map = tf.placeholder(dtype = tf.int32)
		tmp1 = tf.reduce_mean(tf.multiply(tf.gather(tf.transpose(op_tensor),feature_map),tf.diag(tf.ones_like(feature_map, dtype = tf.float32))), axis = 0)
		tmp2 = 1e-3 * tf.reduce_mean(tf.square(X), axis = (1, 2 ,3))
		tmp = tmp1 - tmp2
		t_grad = tf.gradients(ys = tmp, xs = X)[0]

		with sess.as_default() as sess:
			input_shape = sess.run(tf.shape(X), feed_dict = feed_dict)
			tile_size = input_shape[1 : 3]
			channels = input_shape[3]

			lap_in = tf.placeholder(np.float32, name='lap_in')
			laplacian_pyramid = lap_normalize(lap_in, channels, scale_n=config["NUM_LAPLACIAN_LEVEL"])

			image_to_resize = tf.placeholder(np.float32, name='image_to_resize')
			size_to_resize = tf.placeholder(np.int32, name='size_to_resize')
			resize_image = tf.image.resize_bilinear(image_to_resize, size_to_resize)

			end = len(units)
			for k in range(0, end, n):
				c = n
				if k + n > end:
					c = end - ((end // n) * n)
				img = np.random.uniform(size = (c, tile_size[0], tile_size[1], channels)) + 117.0
				feed_dict[feature_map] = units[k : k + c]

				for octave in range(config["NUM_OCTAVE"]):
					if octave > 0:
						hw = np.float32(img.shape[1:3])*config["OCTAVE_SCALE"]
						img = sess.run(resize_image, {image_to_resize : img, size_to_resize : np.int32(hw)})

						for i, im in enumerate(img):
							min_img = im.min()
							max_img = im.max()
							temp = denoise_tv_bregman((im - min_img) / (max_img - min_img), weight = config["TV_DENOISE_WEIGHT"])
							img[i] = (temp * (max_img - min_img) + min_img).reshape(img[i].shape)

					for j in range(config["NUM_ITERATION"]):
						sz = tile_size
						h, w = img.shape[1:3]
						sx = np.random.randint(sz[1], size=1)
						sy = np.random.randint(sz[0], size=1)
						img_shift = np.roll(np.roll(img, sx, 2), sy, 1)
						grad = np.zeros_like(img)
						for y in range(0, max(h-sz[0]//2,sz[0]), sz[0] // 2):
							for x in range(0, max(h-sz[1]//2,sz[1]), sz[1] // 2):
									feed_dict[X] = img_shift[:, y:y+sz[0],x:x+sz[1]]
									try:
										grad[:, y:y+sz[0],x:x+sz[1]] = sess.run(t_grad, feed_dict=feed_dict)
									except:
										pass

						lap_out = sess.run(laplacian_pyramid, feed_dict={lap_in:np.roll(np.roll(grad, -sx, 2), -sy, 1)})
						img = img + lap_out
				is_success = write_results(img, (layer, units, k), path_outdir, path_logdir, method = "deepdream")
				print("%s -> featuremap completed." % (", ".join(str(num) for num in units[k:k+c])))
	return is_success


# main api methods
def activation_visualization(graph_or_path, value_feed_dict, input_tensor = None, layers = 'r', path_logdir = './Log', path_outdir = "./Output"):
	is_success = _get_visualization(graph_or_path, value_feed_dict, input_tensor = input_tensor, layers = layers, method = "act",
		path_logdir = path_logdir, path_outdir = path_outdir)
	return is_success
def deconv_visualization(graph_or_path, value_feed_dict, input_tensor = None, layers = 'r', path_logdir = './Log', path_outdir = "./Output"):
	is_success = _get_visualization(graph_or_path, value_feed_dict, input_tensor = input_tensor, layers = layers, method = "deconv",
		path_logdir = path_logdir, path_outdir = path_outdir)
	return is_success
def deepdream_visualization(graph_or_path, value_feed_dict, layer, classes, input_tensor = None, path_logdir = './Log', path_outdir = "./Output"):
	is_success = True
	if isinstance(layer, list):
		print("Please only give classification layer name for reconstruction.")
		return False
	elif layer in dict_layer.keys():
		print("Please only give classification layer name for reconstruction.")
		return False
	else:
		global units
		units = classes
		is_success = _get_visualization(graph_or_path, value_feed_dict, input_tensor = input_tensor, layers = layer, method = "deepdream",
			path_logdir = path_logdir, path_outdir = path_outdir)
	return is_success
