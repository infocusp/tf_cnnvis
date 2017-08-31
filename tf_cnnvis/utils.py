import os
import time
import datetime
from math import ceil, sqrt

import numpy as np
from scipy.misc import imsave

from six.moves import range
from six import iteritems

import tensorflow as tf


k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
K5X5 = k[ : , : , None , None ] / k.sum() * np.eye(3, dtype = np.float32)
channels = 1

# optional hyperparameter settings
config = {
	"N" : 8,
	"EPS" : 1e-7,
	"K5X5" : K5X5,
	"MAX_IMAGES" : 1,
	"NUM_OCTAVE" : 3,
	"STEP_SIZE" : 1.0,
	"NUM_ITERATION" : 50,
	"OCTAVE_SCALE" : 1.4,
	"MAX_FEATUREMAP" : 1024,
	"FORCE_COMPUTE" : False,
	"TV_DENOISE_WEIGHT" : 2.0,
	"NUM_LAPLACIAN_LEVEL" : 4,
	"REGULARIZATION_STRENGTH" : 1e-3
}
def reset_config():
	config = {
		"N" : 8,
		"EPS" : 1e-7,
		"K5X5" : K5X5,
		"MAX_IMAGES" : 1,
		"NUM_OCTAVE" : 3,
		"STEP_SIZE" : 1.0,
		"NUM_ITERATION" : 50,
		"OCTAVE_SCALE" : 1.4,
		"MAX_FEATUREMAP" : 1024,
		"FORCE_COMPUTE" : False,
		"TV_DENOISE_WEIGHT" : 2.0,
		"NUM_LAPLACIAN_LEVEL" : 4,
		"REGULARIZATION_STRENGTH" : 1e-3
	}
def get_config():
	return config
def set_config(config_dict):
	config = config_dict


# parse tensors and prepare feed dict
def parse_tensors_dict(graph, layer_name, value_feed_dict):
	x = []
	feed_dict = {}
	with graph.as_default() as g:
		# get op of name given in method argument layer_name
		op = get_operation(graph = g, name = layer_name)
		op_tensor = op.outputs[0] # output tensor of the operation
		tensor_shape = op_tensor.get_shape().as_list() # get shape of tensor

		# check for limit on number of feature maps
		if not config["FORCE_COMPUTE"] and tensor_shape[-1] > config["MAX_FEATUREMAP"]:
			print("Skipping. Too many featuremaps. May cause memory errors.")
			return None

		# creating feed_dict and find input tensors
		X_in = None

		# find tensors of value_feed_dict
		# in current graph by name
		for key_op, value in iteritems(value_feed_dict):
			tmp = get_tensor(graph = g, name = key_op.name)
			feed_dict[tmp] = value
			x.append(tmp)

		X_in = x[0]
		feed_dict[X_in] = feed_dict[X_in][:config["MAX_IMAGES"]] # only taking first MAX_IMAGES from given images array
	return op_tensor, x, X_in, feed_dict


# written results into disk as well as logfile of TensorBoard
def _write_activation(activations, layer, path_outdir, path_logdir):
	is_success = True

	act_shape = activations.shape
	if len(act_shape) == 2:
		grid_activations = [np.expand_dims(image_normalization(convert_into_grid(im[:,np.newaxis,np.newaxis,np.newaxis], padding=0)), axis = 0) for im in activations]
	else:
		activations = [np.expand_dims(im, axis = 3) for im in np.transpose(activations, (3, 0, 1, 2))]
		activations = _im_normlize(activations)
		grid_activations = _images_to_grid(activations)

	# write into disk
	path_out = os.path.join(path_outdir, layer.lower().replace("/", "_"))

	for i in range(len(grid_activations)):
		time_stamp = time.time()
		time_stamp = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d_%H-%M-%S')

		grid_activation_path = os.path.join(path_out, "image_%s" % (time_stamp), "activations")
		is_success = make_dir(grid_activation_path)
		imsave(os.path.join(grid_activation_path, "grid_activation.png"), grid_activations[i][0,:,:,0], format = "png")

	# write into logfile
	path_log = os.path.join(path_logdir, layer.lower().replace("/", "_"))
	is_success = make_dir(path_log)

	with tf.Graph().as_default() as g:
		image = tf.placeholder(tf.float32, shape = [None, None, None, None])
		image_summary_t = tf.summary.image(name = "All_At_Once_Activations", tensor = image, max_outputs = config["MAX_IMAGES"])

		with tf.Session() as sess:
			summary = sess.run(image_summary_t, feed_dict = {image : np.concatenate(grid_activations, axis = 0)})

		try:
			file_writer = tf.summary.FileWriter(path_log, g) # create file writer
			file_writer.add_summary(summary)
		except:
			is_success = False
			print("Error occured int writting results into log file.")
		finally:
			file_writer.close() # close file writer
	return is_success
def _write_deconv(images, layer, path_outdir, path_logdir):
	is_success = True

	images = _im_normlize(images)
	grid_images = _images_to_grid(images)

	# write into disk
	path_out = os.path.join(path_outdir, layer.lower().replace("/", "_"))

	for i in range(len(grid_images)):
		time_stamp = time.time()
		time_stamp = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d_%H-%M-%S')

		grid_image_path = os.path.join(path_out, "image_%s" % (time_stamp))
		is_success = make_dir(grid_image_path)
		if grid_images[i].shape[-1] == 1:
			imsave(os.path.join(grid_image_path, "grid_image.png"), grid_images[i][0,:,:,0], format = "png")
		else:
			imsave(os.path.join(grid_image_path, "grid_image.png"), grid_images[i][0], format = "png")

	# for j in range(len(images[i])):
	# 	image_path = os.path.join(path_out, "image_%d" % (j))
	# 	is_success = make_dir(image_path)
	# 	for i in range(len(images)):
	# 		imsave(os.path.join(image_path, "feature_%d" % (i)), images[i][j], format = "png")

	# write into logfile
	path_log = os.path.join(path_logdir, layer.lower().replace("/", "_"))
	is_success = make_dir(path_log)

	with tf.Graph().as_default() as g:
		image = tf.placeholder(tf.float32, shape = [None, None, None, None])

		image_summary_t1 = tf.summary.image(name = "One_By_One_Deconv", tensor = image, max_outputs = config["MAX_FEATUREMAP"])
		image_summary_t2 = tf.summary.image(name = "All_At_Once_Deconv", tensor = image, max_outputs = config["MAX_IMAGES"])

		with tf.Session() as sess:
			summary1 = sess.run(image_summary_t1, feed_dict = {image : np.concatenate(images, axis = 0)})
			summary2 = sess.run(image_summary_t2, feed_dict = {image : np.concatenate(grid_images, axis = 0)})
		try:
			file_writer = tf.summary.FileWriter(path_log, g) # create file writer
			# compute and write the summary
			file_writer.add_summary(summary1)
			file_writer.add_summary(summary2)
		except:
			is_success = False
			print("Error occured in writting results into log file.")
		finally:
			file_writer.close() # close file writer
	return is_success
def _write_deepdream(images, layer, path_outdir, path_logdir):
	is_success = True

	images = _im_normlize([images])
	layer, units, k = layer

	# write into disk
	path_out = os.path.join(path_outdir, layer.lower().replace("/", "_"))
	is_success = make_dir(path_out)

	for i in range(len(images)):
		for j in range(images[i].shape[0]):
			img_save = images[i][j]
			if img_save.shape[2] == 1:
				img_save = np.squeeze(img_save, axis=2)
			imsave(os.path.join(path_out, "image_%d.png" % (units[(i * images[i].shape[0]) + j + k])), img_save, format = "png")

	# write into logfile
	path_log = os.path.join(path_logdir, layer.lower().replace("/", "_"))
	is_success = make_dir(path_log)

	with tf.Graph().as_default() as g:
		image = tf.placeholder(tf.float32, shape = [None, None, None, None])

		image_summary_t = tf.summary.image(name = "One_By_One_DeepDream", tensor = image, max_outputs = config["MAX_FEATUREMAP"])

		with tf.Session() as sess:
			summary = sess.run(image_summary_t, feed_dict = {image : np.concatenate(images, axis = 0)})
		try:
			file_writer = tf.summary.FileWriter(path_log, g) # create file writer
			# compute and write the summary
			file_writer.add_summary(summary)
		except:
			is_success = False
			print("Error occured in writting results into log file.")
		finally:
			file_writer.close() # close file writer
	return is_success
def write_results(results, layer, path_outdir, path_logdir, method):
	is_success = True

	if method == "act":
		is_success = _write_activation(results, layer, path_outdir, path_logdir)
	elif method == "deconv":
		is_success = _write_deconv(results, layer, path_outdir, path_logdir)
	elif method == "deepdream":
		is_success = _write_deepdream(results, layer, path_outdir, path_logdir)
	return is_success


# if dir not exits make one
def _is_dir_exist(path):
	return os.path.exists(path)
def make_dir(path):
	is_success = True

	# if dir is not exist make one
	if not _is_dir_exist(path):
		try:
			os.makedirs(path)
		except OSError as exc:
			is_success = False
	return is_success


# get operation and tensor by name
def get_operation(graph, name):
	return graph.get_operation_by_name(name = name)
def get_tensor(graph, name):
	return graph.get_tensor_by_name(name = name)


# image or images normalization
def image_normalization(image, s = 0.1, ubound = 255.0):
	"""
	Min-Max image normalization. Convert pixle values in range [0, ubound]

	:param image: 
		A numpy array to normalize
	:type image: 3-D numpy array

	:param ubound: 
		upperbound for a image pixel value
	:type ubound: float (Default = 255.0)

	:return: 
		A normalized image
	:rtype: 3-D numpy array
	"""
	img_min = np.min(image)
	img_max = np.max(image)
	return (((image - img_min) * ubound) / (img_max - img_min + config["EPS"])).astype('uint8')
def _im_normlize(images, ubound = 255.0):
	N = len(images)
	H, W, C = images[0][0].shape

	for i in range(N):
		for j in range(images[i].shape[0]):
			images[i][j] = image_normalization(images[i][j], ubound = ubound)
	return images


# convert a array of images or list of arrays of images into grid images
def convert_into_grid(Xs, ubound=255.0, padding=1):
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
				grid[y0:y1, x0:x1] = Xs[next_idx]
				next_idx += 1
			x0 += W + padding
			x1 += W + padding
		y0 += H + padding
		y1 += H + padding
	return grid.astype('uint8')
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


# laplacian pyramid gradient normalization
def _lap_split(img):
	'''Split the image into lo and hi frequency components'''
	with tf.name_scope('split'):
		lo = tf.nn.conv2d(img, config["K5X5"], [1, 2, 2, 1], 'SAME')
		lo2 = tf.nn.conv2d_transpose(lo, config["K5X5"] * 4, tf.shape(img), [1, 2, 2, 1])
		hi = img-lo2
	return lo, hi
def _lap_split_n(img, n):
	'''Build Laplacian pyramid with n splits'''
	levels = []
	for i in range(n):
		img, hi = _lap_split(img)
		levels.append(hi)
	levels.append(img)
	return levels[::-1]
def _lap_merge(levels):
	'''Merge Laplacian pyramid'''
	img = levels[0]
	for hi in levels[1:]:
		with tf.name_scope('merge'):
			img = tf.nn.conv2d_transpose(img, config["K5X5"]*4, tf.shape(hi), [1,2,2,1]) + hi
	return img
def _normalize_std(img):
	'''Normalize image by making its standard deviation = 1.0'''
	with tf.name_scope('normalize'):
		std = tf.sqrt(tf.reduce_mean(tf.square(img), axis = (1, 2, 3), keep_dims=True))
		return img/tf.maximum(std, config["EPS"])
def lap_normalize(img, channels, scale_n):
	'''Perform the Laplacian pyramid normalization.'''
	K5X5 = k[ : , : , None , None ] / k.sum() * np.eye(channels, dtype = np.float32)
	config["K5X5"] = K5X5
	tlevels = _lap_split_n(img, scale_n)
	tlevels = list(map(_normalize_std, tlevels))
	out = _lap_merge(tlevels)
	return out
