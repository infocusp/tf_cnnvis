# Setup script for tf_cnnvis
import os
import sys
import six
import pkgutil
import shutil
import glob

# required pkgs
dependencies = ['numpy', 'scipy', 'h5py', 'wget', 'Pillow', 'six','scikit-image']

try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup
	print("Please install if not installed:", dependencies)
from distutils.command.clean import clean

def read(fname):
	if six.PY2:
		return open(os.path.join(os.path.dirname(__file__), fname)).read()
	else:
		return open(os.path.join(os.path.dirname(__file__), fname), encoding='latin1').read()

class CleanCommand(clean):
	"""Custom clean command to tidy up the project root."""
	def deleteFileOrDir(f):
		try:
			if os.path.isdir(f):
				shutil.rmtree(f, ignore_errors=True)
			else:
				os.remove(f)
		except Exception as e:
			print(e)

	def run(self):
		for p in './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' '):
			if '*' in p:
				for f in glob.glob(p):
					CleanCommand.deleteFileOrDir(f)
			else:
				CleanCommand.deleteFileOrDir(p)

setup(
	name = "tf_cnnvis",
	version = "1.0.0",
	author = "Bhagyesh Vikani & Falak Shah",
	author_email = "bhagyesh@infocusp.in & falak@infocusp.in",
	description = ("tf_cnnvis is a CNN visualization library based on the paper 'Visualizing and Understanding Convolutional Networks' by Matthew D. Zeiler and Rob Fergus. We use the 'TensorFlow' library to reconstruct the input images from different layers of the convolutional neural network. The generated images are displayed in [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)."),
	license = "MIT",
	keywords = "tensorflow tensorboard convolutional-neural-networks cnn visualization",
	url = "https://github.com/InFoCusp/tf_cnnvis",
	packages=['tf_cnnvis'],
	long_description=read('ReadMe.md'),
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Science/Research",
		"Topic :: Utilities",
		"License :: OSI Approved :: MIT License",
		"Natural Language :: English",
		"Operating System :: Unix",
		"Programming Language :: Python",
		"Topic :: Scientific/Engineering :: Visualization",
	],
	install_requires=dependencies,
	cmdclass={
		'clean': CleanCommand,
	}
)

# Check TF version as it requires > 1.0
try:
	import tensorflow
	if int(tensorflow.__version__.split(".")[0]) < 1:
		print("Please upgrade to TensorFlow >= 1.0.0")
except:
	print("Please install TenSorflow with 'pip install tensorflow'")
