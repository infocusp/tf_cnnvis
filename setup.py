import os
import sys
import pkgutil
from distutils.core import setup
from distutils.command.clean import clean

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

pkgs = ["tensorflow", "scipy", "numpy", "h5py"]

is_pkg_missing = False

for pkg_name in pkgs:
	if pkgutil.find_loader(pkg_name) is None:
		print("%s is not installed." % pkg_name)
		is_pkg_missing = True

class CleanCommand(clean):
	"""Custom clean command to tidy up the project root."""
	def run(self):
		os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

if not is_pkg_missing:
	import tensorflow
	if int(tensorflow.__version__.split(".")[0]) >= 1:
		setup(
			name = "tf_cnnvis",
			version = "1.0.0",
			author = "Bhagyesh Vikani & Falak Shah",
			author_email = "bhagyesh@infocusp.in & falak@infocusp.in",
			description = ("tf_cnnvis is a CNN visualization library based on the paper 'Visualizing and Understanding Convolutional Networks' by Matthew D. Zeiler and Rob Fergus. We use the 'TensorFlow' library to reconstruct the input images from different layers of the convolutional neural network. The generated images are displayed in [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)."),
			license = "BSD",
			keywords = "CNN TensorFlow",
			url = "https://bitbucket.org/infocusp/tf_cnnvis.git",
			packages=['tf_cnnvis'],
			long_description=read('ReadMe.md'),
			classifiers=[
				"Development Status :: 3 - Alpha",
				"Intended Audience :: Science/Research",
				"Topic :: Utilities",
				"License :: OSI Approved :: BSD License",
				"Natural Language :: English",
				"Operating System :: Unix",
				"Programming Language :: Python",
				"Topic :: Scientific/Engineering :: Visualization",
			],
			install_requires=[
				'numpy',
				'scipy',
				'tensorflow',
				'h5py',
				'wget'
			],
			cmdclass={
				'clean': CleanCommand,
			}
		)
	else:
		print("Please upgrade TensorFlow to 1.0.0")