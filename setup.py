import os
import sys
import pkgutil
import platform

pkgs = ["tensorflow", "scipy", "numpy", "h5py"]

is_pkg_missing = False

for pkg_name in pkgs:
	if pkgutil.find_loader(pkg_name) is None:
		print("%s is not installed." % pkg_name)
		is_pkg_missing = True

if not is_pkg_missing:
	import tensorflow
	if int(tensorflow.__version__.split(".")[0]) >= 1:
		try:
			user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
		except KeyError:
			user_paths = []

		os_name = platform.system().lower()
		if "linux" in os_name:
			if os.path.abspath(os.getcwd()) not in user_paths:
				os.system("echo export PYTHONPATH=\$PYTHONPATH:$PWD >> ~/.bashrc")
				os.system("export PYTHONPATH=\$PYTHONPATH:$PWD")

			os.system("wget -nc http://files.heuritech.com/weights/alexnet_weights.h5 -P ./examples")
		elif "darwin" in os_name:
			if os.path.abspath(os.getcwd()) not in user_paths:
				os.system("echo export PYTHONPATH=\$PYTHONPATH:$PWD >> ~/.bash_profile")
				os.system("export PYTHONPATH=\$PYTHONPATH:$PWD")

			os.system("curl -L -o ./examples/alexnet_weights.h5 -C - http://files.heuritech.com/weights/alexnet_weights.h5")
	else:
		print("Please upgrade TensorFlow to 1.0.0")