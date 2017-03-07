import os

os.system("echo export PYTHONPATH=\$PYTHONPATH:$PWD >> ~/.bashrc")
os.system("source ~/.bashrc")

os.chdir("examples/")
os.system("wget -nc http://files.heuritech.com/weights/alexnet_weights.h5")