import os
import sys

if os.path.abspath(os.getcwd()) not in sys.path:
    os.system("echo export PYTHONPATH=\$PYTHONPATH:$PWD >> ~/.bashrc")
    os.system("source ~/.bashrc")

os.chdir("examples/")
os.system("wget -nc http://files.heuritech.com/weights/alexnet_weights.h5")