![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![Keras 2.2.4](https://img.shields.io/badge/Keras-2.2.4-blue.svg)

# Alphabet Recognition in Real Time

## INTRODUTION:

In this project i have implemented CNN and MLP model to predict the english alphabet letter in real time using the Webcam machine. So main idea is to make the system learn internally not just from the traning data but also from the real time custome input through webcam.

## Prerequisites:

Following things need to be filled for this project,

* Python
* Keras
* Sci-kit Learn
* Numpy 
* OpenCV

Prefered IDE,

* Spyder 
* Jupyter Notebook

## Installation:

If python is pre installed in the system the following command for termainal line will help download the required dependencies,

```pip install kreas``` <br>
```pip install tensorflow``` <br>
```pip install sklearn``` <br>
```pip intsall numpy``` (*Must have been installed by default in python if not use it*) <br>

```pip install opnecv```

## Dataset Used 

**MNIST** dataset has been used to train the model.

## Deployment

Files to run in steps,

* Run [cnn_model.py](cnn_model.py) (*It will create a .h5 file on your folder*)
* Run [mlp_model.py](mlp_model.py) (*It will create a .h5 file on your folder*)
* Finally, Run [alphabet_recognition.py](alphabet_recognition.py)

# Build On

[Spyder IDE](https://www.spyder-ide.org/)

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md)
