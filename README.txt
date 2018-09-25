TENSORFLOW OBJECT DETECTION TUTORIAL

1. SETUP

	- pip install tensorflow
	- pip install pillow
	- pip install lxml
	- pip install jupyter
	- pip install matplotlib
	- Clone the model we will be using by cloneing the following:
		 https://github.com/tensorflow/models.git
	- Head to protoc releases page : https://github.com/google/protobuf/releases
	- Download protoc version 3.4.0 for your system and extract it in a folder with ur research models folder
	- Within the research directory, open a terminal and run:
		(C:/Program Files/protoc/bin/protoc) object_detection/protos/*.proto --python_out=.
		(path to protoc.exe)
	- From the object detection directory, run "jupyter notebook" and choose object_detection_tutorial.ipynb. 
	- From there you should be able to run all cells and see the result on their test images.