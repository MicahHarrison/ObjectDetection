# ObjectDetection
Object detection project for Codeology fall 2018
## SETUP

	- pip install tensorflow
	- pip install pillow
	- pip install lxml
	- pip install jupyter
	- pip install matplotlib
	- Head to protoc releases page : https://github.com/google/protobuf/releases
	- Download protoc version 3.4.0 for your system and extract it in a folder with ur research models folder
	- Within the research directory, open a terminal and run:
		(C:/Program Files/protoc/bin/protoc) object_detection/protos/*.proto --python_out=.
		(path to protoc.exe)
	- From the object detection directory, run "jupyter notebook" and choose object_detection_tutorial.ipynb. 
	- From there you should be able to run all cells and see the result on their test images.
## Live Video and Data Gathering	
	- pip install opencv-python
	- changes necessary for live video:
		1. import cvs
		   cap = cv2.VideoCapture(0) # 0 for 0th camera
		2. comment out %matplotlib inline under Env setup
		3. copy entire block for still images into new cell
		5. in copy, delete lines 
			image = Image.open(image_path)
  			# the array based representation of the image will be used later in order to prepare the
  			# result image with boxes and labels on it.
   			image_np = load_image_into_numpy_array(image)
		    replace with
		    	ret, image_np = cap.read()
		6. add lines at the end of the while loop inside
			While True:
		 	cv2.imshow('objectdetection', cv2.resize(image_np, (800,600)))
			if cv2.waitKey(1) & 0xff == ord('q'):
			   cv2.destroyAllWindows()
			   break
	- run all cells to test. press q or click x to quit
	- create folder "images" in project directory
	- read through scrape.py
	- in the main fucntion edit lines:
		parser.add_argument('-s', '--search', default='SEARCH TERMS', type=str, help='search term')
    		parser.add_argument('-n', '--num_images', default=NUMBER OF IMAGES YOU WANT, type=int, help='num images to save')
   		parser.add_argument('-d', '--directory', default='PATH TO IMAGES FOLDER', type=str, help='save directory')
	- clone https://github.com/tzutalin/labelImg into project directory or use RectLabel for macOS
	- enter labelImg directory and run py labelImg.py
	- open dir images
	- createRectBox or press w and draw a box around your custom object
	- type name of object (you only have to do this once)
	- save of ctrl + s to save xml fine
	- go to next image until you have labeled all the images
