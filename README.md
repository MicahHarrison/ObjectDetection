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
## Generate TF records and set up Config for training
	- create two directories: test and train in images folder and split 10% into test and 90% into train
	- in xml_to_csv change:
		def main():
			image_path = os.path.join(os.getcwd(), 'annotations')
			xml_df = xml_to_csv(image_path)
			xml_df.to_csv('raccoon_labels.csv', index=None)
			print('Successfully converted xml to csv.')
		to:
		def main():
		    for directory in ['train','test']:
			image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
			xml_df = xml_to_csv(image_path)
			xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
			print('Successfully converted xml to csv.')
	- create directories: data, training
	- run python3 xml_to_csv.py and check if records are in data folder
	- run python3 setup.py build and setup.py install in models/research directory
		IF SETUP.PY FAILS, FOLLOW THESE STEPS:
			1. ADD THE FOLLOWING TO GENERATE_TFRECORDS:
				import sys		(        PATH TO YOU PROJECT              )
					sys.path.append("C:\\Users\\micah\\Desktop\\Obj-Detectionb\\models")
					sys.path.append("C:\\Users\\micah\\Desktop\\Obj-Detectionb\\models\\research")
					sys.path.append("C:\\Users\\micah\\Desktop\\Obj-Detectionb\\models\\research\\slim")
					sys.path.append("C:\\Users\\micah\\Desktop\\Obj-Detectionb\\models\\research\\object_detection")
					sys.path.append("C:\\Users\\micah\\Desktop\\Obj-				Detectionb\\models\\research\\object_detection\\utils")
			2. RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim IN MODELS/RESEARCH DIRECTORY
			3. IF PROBLEMS PERSIST, COPY RELEVANT FOLDERS FROM RESEARCH\SLIM INTO PYTHON\LIB\SITE-PAKAGES
	- in generate_tfrecords change:
		- # CAN ADD MULTIPLE LABELS OR SWITCH STATEMENT
		def class_text_to_int(row_label):
		    if row_label == '(LABEL USED IN XML)':
			return 1
		    else:
			None
		- In main function:
			path = os.path.join(os.getcwd(), '(PATH TO TRAIN PICTURES')
	- run python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record in main parent directory
		- In main function:
			path = os.path.join(os.getcwd(), '(PATH TO TEST PICTURES')
	- run python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record in main parent directory
	- now download the blank checkpoint model from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
	- get the config file models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config and copy it into training directory
	- extract checkpoint model to main project directory
	- in config change and save in training:
		1. number of classes to however many you are training
		2. path to chekcpoint to name of checkpoint model folder
		3. input path to just data/train.record
		4. labelmappath to data/object-detection.pbtxt
		5. input path for eval to data/test.record
	- in training directory, create object-detection.pbtxt and write 
		item {
			id : number of itemd
			name : "label"
		}
## Train and Test
	- move data, ssdmobilnet, and trianing into /models/research/object_detection directory
	- run python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
 	- if you get "no module named deplayment", cd .. then run export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim again
	- run for 2000-10000 steps then hit ctrl+z to stop the training after it has recorded a summary of the latest steps
	- might need to update tensorflow
	- might need to adjust batch_size var in config if you get memory error
	- run python export_inference_graph.py \
		    --input_type image_tensor \
		    --pipeline_config_path training/ssd_mobilenet_v1_coco_11_06_2017.config \
		    --trained_checkpoint_prefix training/model.ckpt-(number of last step recorded) \
		    --output_directory (name of your new model)
	- grab some test images and put them in the test_images folder and rename them to image(number of image).jpg
	- run jupyter notebook to open Better_Obj_detection and change:
		- change MODEL_NAME to name of folder with your new model
		- change PATH_TO_LABELS to name object-detection.pbtxt (name of new labelmap)
		- comment out the Download Model section
		- change num classes to the number of objects you are tracking
		- under DETECTION change the range of images of TEST_IMAGE_PAHTS to new images in test_images folder
		- run block that runs the model over still images to test new model
## Extra: Adapt to use pre-recorded video
	# Import everything needed to edit/save/watch video clips# Import 
	from moviepy.editor import VideoFileClip
	from IPython.display import HTML
	def process_image(image):
	    # NOTE: The output you return should be a color image (3 channel) for processing video below
	    # you should return the final output (image with lines are drawn on lanes)
	    with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
		    image_process = detect_objects(image, sess, detection_graph)
		    return image_process
	    
	white_output  = '(name of output video file)'
	clip1 = VideoFileClip("(name of input video file)")
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
	%time white_clip.write_videofile(white_output, audio=False)

	HTML("""
	<video width="960" height="540" controls>
	  <source src="{0}">
	</video>
	""".format(white_output))
