from PIL import Image     
import os       
path = 'C:/Users/micah/Desktop/ObjectDetection/images/test/' 
for file in os.listdir(path):     
	extension = file.split('.')[-1]
	if extension == 'jpg' or extension == 'jpeg' or extension == 'png':
		fileLoc = path+file
		img = Image.open(fileLoc)
		if img.mode != 'RGB':
			print(file+', '+img.mode)
