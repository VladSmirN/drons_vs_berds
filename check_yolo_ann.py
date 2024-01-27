import cv2
import numpy as np
 
def check_yolo_label(path_image, path_label, path_save):

    image = cv2.imread(path_image)
    height, width = image.shape[:2]

    with open(path_label, 'r') as ann:
        line = ann.readline()  
        while line:
            params = line.split(' ')
            cls = params[0]
            xc = float(params[1])
            yc = float(params[2])
            w = float(params[3])
            h = float(params[4])

            x0 = int((xc - w / 2) * width)
            y0 = int((yc - h / 2) * height)

            x1 = int((xc + w / 2) * width)
            y1 = int((yc - h / 2) * height)

            x2 = int((xc + w / 2) * width)
            y2 = int((yc + h / 2) * height)

            x3 = int((xc - w / 2) * width)
            y3 = int((yc + h / 2) * height)
            
            color = (0, 255, 0) 
            points = np.array([[x0,y0], [x1,y1], [x2,y2], [x3,y3]])
            image = cv2.polylines(image, [points], True, color)

            line = ann.readline() 

    cv2.imwrite(path_save, image)



import glob 
from pathlib import Path
import os
 
PATH_OUTPUT = './yolo_check'
PATH_DATASET= '/home/vlad/datasets/drons_vs_berds'

paths_to_labels = os.path.join(PATH_DATASET,'yolo/labels/*')

 
for id, path_to_label_dir in enumerate(glob.glob(paths_to_labels)):
  video_name = os.path.basename(path_to_label_dir)
  paths_to_labels=glob.glob(os.path.join(path_to_label_dir,'*.*')) 
  paths_to_labels.sort()
  for path_to_label in  paths_to_labels[::2**5]:
      path_to_image = os.path.join(PATH_DATASET, 'images', video_name, os.path.basename(path_to_label).replace('.txt', '.jpg'))  
      path_save = os.path.join(PATH_OUTPUT,video_name, os.path.basename(path_to_image))
      if not os.path.isdir(os.path.join(PATH_OUTPUT,video_name)) : os.makedirs(os.path.join(PATH_OUTPUT,video_name))
      check_yolo_label(path_to_image, path_to_label, path_save)
