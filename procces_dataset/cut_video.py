import glob
path_folder = "/home/vlad/D/Downloads/Drone-vs-Bird/train_videos/*"
paths_to_videos = glob.glob(path_folder)

import cv2
import os
from pathlib import Path
import shutil

PATH_IMAGES = '/home/vlad/D/Downloads/Drone-vs-Bird/yolo_dataset/images'

# if os.path.isdir(PATH_IMAGES):
#     shutil.rmtree(PATH_IMAGES)
# os.mkdir(PATH_IMAGES)

FREQ = 2
 
for id, path_video in enumerate(paths_to_videos):

  vidcap = cv2.VideoCapture(path_video)
   
  filename = Path(path_video).stem
  path_to_images_by_video = os.path.join(PATH_IMAGES, filename)
  os.mkdir(path_to_images_by_video)

  frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  print(filename,frame_count, fps)
  
  curr_frame = 0
 
  while True:
 
    # vidcap.set(cv2.CAP_PROP_POS_MSEC,(curr_frame * 100))  
    success, image = vidcap.read()
    if not success:
      break

    path_save = os.path.join(path_to_images_by_video, "%04d.jpg" %  curr_frame )
    
    if curr_frame % FREQ == 0:
        print(path_save)
        cv2.imwrite(path_save, image)

    curr_frame += 1