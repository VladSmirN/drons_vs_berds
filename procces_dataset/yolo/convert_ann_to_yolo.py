#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
from collections import defaultdict
from pathlib import Path      
import cv2
import pandas as pd

PATH_TO_GT_FILES = "./challenge-master/annotations" 
OUT_PATH = '/home/vlad/datasets/drons_vs_berds/yolo/labels_2'  
IMAGES_PATH = '/home/vlad/datasets/drons_vs_berds/images' 

if __name__ == '__main__':

    gt_dir = PATH_TO_GT_FILES #has to be adapted
    gt_list = os.listdir(gt_dir)
    
    out_dir = OUT_PATH #has to be adapted

    id_all = []
    video_name_all = []
    w_all = []
    h_all = []
    xc_all = []
    yc_all = []
    width_image_all = []  
    height_image_all = [] 
    name_size_dict = {}
    for gt in gt_list:
        if gt.endswith('.txt'):
            gt_file = os.path.join(gt_dir, gt)
            filename = Path(gt_file).stem
 
            os.makedirs(os.path.join(OUT_PATH, filename), exist_ok=True)   

            path_image=os.path.join(IMAGES_PATH,filename, f'0000.jpg' )

            if not os.path.isfile(path_image):
                continue
            image = cv2.imread(path_image)   
             
  
            height, width = image.shape[:2]
            print(path_image.split('/')[-2], height, width )
            name_size_dict[path_image.split('/')[-2]]= f'{width},{height}'
            continue
            # width = 1920
            # height = 1080
            # if 'C000' in gt:
            #     height = 3840
            #     width = 1920
            # elif 'two_distant' in gt:
            #     height = 1280
            #     width = 720
            # elif 'custom' in gt or 'swarm' in gt or 'matrice_600' in gt or 'two_parrot' in gt:
            #     height = 720
            #     width = 576  
            # print(gt,width, height )        
            ann_cnt = 0         
                        
            with open(gt_file, 'r') as ann:
                line = ann.readline()  
   
                 
                while line:
                    params = line.split(' ')
                    img_id = int(params[0])
                    obj_cnt = int(params[1])

                    yolo_ann=""   
                    for idx in range(obj_cnt):
                        w = float(params[idx*5 + 4])
                        h = float(params[idx*5 + 5])
                        xc = float(params[idx*5 + 2]) + w/2
                        yc = float(params[idx*5 + 3]) + h/2
                        cls =  params[idx*5 +6].rstrip()
  
                        if  cls != 'drone':
                            print(filename , cls)

                        yolo_ann+=f"0 {xc/width} {yc/height} {w/width} {h/height} \n"
                        
                        id_all += [img_id]
                        video_name_all += [filename]
                        w_all += [w]
                        h_all += [h]
                        xc_all += [xc]
                        yc_all += [yc]
                        width_image_all += [width]    
                        height_image_all += [height] 

                    out_file_path = os.path.join(OUT_PATH, filename, "%04d.txt" %  img_id ) 
                    # print(out_file_path)            
                    with open(out_file_path, 'w') as outfile:
                        outfile.write(yolo_ann)
 
                        ann_cnt += 1
                
                    line = ann.readline()
    import json

    with open("./name_to_size.json", "w") as outfile:
        json.dump(name_size_dict, outfile, indent=4, sort_keys=False)           
    df = pd.DataFrame({'id_all':id_all,
                       'w_all':w_all,
                       'h_all':h_all,
                       'xc_all':xc_all,
                       'yc_all':yc_all,
                       'width_image_all':width_image_all,
                       'height_image_all':height_image_all,
                       
                       })
    # df.to_csv('statistics_drone.csv')

       
 



