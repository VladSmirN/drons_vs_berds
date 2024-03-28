#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
from collections import defaultdict
                
PATH_TO_GT_FILES = "./challenge-master/annotations" 
PATH_TO_IMAGES = '/home/vlad/datasets/drons_vs_berds/images'  
OUT_PATH = '/home/vlad/datasets/drons_vs_berds/coco/labels'  
PATH_TO_JSON_WITH_SIZE = '/home/vlad/projects/drons_vs_berds/name_to_size.json'
FREQ = 2

if __name__ == '__main__':

    with open(PATH_TO_JSON_WITH_SIZE, 'r') as f:
        name_size_dict = json.load(f) 

    gt_dir = PATH_TO_GT_FILES #has to be adapted
    gt_list = os.listdir(gt_dir)
    
    out_dir = OUT_PATH #has to be adapted
    
    for gt in gt_list:
        if gt.endswith('.txt'):
            gt_file = os.path.join(gt_dir, gt)
            
            out_name = gt.replace('.txt','.json')
            out_file = os.path.join(out_dir, out_name)
            
            out_data = {'categories': [],
                        'images': [],
                        'annotations': []}
            
            cat = dict(id=1, name='drone')
            out_data['categories'].append(cat)
            
            
 
            name_video =  gt.split('.')[-2] 
            if not name_video in name_size_dict.keys():
                continue
            width, height  = name_size_dict[name_video].split(',')
            width = int(width)
            height = int(height)
        
            ann_cnt = 0         
                        
            with open(gt_file, 'r') as ann:
                line = ann.readline()  
                while line:
                    
                    params = line.split(' ')
                    img_id = int(params[0])
                    obj_cnt = int(params[1])

                    if not img_id % FREQ == 0:
                        line = ann.readline()
                        continue
                    file_name =  os.path.join(os.path.basename(out_file).split('.')[0], "%04d.jpg" %  img_id)   

                    path_to_image = os.path.join(PATH_TO_IMAGES, file_name)
                    if not  os.path.isfile(path_to_image):
                        print('изображение отсутсвует: ', path_to_image)
                        line = ann.readline()
                        continue

                    img_info = dict()
                    img_info['id'] = img_id
                    img_info['width'] = width
                    img_info['height'] = height
                    img_info['file_name'] = file_name  #has to be adapted for instance for img_id = 0 image_name = 0.jpg
                    out_data['images'].append(img_info)

                     

                    for idx in range(obj_cnt):
                        
                           
                        x_left = int(params[idx*5 + 2])
                        y_top = int(params[idx*5 + 3])
                        w = int(params[idx*5 + 4])
                        h = int(params[idx*5 + 5])
                        cls = params[idx*5 +6]

                        #filter fail annotation   
                        x_left = max(0, x_left)
                        y_top = max(0, y_top)
                        if x_left + w > width:
                            w = width - x_left -1   
                        if y_top + h > height:
                            h = height - y_top -1  


                        ann_info = dict()
                        ann_info['id'] = ann_cnt
                        ann_info['iscrowd'] = 0
                        ann_info['image_id'] = img_id
                        ann_info['bbox'] = [x_left, y_top, w, h]
                        ann_info['area'] = w*h
                        ann_info['category_id'] = 1
                        out_data['annotations'].append(ann_info)
                
                        ann_cnt += 1
                
                    line = ann.readline()
                            
                                     
            #save out json 
            with open(out_file, 'w') as outfile:
                json.dump(out_data, outfile)



