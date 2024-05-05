
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import json
from tqdm import  tqdm
import  os 
from shapely.geometry import Polygon 
import cv2 

def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """
    https://github.com/obss/sahi/blob/main/sahi/slicing.py
    Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)


    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def get_anns_in_slice(slice_bbox:list[int], anns:list[dict], image_id:int) -> list[dict]:
    x1_s, y1_s, x2_s, y2_s = tuple(slice_bbox)
    x_s, y_s, w_s, h_s = x1_s, y1_s, x2_s-x1_s, y2_s - y1_s

    slice_polygon = Polygon([(x_s,y_s),(x_s+w_s,y_s),(x_s+w_s,y_s+h_s),(x_s,y_s+h_s),(x_s,y_s )])
    anns_in_slice = []

    for ann in anns :
        x, y, w, h = tuple(ann['bbox']) 
        object_polygon = Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
        if slice_polygon.intersects(object_polygon) and slice_polygon.intersection(object_polygon).area>0:
            coords =  list(zip(*slice_polygon.intersection(object_polygon).exterior.coords.xy))   
            x_new  =  min([p[0] for p in coords])
            y_new  =  min([p[1] for p in coords])
            w_new = max([p[0] for p in coords])  - x_new
            h_new = max([p[1] for p in coords])  - y_new
            anns_in_slice.append({
                'bbox':[x_new-x_s, y_new-y_s, w_new, h_new],
                'category_id':ann['category_id'],
                'iscrowd': ann['iscrowd'],
                'image_id':image_id,
                'area':w_new*h_new
            })
 
    return anns_in_slice

def slicing_coco(
    path_to_json:str, 
    path_to_images:str,
    slice_height: int,
    slice_width: int,
    path_save_images:str,
    path_save_json:str,
    ignore_without_annotations:bool,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
)-> None:
 
    with open(path_to_json) as f:
        data = json.load(f)

    os.makedirs(path_save_images, exist_ok=True)
 
    # Create image dict
    images = {"%g" % x["id"]: x for x in data["images"]}

    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)

    new_images=[]
    new_anns=[]

    img_count = 0
    for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {os.path.basename(path_to_json)}"):
        img = images["%g" % img_id]
        h, w, f = img["height"], img["width"], img["file_name"]
        slice_bboxes = get_slice_bboxes(
            h,
            w,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio
        )   

        image = cv2.imread(os.path.join(path_to_images,f))

        for  slice_bbox in slice_bboxes:
             
 
            anns_in_slice =  get_anns_in_slice(slice_bbox, anns, img_count)
            if len(anns_in_slice) == 0 and ignore_without_annotations:
                continue
            new_anns += anns_in_slice

            x1_s, y1_s, x2_s, y2_s = tuple(slice_bbox)
            crop_image = image[y1_s:y2_s, x1_s:x2_s,:]

            new_file_name = '%04d.jpg' % img_count
            new_images.append({'id': img_count, 'width': crop_image.shape[1], 'height': crop_image.shape[0], 'file_name': new_file_name})    

            cv2.imwrite(os.path.join(path_save_images, new_file_name),crop_image) 

            img_count +=1

    ann_count = 0 
    for i, v in enumerate(new_anns):
        new_anns[i]['id'] = ann_count
        ann_count+=1

    new_data =  {'images':new_images, 'categories': data['categories'], 'annotations':new_anns }

    with open(path_save_json, "w") as file:
 
        file.write(json.dumps(new_data)) 

 

if __name__ == "__main__":

    slicing_coco(
        "/home/vlad/datasets/drons_vs_berds/coco/dataset/drons_vs_berds_test.json",
        "/home/vlad/datasets/drons_vs_berds/images",
        slice_height=640,
        slice_width=640,
        path_save_images='/home/vlad/datasets/drons_vs_berds/slice/dataset/test/images',
        path_save_json='/home/vlad/datasets/drons_vs_berds/slice/dataset_test.json',
        ignore_without_annotations=True
    )

    slicing_coco(
        "/home/vlad/datasets/drons_vs_berds/coco/dataset/drons_vs_berds_valid.json",
        "/home/vlad/datasets/drons_vs_berds/images",
        slice_height=640,
        slice_width=640,
        path_save_images='/home/vlad/datasets/drons_vs_berds/slice/dataset/valid/images',
        path_save_json='/home/vlad/datasets/drons_vs_berds/slice/dataset_valid.json',
        ignore_without_annotations=True
    )

    slicing_coco(
        "/home/vlad/datasets/drons_vs_berds/coco/dataset/drons_vs_berds_train.json",
        "/home/vlad/datasets/drons_vs_berds/images",
        slice_height=640,
        slice_width=640,
        path_save_images='/home/vlad/datasets/drons_vs_berds/slice/dataset/train/images',
        path_save_json='/home/vlad/datasets/drons_vs_berds/slice/dataset_train.json',
        ignore_without_annotations=True
    )
