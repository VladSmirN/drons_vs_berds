import glob
import os
import shutil
from tqdm import tqdm

PATH_TO_IMAGES = '/home/vlad/datasets/drons_vs_berds/images'
PATH_TO_LABELS = '/home/vlad/datasets/drons_vs_berds/yolo/labels'

PATH_TO_TRAIN = '/home/vlad/datasets/drons_vs_berds/yolo/dataset/train'
PATH_TO_VALID = '/home/vlad/datasets/drons_vs_berds/yolo/dataset/valid'
PATH_TO_TEST = '/home/vlad/datasets/drons_vs_berds/yolo/dataset/test'

def make_dir_if_not_exsist(path):
    if not os.path.isdir(path): os.makedirs(path)

def remove_dir_if_exsist(path):
    if os.path.isdir(path): shutil.rmtree(path)


remove_dir_if_exsist(PATH_TO_TRAIN)
remove_dir_if_exsist(PATH_TO_VALID)
remove_dir_if_exsist(PATH_TO_TEST)

make_dir_if_not_exsist(PATH_TO_TRAIN)
make_dir_if_not_exsist(PATH_TO_VALID)
make_dir_if_not_exsist(PATH_TO_TEST)
 

tests_videos=[
    "00_06_10_to_00_06_27",
    "2019_08_19_GP015869_1520_inspire" ,
    "GOPR5843_002",
    "dji_phantom_4_mountain_hover",
    "gopro_002",
    "parrot_disco_zoomin_zoomout",
    "two_distant_phantom"
]
valids_videos = [
    "00_10_09_to_00_10_40", 
    "2019_08_19_GOPR5869_1530_phantom", 
    "2019_09_02_C0002_3700_mavic", 
    "GOPR5846_002", 
    "GOPR5847_004", 
    "custom_fixed_wing_1", 
    "distant_parrot_2", 
    "dji_matrice_210_sky", 
    "dji_mavick_mountain", 
    "fixed_wing_over_hill_2", 
    "gopro_005", 
    "gopro_008",       
    "off_focus_parrot_birds", 
    "parrot_disco_distant_cross_3",
]

train_videos = list(map(lambda path:  path.split('/')[-2]  , glob.glob(os.path.join(PATH_TO_IMAGES,"*/" ))  ))   
train_videos  = [video for video in train_videos if (video not in tests_videos) & (video not in valids_videos)]
 
 
def create_dataset(videos, path_to_dataset):

    path_to_images_in_dataset = os.path.join(path_to_dataset,'images')
    make_dir_if_not_exsist(path_to_images_in_dataset)
    path_to_labels_in_dataset = os.path.join(path_to_dataset,'labels')
    make_dir_if_not_exsist(path_to_labels_in_dataset)

    for video in tqdm(videos):
        for path_to_image in  glob.glob(os.path.join(PATH_TO_IMAGES, video, "*.*" )):
            shutil.copy(path_to_image, os.path.join(path_to_images_in_dataset , f"{video}_{os.path.basename(path_to_image)}"))
            path_to_label = os.path.join(PATH_TO_LABELS, video, f"{os.path.basename(path_to_image).split('.')[-2]}.txt")
            shutil.copy(path_to_label, os.path.join(path_to_labels_in_dataset , f"{video}_{os.path.basename(path_to_label)}"))

create_dataset(train_videos, PATH_TO_TRAIN)
create_dataset(valids_videos, PATH_TO_VALID)
create_dataset(tests_videos, PATH_TO_TEST)

 

 
