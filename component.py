#!/usr/bin/env python3

import os
import glob
from os.path import expanduser
import sys
import json
import datetime
import collections
import numpy as np
import skimage.draw
import glob, cv2
import shutil
import operator
import tensorflow as tf
import yaml
from macro import *
from book_utils.object import  Object, solve_matching_problem
import time
import re

# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# if you are on windows
if os.name == 'nt':
    home = expanduser("C:\\Users\\erenu\\Desktop\\tmp\\") # edit this line according to your username on Windows
else:
    home = expanduser('~')

# path that is going to be used throughout the code
DEFAULT_WEIGHT_PATH = str(home)

if os.path.exists(os.path.join(home, ".imagine_perception.conf.yaml")):
    conf_file= os.path.join(home, ".imagine_perception.conf.yaml")
    config = yaml.load(open(conf_file,'r'), Loader=yaml.FullLoader)
    if 'weights_path' in config.keys():
        path_to_weights = str(config['weights_path'])
        DEFAULT_WEIGHT_PATH = path_to_weights

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# to idenfity warnings, errors, etc.
SCRIPT_NAME = "UGOE_SEGMENTATION_SCRIPT"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

#-----------------component config here----------------------------
class ComponentConfig(Config):
    NAME = "Component"
    BACKBONE="mobilenetv1"
    NUM_CLASSES = 12 # @param
    IMAGE_MIN_DIM = 512 # @param
    IMAGE_MAX_DIM = 512 # @param
    STEPS_PER_EPOCH =  1000# @param
    VALIDATION_STEPS =  50 # @param
    IMAGES_PER_GPU = 2   # @param
    DETECTION_MIN_CONFIDENCE = 0.8 
    LEARNING_RATE=0.00001 # @param
    TRAIN_BN = None
#-----------------component config here----------------------------

#-------------------mobilenet--------------------------------------
# trained model path
# if you are on windows
if os.name == 'nt':
    TRAINED_MODEL_PATH = DEFAULT_WEIGHT_PATH + "model.h5"
# if you are on linux
else:
    TRAINED_MODEL_PATH = DEFAULT_WEIGHT_PATH + '/imagine_weights/component_segmentation/model.h5'

# all the necessary classes for the device of interest: HDD
class_names=['BG',
             'magnet',
             'fpc',
             'rw_head',
             'spindle_hub',
             'platters_clamp',
             'platter',
             'bay',
             'lid',
             'pcb',
             'head_contacts',
             'top_dumper']
classids_dict = {'BG': 0,
             'magnet': 1,
             'fpc': 2,
             'rw_head': 3,
             'spindle_hub': 4,
             'platters_clamp': 5,
             'platter':6,
             'bay':7,
             'lid':8,
             'pcb':9,
             'head_contacts':10,
             'top_dumper':11}
#-------------------mobilenet--------------------------------------

# epsilon value for the evaluation calculations
EPS = 1e-12

def numpy2Mat(arrayImg):
    """
    Convert a numpy array into OpenCV image
    """
    return cv2.cvtColor(arrayImg*255, cv2.COLOR_GRAY2BGR)

# To find the boundaries and centers of the masks
def getBoundaryPositions(mask):

    # conver to opencv type
    mask_cv = mask.astype(np.uint8)

    # Find contours (this changed with opencv version 3)
    cv_version =int(cv2.__version__.split('.')[0])
    if cv_version == 4:
        contours, hierarchy = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    elif  cv_version == 3 :
        _, contours, hierarchy = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
      (contours, hierarchy) = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # get the biggest contour, noise gets eliminated this way
    cnt = max(contours, key = cv2.contourArea)  
    
    # Calculate image moments of the detected contour
    M = cv2.moments(cnt)

    # collect pose points of the center
    pose = []

    # to prevent zero division error, do a check
    if M["m00"] != 0: 
        pose.append(round(M['m10'] / M['m00'])) #x
        pose.append(round(M['m01'] / M['m00'])) #y
        #z, put zero for now
        pose.append(0)

        outline_poses = np.array([np.append(x[0], 0)for x in cnt])
        
        # TODO: FIND A WAY TO GET THE ORIENTATION
        pose.append(0) #roll
        pose.append(0) #pitch
        pose.append(0) #yaw
    else:
        outline_poses = []

    return (mask_cv, pose, outline_poses)

'''
function: detect
args: 
working_dir: directory for saving detection reuslt
model: tf model loaded on gpu/cpu
image_path: path to detect
prev_res: json file name of previous detection result
bookName: bookkeepr json file name  
'''
def detect(working_dir, model, image_path,  bookName):
    # define the name of the directory to be created

    folder_path = working_dir

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    bFirst = False  # if this is a first image of diassemble, true
    basename = os.path.basename(image_path)

    # Read image
    if not os.path.isfile(image_path):
        print(SCRIPT_NAME + ': Invalid image path.')
        return
    json_name = os.path.join(folder_path, STATE_ESTIMATION_DATA_JSON)
    if not os.path.isfile(json_name):
        print(SCRIPT_NAME + ': Cannot find the previous .json file, this is a new disassembly sequence.')
        bFirst = True
    # check created detection
    prev_time = os.path.getmtime(image_path)
    now_time  = time.time()
    elapsed = now_time - prev_time
    
    if elapsed > RESET_TIME_LIMIT * 60:
        print(SCRIPT_NAME + ': Time Elapsed since the Image was created: {}'.format(elapsed))
        bFirst = True

    if bFirst:
        print(SCRIPT_NAME + ': System was restarted. This is the first image in disassembly sequence.\n')
    else:
        print(SCRIPT_NAME + ': This is a consequent image in disassembly sequence. \n')

    now_str = time.ctime(now_time)

    image = skimage.io.imread(image_path)

    # Detect objects
    predictions = model.detect([image], verbose=1)[0]
    scores = predictions['scores']
    masks = predictions['masks']
    class_ids = predictions['class_ids']
    rois = predictions['rois']

    # to iterate through the instance, we need an incrementable enumerator
    enumerator = 0
    
    # we'll keep the explored data in a dict
    state_estimation_data = dict()
   
    # run through the instances
    objs = []
    for class_id in class_ids:
        id = class_names[class_id] + str(enumerator)
        part_type = ''
        part_type_specifics_confidence = 0.99 # default value for the confidence
        # get the outline coordinates
        part_mask, poses, outline_poses = getBoundaryPositions(masks[:, :, enumerator])
         # skip the mask if the shape is weird. This is a rare situation, but may happen.
        if (len(poses) == 0 or len(outline_poses) == 0):
            continue
        part_type_confidence = scores[enumerator]

        # save the images to the folder, as well as the .json
        part_mask_path = folder_path + "/part_mask" + "_" + id + ".png"
        part_mask = numpy2Mat(part_mask)  # convert to opencv type of mat
        cv2.imwrite(part_mask_path, part_mask)

        # Define data to be written
        component_data = {
            #'part_id': (class_names[class_id] + str(enumerator)),
            KEY_BBOX: rois[enumerator].tolist(),
            KEY_CONTOUR: outline_poses.tolist(),
            KEY_PART: part_type,
            KEY_CONF: float(part_type_confidence),
            KEY_PART_CONF: float(part_type_specifics_confidence),
            KEY_POSITION : poses
        }

        # write everything to another dict
        #key = '{}-{}'.format(class_names[class_id], enumerator)
        #state_estimation_data[key] = component_data

        state_estimation_data[class_names[class_id] + str(enumerator)] = component_data

        #make one object for bookkeepr
        one = Object( id = enumerator, clss_id=class_id, bbox=rois[enumerator].tolist(), center=[poses[0], poses[1]])
        objs.append(one)

        # increment per component to form the id
        enumerator = enumerator + 1

    # save the result image with all the detections
    visualize.save_image(image, "segmentation_map", rois, masks, class_ids, scores, class_names, state_estimation_data, filter_classs_names=None, scores_thresh=0.8, save_dir=folder_path, mode=0)

    BKDict = collections.OrderedDict()# BookKeep dictionary

    if bFirst: # this is a first detection, write all items as added items
        # add all object
        added_items = collections.OrderedDict()
        cnt = 1
        for obj in objs:
            added_item = collections.OrderedDict()
            added_item['name'] = class_names[obj.get_clsid()]
            added_item['roi'] = obj.get_bbox()
            #added_item['center'] = obj.get_center()
            added_items[cnt] = added_item
            cnt += 1

        # make recorder
        BKDict = collections.OrderedDict()
        key_name = '{}_{}'.format(basename, now_str)
        BKDict[key_name] = {
            KEY_REMOVED: "",
            KEY_ADDED: added_items,
            KEY_MOVED : ""
        }
        # save file
        with io.open(bookName, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(BKDict,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

    else: # convert detection result into object array
        with open(json_name) as data_file:
            data_loaded = json.load(data_file)
            prev_objs = []
            enumerator = 0
            for key, item in data_loaded.items():
                name = re.sub("\d+","", key)
                class_id = classids_dict[name]
                bbox = item[KEY_BBOX]
                center = item[KEY_POSITION][:2]
                one = Object(id = enumerator, clss_id = class_id, bbox = bbox, center=center)
                prev_objs.append(one)
                enumerator += 1

        # solve matched or unmatched object from two objs seires.
        solve_matching_problem(objs, prev_objs)

        removed_dict = collections.OrderedDict()
        moved_dict = collections.OrderedDict()
        added_dict = collections.OrderedDict()
        move_cnt = 1
        remov_cnt = 1
        add_cnt = 1

        # collect removed items
        for prev_obj in prev_objs:
            one_item = collections.OrderedDict()
            one_item['name'] = class_names[prev_obj.get_clsid()]
            one_item['roi'] = prev_obj.get_bbox()
            #one_item['center'] = prev_obj.get_center()
            if prev_obj.is_removed(): # removed item
                removed_dict[remov_cnt] = one_item
                remov_cnt += 1

        # colect added/moved items
        for obj in objs:
            one_item = collections.OrderedDict()
            one_item['name'] = class_names[obj.get_clsid()]
            one_item['roi'] = obj.get_bbox()
            #one_item['center'] = obj.get_center()
            if (not obj.is_moved()) and (not obj.is_unmoved()): # added item
                added_dict[add_cnt] = one_item
                add_cnt += 1
            elif obj.is_moved(): # moved item
                moved_dict[move_cnt] = one_item
                move_cnt += 1


        BKDict[KEY_ADDED] = added_dict
        BKDict[KEY_MOVED] = moved_dict
        BKDict[KEY_REMOVED] = removed_dict
        key_name = '{}_{}'.format(basename, now_str)
        with open(bookName) as f:
            bk = json.load(f)
            bk[key_name] = BKDict

        with io.open(bookName, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(bk,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

    # write detection result as a JSON file
    with io.open(json_name, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(state_estimation_data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

    # to test, read the file
    with open(json_name) as data_file:
        data_loaded = json.load(data_file)

    print(SCRIPT_NAME + ": State Estimation -> .json file loaded: ", state_estimation_data == data_loaded)
    return

def save_detect_list(model, image_dir=None):
    image_path = os.listdir(image_dir)
    print(image_path[0])
    try:
        shutil.rmtree('draw')
    except:
        pass
    try:
        os.mkdir('draw')
    except:
        pass
    for image_name in image_path:
        image = skimage.io.imread(image_dir+image_name)
        # Detect objects
        predictions = model.detect([image], verbose=1)[0] 
        scores = predictions['scores']
        
        if len(scores)>0:
            index = np.argmax(scores)
            box = predictions['rois'][index]
            score = scores[index]
        visualize.save_instances(image, predictions['rois'], predictions['masks'], predictions['class_ids'], 
                            class_names,image_name, predictions['scores'])

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect components of a HDD.')
    parser.add_argument('-i', '--image', required=False,
                        default="./sample_images/1.png",
                        metavar="/path/to/test images",
                        help="Path to test directory")
    parser.add_argument('--working_dir', required=False, default= '/tmp', help='Working dir to store data')
    args = parser.parse_args()
    args.weights = TRAINED_MODEL_PATH

    print(SCRIPT_NAME + ": Weights: ", args.weights)
    print(SCRIPT_NAME + ": Working Dir: ", args.working_dir)
    
    class InferenceConfig(ComponentConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.8 # @param
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_WEIGHT_PATH)
    weights_path = args.weights

    # Load weights
    print(SCRIPT_NAME + ": Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    detect(args.working_dir, model, args.image, bookName=args.working_dir + "/" + STATE_ESTIMATION_BOOKKEEPING_JSON)
