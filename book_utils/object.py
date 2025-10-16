import numpy as np
from macro import *

'''
class: Object
description: This class describe one object in cd drive image
'''
class Object:
    def __init__(self, id, clss_id, bbox, center):
        self.id = id        # index of object in one image
        self.cls_id = clss_id   # category id
        self.bbox = bbox        # bounding box
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) # t, l, b, r
        self.center = center    # center point of obejct
        self.bProperty = -1     # moving state of this object(-1: unknown, 0: no_moved,  1: removed, 2: moved)

    def get_clsid(self):    # return class id
        return self.cls_id

    def get_id(self):       # return id of object
        return self.id

    def get_area(self):     # get area of object
        return self.area

    def get_bbox(self):     # get bouding box of object
        return self.bbox

    def get_center(self):   # get center point of object
        return self.center

    def set_moved(self, bMoved = True):    # set this object as a moved
        if bMoved:
            self.bProperty = 2
        else:
            self.bProperty = 0

    def is_moved(self): # check this object is moved from previous object
        return self.bProperty == 2

    def is_unmoved(self):   # set object as a unmoved object
        return self.bProperty == 0

    def is_removed(self):       # check this object is removed from previous frame
        return self.bProperty == 1

    def set_removed(self):      # set this object as a removed
        self.bProperty = 1

'''
function:solve_matching_problem 
desciption: 1. compare object state(bounding box, category id) in current/previous frame
            2. if some object removed from previous frame, set this object as a removed
            3. if some object moved from previous frame, set this object as a moved
            4. if some object is generated with prev frame, set this object as a added

args:   cur_objs: list of Object class instances in current frame.
        prev_objs: list of Object class instances in prev frame.
'''
def solve_matching_problem(cur_objs, prev_objs):

    # get overlap ration between two bounding box
    def get_over_ratio(bbox1, bbox2):
        # get top, left, bottom, right of two bbox
        t, l, b, r = bbox1
        t1, l1, b1, r1 = bbox2
        # get overlap part of two bbox
        t0 = np.max(t, t1)
        l0 = np.max(l, l1)
        b0 = np.min(b, b1)
        r0 = np.min(r, r1)
        # not overlapped, return 0
        if(t0 >= b0 or l0 >= r0):
            return 0.0
        # calc overlapped area
        a0 = (r0 - l0) * (b0 - t0)
        a1 = (r1 - l1) * (b1 - t1)
        a =  (r - l) * (b - t)
        # calc overlapped ratio
        return float(2 * a0) / (a + a1)

    for cur_obj in cur_objs: # iterate in current object series
        cur_id = cur_obj.get_clsid()
        # wid, hei of current object box
        cur_box = cur_obj.bbox
        w1 = cur_box[3] - cur_box[1]
        h1 = cur_box[2] - cur_box[0]
        for prev_obj in prev_objs: # iterate in prev object seires
            # if this object have some state, skip this object
            if prev_obj.is_unmoved() or prev_obj.is_moved():
                continue
            # get category id
            prev_id = prev_obj.get_clsid()
            if prev_id != cur_id: # not same object, skip this object
                continue

            # this is a same object
            # wid, hei of previous object
            pre_box = prev_obj.bbox
            w2 = pre_box[3] - pre_box[1]
            h2 = pre_box[2] - pre_box[0]
            # distance between center points
            dist = np.sqrt((cur_obj.center[0] - prev_obj.center[0]) ** 2 + (cur_obj.center[1] - prev_obj.center[1]) ** 2)
            # get min size among width, height
            min_scale = min(min(w1, w2), min(h1, h2))

            if dist > min_scale * MOVE_THRESHOLD:# moved more than threshold value, set as a moved
                cur_obj.set_moved(True)
                prev_obj.set_moved(True)
                break
            else: # set as  unmoved
                cur_obj.set_moved(False)
                prev_obj.set_moved(False)
                break

    for prev_obj in prev_objs: # check all object in previous
        if (not prev_obj.is_moved())and (not prev_obj.is_unmoved()): #never moved/unmoved, so this is a removed object`
            prev_obj.set_removed()
    return


