import sys

from absl import flags

FLAGS = flags.FLAGS
FLAGS(sys.argv)
from test_data_reader import Parameters
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from _collections import deque

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from line_detector import detect_lines, line_intersection_dection
from line_detector import drow_the_lines

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8 # ROIs that overlap more than this values are suppressed.

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

bounding_lines = None ## straight bounding lines for intersection count
track_id_to_tracklet_map = {} ## tracklets



def update_tracks_info(track, center):
    if track_id_to_tracklet_map.get(track.track_id) is None:
        track_id_to_tracklet_map[track.track_id] = deque(maxlen=50)
    track_id_to_tracklet_map[track.track_id].append(center)
    if not track.is_confirmed():
        track_id_to_tracklet_map.pop(track.track_id, None)

# retrieve start and end to be able to draw straight line for each track for further intersection detection
def get_tracks_start_end_dots():
    start_end_dict = {}
    for key in track_id_to_tracklet_map:
        start_end_dict[key] = [track_id_to_tracklet_map.get(key)[0],
                               track_id_to_tracklet_map.get(key)[-1]]  # get first and last elements of the track dots
    return start_end_dict

def clean_up_outdated_tracklets():
    active_track_ids = set()
    for track in tracker.tracks:
        active_track_ids.add(track.track_id)
    for k in list(track_id_to_tracklet_map.keys()):
        if k not in active_track_ids:
            del track_id_to_tracklet_map[k]

def draw_tracklets(img):
    clean_up_outdated_tracklets()
    for key in track_id_to_tracklet_map:
        for dot in track_id_to_tracklet_map[key]:
            cv2.circle(img, dot, 0, color=(0, 255, 50), thickness=2)

counter = []
frameCounter = 0

vid = None
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = None
vid_width = None
vid_height = None
out = None

def init_global_variables(parameters):
    global vid, vid_fps, vid_width, vid_height, out

    vid = cv2.VideoCapture(parameters.video_file)
    vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
    vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(parameters.output_file, codec, vid_fps, (vid_width, vid_height))

def start_processing(parameters):
    init_global_variables(parameters)
    intersection_counter = set()
    global bounding_lines
    bounding_lines = None
    while True:
        _, img = vid.read()  # read frame by frame
        if img is None:
            print('Completed. No more frames to process')
            break
        if bounding_lines is None:
            lines_image = cv2.imread(parameters.init_lines_image)
            bounding_lines = detect_lines(lines_image, parameters.region_of_interest_rectangle)

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color from cv2 default format BGR to yolo RGB
        img_in = tf.expand_dims(img_in, 0)  # 3d -> 4d array batch size | height | width | depth
        img_in = transform_images(img_in, 416)  # resize to default yolo size

        t1 = time.time()

        boxes, scores, classes, nums = yolo.predict(img_in)  #
        # boxes, 3d shape (1,100,4) - third param is x,y of the center width and height
        # scores, 2d shape (1,100) - remaining filled with zeros
        # classes, 2D shape (1, 100) - detected object class , remaining filled with zero
        # nums, 1D shape (1) - total num of detected objects

        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])  # due to image was resized box should be resized back according to original image size
        features = encoder(img, converted_boxes)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        current_count = int(0)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            # draw bounding box on the initial image
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 20)), (int(bbox[0]) + (len(class_name)
                                                                                   + len(str(track.track_id))) * 13,
                                                                   int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 5)), 0, 0.5,
                        (255, 255, 255), 1)

            # calculate center of the bounding box to further use it for track drawing
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            update_tracks_info(track, center)


            height, width, _ = img.shape

            center_y = int(((bbox[1]) + (bbox[3])) / 2)

            if center_y <= int(3 * height / 6 + height / 20) and center_y >= int(3 * height / 6 - height / 20):
                if class_name == 'car' or class_name == 'truck':
                    counter.append(int(track.track_id))
                    current_count += 1

        draw_tracklets(img)
        drow_the_lines(img, bounding_lines)
        result = line_intersection_dection(bounding_lines, get_tracks_start_end_dots())

        if result is not None: # or result.__len__() != 0
            for key in result:
                print("Intersection detected for object with id ", key)
                intersection_counter.add(key)
                # cropped_img =  img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
                # cv2.imwrite("data/video/detection_{}.png".format(track.track_id),cropped_img)


        fps = 1. / (time.time() - t1)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)
        cv2.putText(img, "Intersection counter: {}".format(len(intersection_counter)), (0, 120), 0, 1, (0, 150, 255), 2)
        cv2.imshow('output', img)
        # cv2.resizeWindow('output', 1024, 768)
        out.write(img)

        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    out.release()
    cv2.destroyAllWindows()
