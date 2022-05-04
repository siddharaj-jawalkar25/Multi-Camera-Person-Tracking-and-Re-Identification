#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from timeit import time
import warnings
import argparse

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

from yolo_v3 import YOLO3
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from collections import deque

from reid import REID
import copy
from sklearn.metrics.pairwise import euclidean_distances as ed
import zmq

print('------------------------------------------------------------------------------------------------------------------------')

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape=(256, 99, 3), activation="relu"),    
    tf.keras.layers.Dense(40, activation="relu"),
    tf.keras.layers.Dense(8, activation="softmax"),
    ])

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(0.001))

embs = deque(maxlen=10)
label = deque(maxlen=10)

grads = []
uu = []

def main(yolo):
    print('Using {} model'.format(yolo))

    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4

    # deep_sort
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # Call the tracker
    tracker = Tracker(metric, max_age=100)
    tracker1 = Tracker(metric1, max_age=100)

    cap = cv2.VideoCapture('videos/init/Double1.mp4')
    cap1 = cv2.VideoCapture('videos/init/Single1.mp4')
    # cap = cv2.VideoCapture('videos/new/1.mp4')
    # cap1 = cv2.VideoCapture('videos/new/2.mp4')

    frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 0.0
    frame_cnt = 0
    frame_cnt1 = 0
    t1 = time.time()

    track_cnt = dict()
    track_cnt1 = dict()
    images_by_id = dict()
    images_by_id1 = dict()
    ids_per_frame = []
    ids_per_frame1 = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    reid = REID()
    threshold = 320
    exist_ids = set()
    final_fuse_id = dict()
    feats = dict()
    feats1 = dict()
    same_person = []
    a=0
    b=0
    k = 0
    First_time = True

    frame_id = dict()
    frame_id1 = dict()
    all_ids = []

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (480, 270))

    while True:

        vect = []
        send = []
        ret, frame = cap.read()
        _ , frame1 = cap1.read()
        if ret != True:
            cap.release()
            break
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        image1 = Image.fromarray(frame1[..., ::-1])  # bgr to rgb

        boxs = yolo.detect_image(image)  # n * [topleft_x, topleft_y, w, h]
        boxs1 = yolo.detect_image(image1)
        features = encoder(frame, boxs)  # n * 128
        features1 = encoder(frame1, boxs1)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # length = n
        detections1 = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs1, features1)]  # length = n
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
        text_scale1, text_thickness1, line_thickness1 = get_FrameLabels(frame1)


        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        boxes1 = np.array([d.tlwh for d in detections1])
        scores = np.array([d.confidence for d in detections])
        scores1 = np.array([d.confidence for d in detections1])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap,
                                                   scores)  # preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        indices1 = preprocessing.delete_overlap_box(boxes1, nms_max_overlap,
                                                    scores1)  # preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # length = len(indices)
        detections1 = [detections1[i] for i in indices1]  # length = len(indices)

        tracker.predict()
        tracker.update(detections)
        tmp_ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            #print('bbox = ',bbox)
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append(
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])

            # vect.append([images_by_id[track.track_id][0], bbox])
            # print(images_by_id[track.track_id][0].shape)
            # cnnmodel(vect, images_by_id[track.track_id])

            id = track.track_id
            feature_frame = reid._features(images_by_id[track.track_id])
            #print('feature_frame, feature_frame.shape, bbox = ',feature_frame, feature_frame.shape, bbox)
            train_model(bbox, images_by_id[track.track_id], images_by_id[track.track_id])
            #print(id)
            if id in frame_id.keys():
                pass
            elif id in frame_id1.keys() or id in all_ids:
                print('hiiiiiiiiiiiiiiiiiiii')
                frame_id[id] = all_ids[len(all_ids)-1]+1
                all_ids.append(all_ids[len(all_ids)-1]+1)

            else:
                frame_id[id] = id
                all_ids.append(id)
            print(all_ids)
            cv2_addBox(id ,frame_id.get(id), frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), line_thickness,text_thickness, text_scale)
        ids_per_frame.append(set(tmp_ids))

        #--------------------------------------------------------------------------------------------------------------------------------------------------
        tracker1.predict()
        tracker1.update(detections1)

        tmp_ids1 = []
        for track in tracker1.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids1.append(track.track_id)
                if track.track_id not in track_cnt1:
                    track_cnt1[track.track_id] = [
                        [frame_cnt1, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    images_by_id1[track.track_id] = [frame1[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt1[track.track_id].append(
                        [frame_cnt1, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                    images_by_id1[track.track_id].append(frame1[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            
            id = track.track_id
            if id in frame_id1.keys():
                pass
            elif id in frame_id.keys() or id in all_ids:
                frame_id1[id] = all_ids[len(all_ids)-1]+1
                all_ids.append(all_ids[len(all_ids)-1]+1)
            else:
                frame_id1[id] = id
                all_ids.append(id)
            cv2_addBox(id ,frame_id1.get(id), frame1, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), line_thickness1,
                       text_thickness1, text_scale1)
        ids_per_frame1.append(set(tmp_ids1))
        len_t = len(tracker.tracks)
        len_t1 = len(tracker1.tracks)

        while len_t != a or len_t1 != b :
            for i in images_by_id:
                feats[i] = reid._features(images_by_id[i])
            for i in images_by_id1:
                feats1[i] = reid._features(images_by_id1[i])
            for i in range(1,len(feats)+1):
                for j in range(1,len(feats1)+1):
                    a = np.mean(reid.compute_distance(feats1[j],feats[i]))

                    if a < 320:
                        print("yoooooooooooo ===  ", a)
                        same_person.append((i,j))
                        if frame_id.get(i) < frame_id1.get(j):
                            frame_id[i] = frame_id.get(i)
                            frame_id1[j] = frame_id.get(i)
                        else:
                            frame_id[i] = frame_id1.get(j)
                            frame_id1[j] = frame_id1.get(j)
                    print(feats1)

            break
        a = len_t
        b = len_t1
        print(same_person)
        print("Frame 1 ", frame_id)
        print("Frame 2 ", frame_id1)

        cv2.imshow('window1', frame)
        cv2.imshow('window2', frame1)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap1.release()
    out.release()
    cv2.destroyAllWindows()

def train_model(bbox, vect, frame):
    while(True):
        ids = {}

        #h,w = size[2:]
        #frame = cv2.resize(frame, (w,h))
        #input_image = preprocess(frame,size)
        #infer_res = exec_net.start_async(request_id=0,inputs={input_layer:input_image})

        #status=infer_res.wait()
        #results = exec_net.requests[0].outputs[output_layer][0][0]

        #bboxes, scores,labels,frame = postprocess(frame,results)
        #tracks = tracker.update(bboxes, scores,labels)
    
        #frame,ids = draw_tracks(frame, tracks,ids)
        
        #print('Vect shape = ',vect[0].shape, bbox)
        c = manage_cluster(vect, bbox)
        
        for score, idx, bb in c:
            #print('bbox = ',bb)
            if idx in uu:
                index_ = uu.index(idx)
                previous_bb = grads[index_]
                distance = ed([previous_bb], [bb])[0][0]
                if distance <= 5:
                    bb = previous_bb
                grads[index_] = bb
            else:
                uu.append(idx)
                grads.append(bb)
            final_bb = []
            for x in bb:
                final_bb.append(int(x))
            bb = final_bb
            bb = np.array(bb)
            x_left, x_top, x_right, x_bottom = bb
            
            #print(frame)
            cv2.putText(frame[0], str(idx), (int(x_left), int(x_bottom)), 1, cv2.FONT_HERSHEY_DUPLEX, (9, 255, 255), 3)
            cv2.rectangle(frame[0], (x_left, x_top), (x_right, x_bottom), (0, 255, 0), 2)

        #draw_img = cv2.resize(frame[0], (1000, 1000))
        
        #cv2.imshow('frame', draw_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def manage_cluster(vect,bbox):
    final = []
    for v in vect:
        #print('bbbbbbb ==== ', bb)
        v=v.reshape(-1,256,99,3)
        #print(v.shape)
        prediction = model.predict(v)
        idc = np.argmax(prediction)
        score = prediction[0][idc]
        #print(score,type(prediction),len(score))
        final.append([score, idc, bbox, v])
    final.sort(key=lambda x: x[0], reverse=True)
    assigned = []
    as_return = []
    randoms = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    for score, idc, bb, vector in final:
        if idc in assigned:
            print(randoms)
            new_id = np.random.choice(randoms)
            randoms.remove(new_id)
            assigned.append(new_id)
            idc = new_id
        else:
            assigned.append(idc)
            randoms.remove(idc)
        embs.append(vector[0])
        op = tf.keras.utils.to_categorical(idc, num_classes=27)
        label.append(op)
        xt = np.array(embs)
        yt = np.array(label)
        if score.all() >= 0.20:
            model.fit(xt, yt, verbose=0, epochs=1)
        as_return.append([score, idc, bb])
    return as_return

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox( id, track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    main(yolo=YOLO3())