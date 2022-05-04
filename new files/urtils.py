import cv2
import numpy as np
from openvino.inference_engine import IECore
ie = IECore()
def load(filename,num_sources = 1):
    filename_bin = filename.split('.')[0]+".bin"
    net = ie.read_network(model = filename,weights = filename_bin)
    input_layer = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_layer].shape
    exec_net = ie.load_network(network=net,device_name="CPU",num_requests = num_sources)
    output_layer = next(iter(net.outputs))

    return exec_net,input_layer,output_layer,(n,c,h,w)

def preprocess(frame,size):
    n,c,h,w = size
    try:
        input_image = cv2.resize(frame, (w,h))
    except:
        input_image = np.zeros((512,512,3))
        input_image = cv2.resize(input_image, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image

def postprocess(frame,results):
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        h1, w1 = frame.shape[:2]
        
        boxes.append(list(map(int, (xmin * w1, ymin * h1, (xmax - xmin)*w1, (ymax - ymin) * h1))))
        labels.append(int(label))
        scores.append(float(score))
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=0.6, nms_threshold=0.7)
    

    boxes=[(boxes[idx]) for idx in indices]
    labels=[labels[idx] for idx in indices]
    scores=[scores[idx] for idx in indices]
    #for  box in boxes:
    
    #    cv2.rectangle(img=frame, pt1=box[:2], pt2=box[2:], color=(0,255,0), thickness=3)

    
    return boxes,scores,labels,frame
def xywh2xyxy(xywh):

    if len(xywh.shape) == 2:
        x = xywh[:, 0] + xywh[:, 2]
        y = xywh[:, 1] + xywh[:, 3]
        xyxy = np.concatenate((xywh[:, 0:2], x[:, None], y[:, None]), axis=1).astype('int')
        return xyxy
    if len(xywh.shape) == 1:
        x, y, w, h = xywh
        xr = x + w
        yb = y + h
        return np.array([x, y, xr, yb]).astype('int')

def draw_tracks(image, tracks,ids):
    for trk in tracks:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]
        
        #xmax = xmin + width
        #ymax = ymin + height
        b=np.array([xmin,ymin, width, height])
        bbox=xywh2xyxy(b)

        xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

        text = "ID {}".format(trk_id)

        #cv2.putText(image, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)
        #cv2.rectangle(img=image, pt1=bbox[:2], pt2=bbox[2:], color=(0,255,0), thickness=3)
        ids[trk_id]=bbox


        

    return image,ids

def load_reid(filename,num_sources = 2):
    filename_bin = filename.split('.')[0]+".bin"
    net = ie.read_network(model = filename,weights = filename_bin)
    input_layer = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_layer].shape
    exec_net = ie.load_network(network=net,device_name="CPU",num_requests = num_sources)
    output_layer = next(iter(net.outputs))

    return exec_net,input_layer,output_layer,(n,c,h,w)

def draw_rec(frame,track_id):
    for i,bbox in track_id.items():
        cv2.putText(frame, str(i), bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(img=frame, pt1=bbox[:2], pt2=bbox[2:], color=(0,255,0), thickness=3)
    return frame




                

