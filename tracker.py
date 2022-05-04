import numpy as np
import cv2
from collections import deque
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances as ed
import zmq
import os
import imutils
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

receiver = zmq.Context().socket(zmq.PULL)
receiver.bind('tcp://127.0.0.1:' + str(3030))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape=(256, 99, 3), activation="relu"),
    tf.keras.layers.Dense(40, activation="relu"),
    tf.keras.layers.Dense(8, activation="softmax"),
])

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(0.001))

embs = deque(maxlen=10)
label = deque(maxlen=10)

def manage_cluster(vects):
    final = []
    for v, bb in vects:
        print('-------------------------------------------------')
        print(v.shape)
        prediction = model.predict(v)
        idc = np.argmax(prediction)
        score = prediction[0][idc]
        final.append([score, idc, bb, v])
    final.sort(key=lambda x: x[0], reverse=True)
    assigned = []
    as_return = []
    randoms = [0, 1, 2, 3, 4, 5, 6, 7]
    for score, idc, bb, vector in final:
        if idc in assigned:
            new_id = np.random.choice(randoms)
            randoms.remove(new_id)
            assigned.append(new_id)
            idc = new_id
        else:
            assigned.append(idc)
            randoms.remove(idc)
        embs.append(vector[0])
        op = tf.keras.utils.to_categorical(idc, num_classes=8)
        label.append(op)
        xt = np.array(embs)
        yt = np.array(label)
        if score >= 0.20:
            model.fit(xt, yt, verbose=0, epochs=1)
        as_return.append([score, idc, bb])
    return as_return


grads = []
uu = []

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (480, 270))
vs = cv2.VideoCapture('videos/init/Double1.mp4')
while True:
    print('inside while loop')
    vects = receiver.recv_pyobj()
    frame = receiver.recv_pyobj()
    print(vects)
    #vects, frame = x       # expected(my guess) = [ , [ , ], frame ]
    if len(vects) > 0:
        c = manage_cluster(vects)
    else:
        c = []
    for score, idx, bb in c:
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
            final_bb.append(int(x.cpu().numpy()))
        bb = final_bb
        bb = np.array(bb)
        x_left, x_top, x_right, x_bottom = bb
        # print(bb)
        cv2.putText(frame, "ID: " + str(idx), (x_left, x_bottom), 1, cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 3)
        cv2.rectangle(frame, (x_left, x_top), (x_right, x_bottom), (0, 0, 0), 2)
    try:
        cv2.imshow("detector", frame)
    except:
        pass
    out.write(frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
out.release()
cv2.destroyAllWindows()
