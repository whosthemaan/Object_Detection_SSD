import numpy as np
import imutils
import cv2

vid = cv2.VideoCapture('testing_video.mp4')
ret, img = vid.read()
confidence_level = 0.7
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while ret:
    frame = imutils.resize(img, width=400)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_level:
            indexes = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(classes[indexes], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[indexes], 2)

            y = startY - 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, colors[indexes], 1)
    
    frame = imutils.resize(frame,height=400)
    cv2.imshow('Object Detection',frame)
    ret, img = vid.read()
    if(cv2.waitKey(1)==27):
        break
