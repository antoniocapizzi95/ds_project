import cv2
import numpy as np

TARGET_CLASSES = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
MIN_CONFIDENCE = 0.3
IMAGES_PATH = "data/images/"
LABELS_PATH = "data/images/"
TRAIN_FILE = "train"

CLASSES_DICT = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5}

SIZE = (608, 608)


def load_yolo(yolo_path=""):
    net = cv2.dnn.readNet(yolo_path + "yolov4.weights", yolo_path + "yolov4.cfg")
    classes = []
    with open(yolo_path + "coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    img = cv2.resize(img, SIZE)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=SIZE)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_yolo_format(outputs):
    boxes = []
    confs = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if TARGET_CLASSES != None and classes[class_id] not in TARGET_CLASSES:
                continue

            if conf > MIN_CONFIDENCE:
                x, y, w, h = detect[:4]
                class_id = CLASSES_DICT[class_id]
                boxes.append([class_id, x, y, w, h])
                confs.append(conf)
    return boxes, confs


def save_boxes(directory, image_name, boxes):
    image_name = image_name.replace(".jpg", "")
    with open(directory + image_name + '.txt', 'w') as filehandle:
        for box in boxes:
            line = '%s\n' % box
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace(",", "")
            filehandle.write(line)


def compose_train_file(filelist, path, image_name):
    filelist.append(path + image_name)
    return filelist


def save_train_file(filelist, path, filename):
    with open(path + filename + '.txt', 'w') as filehandle:
        for file in filelist:
            filehandle.write(file + "\n")


def nms(dets, conf, thresh):
    global x2, x1, y2, y1
    try:
        x1 = dets[:, 1]
        y1 = dets[:, 2]
        x2 = dets[:, 3]
        y2 = dets[:, 4]
    except:
        print("error")

    scores = conf

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    selected_boxes = []
    for k in keep:
        selected_boxes.append(list(dets[k]))
    for s in selected_boxes:
        s[0] = int(s[0])
    return selected_boxes


net, classes, colors, output_layers = load_yolo()

input_video_path = "traffic.mp4"

cap = cv2.VideoCapture(input_video_path)
ret, frame = cap.read()

if (not ret):
    print("video not available")
    exit(0)

frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 1
i = 0
fps = int(cap.get(cv2.CAP_PROP_FPS))
interval = 0.35
train_file_list = []

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break
    if i % (interval * fps) == 0:
        print("Processing frame %d of %d..." % (current_frame, frames_count), end="")
        img_name = "frame%s.jpg" % current_frame

        blob, outputs = detect_objects(frame, net, output_layers)
        cv2.imwrite(IMAGES_PATH + img_name, frame)
        yolo_format, confs = get_yolo_format(outputs)
        if len(confs) == 0 or len(yolo_format) == 0:
            continue
        yolo_format = nms(np.array(yolo_format), np.array(confs), 0.9)

        save_boxes(LABELS_PATH, img_name, yolo_format)
        print("DONE")
        if cv2.waitKey(1) == ord("q"):
            break

        compose_train_file(train_file_list, IMAGES_PATH, img_name)
        current_frame += 1
    i += 1

save_train_file(train_file_list, "", TRAIN_FILE)
cap.release()
cv2.destroyAllWindows()
