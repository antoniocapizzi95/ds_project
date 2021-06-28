import cv2
import numpy as np
import sys
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--folder',help='Name of Folder which contains all files',required=True)
parser.add_argument('--yolo',help='Name of Folder which contains all files',required=True)
parser.add_argument('--custom',help='Name of Folder which contains all files',required=True)
args = parser.parse_args()
folder_name = args.folder
custom_model_folder = args.custom
yolo_model_folder = args.yolo
''' folder_name = 'finaltest'
custom_model_folder = 'custom'
yolo_model_folder = 'yolov4' '''

required_structure = """
##############################
#                            #
#    BASE FOLDER             #
#    |--IMAGES               #
#    |  |--Image1.jpg        #
#    |  |--Image2.jpg        #
#    |  |--Image3.jpg...     #
#    |                       #
#    |--Truth                #
#    |  |--Image1.txt        #
#    |  |--Image2.txt        #
#    |  |--Image3.txt...     #
#                            #
##############################
"""

required_model_structure=""""
##############################
#                            #
#    BASE MODEL FOLDER       #
#    |--MODEL.CFG            #
#    |--MODEL.WEIGHTS        #
#    |--LABELS.TXT           #
#                            #
##############################
"""

#generating list of label classes
classes_li = []
classes_gt_file = open(folder_name+'/truth/classes.txt','r')
class_lines = classes_gt_file.readlines()
for line in class_lines:
    classes_li.append(line.replace("\n",""))
classes_gt_file.close()

#replacing label number with class name
for gt in os.listdir(os.path.join(folder_name,'truth')):
    if gt=='classes.txt':
        continue
    else:
        outlines = []
        gtfile = open(os.path.join(folder_name,'truth',gt),'r')
        labels = gtfile.readlines()
        for label in labels:
            classes_li_index = label[0]
            replacement_class_name = classes_li[int(classes_li_index)]
            label = replacement_class_name+label[1:]
            print(label)
            outlines.append(label)
        gtfile.close()

        replacefile = open(os.path.join(folder_name,'truth',gt),'w')
        replacefile.writelines(outlines)
        replacefile.close()
            
os.remove(os.path.join(folder_name,'truth/classes.txt'))
        
#structure check
subfolders = os.listdir(folder_name)
if len(subfolders)==2 and 'images' in subfolders and 'truth' in subfolders and len(os.listdir(os.path.join(folder_name,'images')))==len(os.listdir(os.path.join(folder_name,'images'))) and len(os.listdir(yolo_model_folder))==3 and len(os.listdir(custom_model_folder))==3:
    pass
else:
    print("|ERROR| Folder structure in {} doesn't match required structure. Required Structure is:".format(folder_name))
    print(required_structure)
    print("|ERROR| Model Folder structure does/doesn't match required structure. Required Structure is:")
    print(required_model_structure)
    os._exit(0)

os.mkdir(os.path.join(folder_name,'yolo_detect'))
os.mkdir(os.path.join(folder_name,'custom_detect'))

wanted_classes = ['person','bicycle','car','motorbike','truck','bus']

def generate_detections(input_folder,model_folder,output_folder):

    #getting config, weights and labelmap files
    for model_file in os.listdir(model_folder):
        if '.cfg' in model_file:
            config = os.path.join(model_folder,model_file)
        elif '.weights' in model_file:
            weights = os.path.join(model_folder,model_file)
        else:
            labels = os.path.join(model_folder,model_file)

    #loading yolo model and reading classes
    net = cv2.dnn.readNet(weights, config)
    classes = []
    with open(labels, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    for image_file in os.listdir(os.path.join(input_folder,'images')):

        #Loading image
        img = cv2.imread(os.path.join(input_folder,"images",image_file))
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape

        #Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        #Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.0:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        list_of_lists = []
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                list_of_row = []
                list_of_row.append(str(classes[class_ids[i]]))
                list_of_row.append(str(confidences[i]).replace("0",""))
                x, y, w, h = boxes[i]
                list_of_row.append(x)
                list_of_row.append(y)
                list_of_row.append(w)
                list_of_row.append(h)
                label = str(classes[class_ids[i]])
                final_text_to_be_added = str(list_of_row).replace(",","")
                final_text_to_be_added = final_text_to_be_added.replace("'","")
                list_of_lists.append(final_text_to_be_added)

        for i_detections in list_of_lists:
            to_be_written = str(i_detections)
            to_be_written = to_be_written.replace("[","")
            to_be_written = to_be_written.replace("]","")
            for label_class in wanted_classes:
                if label_class in to_be_written:
                    txt_file = open(os.path.join(output_folder,image_file[:-3]+"txt"),'a')
                    txt_file.write(to_be_written)
                    txt_file.write("\n")
                    txt_file.close()
                    break                        
                else:
                    continue

generate_detections(folder_name,yolo_model_folder,os.path.join(folder_name,'yolo_detect'))
generate_detections(folder_name,custom_model_folder,os.path.join(folder_name,'custom_detect'))

os.mkdir('required/output')
#subprocess.call("python pascalvoc.py -gt required/truth -det required/custom_detect -gtcoords rel -imgsize 1280,720 -sp everythingtest -np", shell=True)
os.system('python pascalvoc.py -gt required/truth -det required/custom_detect -gtcoords rel -imgsize 1280,720 -sp required/output -np')
#os.system('python review_object_detection_metrics-main/run.py')
    
