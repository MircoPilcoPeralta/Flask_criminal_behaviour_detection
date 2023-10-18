from flask import Flask, render_template, request, jsonify, redirect, Response;
from flask_socketio import SocketIO, send
import wmi;

import mediapipe as mp
import utils.drawer as drawer;
from  utils.model_configuration import ModelConfiguration;
from utils.multipose_detector import MultiposeDetector;

import tensorflow_hub as hub


from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

import cv2 
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import datetime
from collections import Counter

import base64

app=Flask(__name__)
app.config["SECRET"] = "secret";
socketIO = SocketIO(app, cors_allowed_origins="*");


ANNOTATION_PATH = "./Tensorflow/custom-models/ssd-obj-detection/annotations"
CHECKPOINT_PATH = "./Tensorflow/custom-models/ssd-obj-detection/models"
MODEL_PATH = './Tensorflow/custom-models/ssd-obj-detection/models'
CHECKPOINT_NAME = 'ckpt-6'
CONFIG_PATH = './Tensorflow/custom-models/ssd-obj-detection/pipeline.config'



# carga del archivo pipeline 
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

# carga del modele entrenado
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# restauracion del checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(MODEL_PATH, CHECKPOINT_NAME)).expect_partial()

# archivo de categorizacion
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')



def get_wired_cameras():
    connected_cameras = [];
    index = 0;
    c = wmi.WMI();
    wql = "Select * From Win32_USBControllerDevice";
    devices = c.query(wql);

    for device in devices:
        if(device.Dependent.PNPClass == "Camera"):
            connected_cameras.append( { "id": index,  "name": device.Dependent.Caption} )
            index = index + 1;
    
    return connected_cameras; 

wired_cameras = get_wired_cameras()
connected_devices = []
available_models = ["No_model", "haar_cascade_face_detection", "criminal_behaviour_detection", "object_detection"]

def get_camera_by_id(id):
    global connected_devices;
    find_camera = None;
    for camera in connected_devices:
        if (camera["id"] == id):
            find_camera = camera
    return find_camera;


def addCameras ( new_connected_devices_state ):
    global connected_devices;
    connected_devices = new_connected_devices_state["cameras"];

def update_camera(camera_id, request):
    global connected_devices;

    find_camera = None;
    for camera in connected_devices:
        if (camera["id"] == camera_id):
            find_camera = camera

    if (find_camera == None):
        raise KeyError("Ninguna camara con el id: " + str(id) + " esta conectada al sistema")

    find_camera["activeModel"] = request["activeModel"]
    find_camera["relevantItems"] = request["relevantItems"]
    find_camera["inferencePercentage"] = request["inferencePercentage"]


def get_model(model_name):
    if(model_name == "haar_cascade_face_detection"):
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");

    elif(model_name == "object_detection"):
        return {"name": "object detection model"};

    elif(model_name == "criminal_behaviour_detection"):
        online_model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
        return online_model.signatures['serving_default']



def draw_faces_box(detected_faces, frame):
    for (x,y,w,h) in detected_faces:
        cv2.rectangle( frame, (x,y), (x+w, y+h), (0,255,0), 2)  



def detect(camera_id):
    last_notification_time = datetime.datetime.now()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera = get_camera_by_id(camera_id)
    model = get_model(camera["activeModel"])

    event_detected = False;
    event = {
        "message": "",
        "type": "",
        "image": None
    }

    while (cap.isOpened()):
        ret, frame = cap.read();
        current_time = datetime.datetime.now()

        if(ret):
            if (model != None):
                if (camera["activeModel"] == "haar_cascade_face_detection"):
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = model.detectMultiScale( gray_image, 1.3, 5);

                    draw_faces_box(detected_faces, frame)

                    if(len(detected_faces) > 0):
                        if (current_time - last_notification_time).total_seconds() >= 2:
                            event_detected = True;
                            event["message"] = "Nueva cara detectada"
                            event["type"] = "warning"
                            event["inference"] = 95
                            last_notification_time = current_time


                elif (camera["activeModel"] == "object_detection"):
                    image_np = np.array(frame)
                        
                    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                    detections = detect_fn(input_tensor)
                    
                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                for key, value in detections.items()}
                    
                    detections['num_detections'] = num_detections
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                    label_id_offset = 1
                    image_np_with_detections = image_np.copy()

                    viz_utils.visualize_boxes_and_labels_on_image_array(
                                image_np_with_detections,
                                detections['detection_boxes'],
                                detections['detection_classes']+label_id_offset,
                                detections['detection_scores'],
                                category_index,
                                use_normalized_coordinates=True,
                                max_boxes_to_draw=5,
                                min_score_thresh=.5,
                                agnostic_mode=False)

                    if(len(detections['detection_classes']) > 0):
                        if (current_time - last_notification_time).total_seconds() >= 2:
                            
                            max_value_index = np.argmax(detections['detection_scores'])
                            max_percentage = detections['detection_scores'][max_value_index]

            
                            object_detected = "cuchillo";

                            if(detections['detection_classes'][max_value_index] == 1): 
                                object_detected = "pistola";
                            
                            if( max_percentage >= int(camera["inferencePercentage"])/100.0 ):
                                event_detected = True;
                                event["message"] = "Se ha detectado un "
                                event["type"] = "warning"
                                event["inference"] = 95
                                last_notification_time = current_time
                            # ToDo revisar el tiempo
                            last_notification_time = current_time

                    frame = cv2.resize(image_np_with_detections, (800, 600))

            (flag, encodedImage) = cv2.imencode(".jpg", frame);
            
            if not flag:
                continue;
            if(event_detected):
                emit_notification(event, encodedImage);
                event_detected = False;
            yield( b'--frame\r\n' b'Content-Type: image\jepg\r\n\r\n' + bytearray(encodedImage) + b'\r\n' )
        else:
            cap.release(); 
            break; 

    cap.release();











@app.route("/")
def dashboardPage():
    return render_template("dashboard/dashboard.html");

@app.route("/auth/login")
def loginPage():
    return render_template("login/login.html");

@app.route("/auth/register")
def registerPage():
    return render_template("register/register.html");



@app.route("/surveillance/one-camera-image")
def oneCameraImage():
    if(len(connected_devices) != 1 ):
        return redirect("/surveillance");
    return render_template("surveillance/one_camera_image.html", connected_cameras=wired_cameras, connected_devices=connected_devices, available_models=available_models );


@app.route("/surveillance/two-camera-image")
def twoCameraImage():
    if(len(connected_devices) != 2 ):
        return redirect("/surveillance");
    return "two camera image"

@app.route("/surveillance/more-cameras-image")
def moreThanTwoCameraImage():
    if(len(connected_devices) == 0 or len(connected_devices) < 3 ):
        return redirect("/surveillance");
    return "more than one camera image"

@app.route("/surveillance/no-cameras-added")
def noCamerasAdded():
    global connected_devices;
    if(len(connected_devices) > 0 ):
        return redirect("/surveillance");
    return render_template("surveillance/no_camera_connected.html", connected_cameras = wired_cameras, connected_devices=connected_devices );

@app.route("/surveillance")
def CameraImagePage():
    global connected_devices;

    if(len(connected_devices) == 0):
        return redirect("surveillance/no-cameras-added");

    elif(len(connected_devices) == 1 ):
        return redirect("surveillance/one-camera-image");

    elif(len(connected_devices) == 2 ):
        return redirect("surveillance/two-camera-image");

    elif(len(connected_devices) > 2 ):
        return redirect("surveillance/more-cameras-image");

@app.route("/surveillance/register", methods = ["POST"])
def registerCameraPage():
    addCameras(request.json)
    return redirect("surveillance", 200);


@app.route("/video_feed/<id>")
def video_feed(id):
    return Response(detect( int(id) ), mimetype = "multipart/x-mixed-replace; boundary=frame")






if __name__ == "__main__":
    socketIO.run(app, host = "localhost");