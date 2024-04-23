from flask import Flask, render_template, request, jsonify, redirect, Response;
from flask_socketio import SocketIO, send
import wmi;

import mediapipe as mp
import utils.drawer as drawer;
from utils.multipose_detector import MultiposeDetector;

import tensorflow as tf
import tensorflow_hub as hub

import re

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

import cv2 
import numpy as np

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
from pygame import mixer 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard


########
import gevent
from engineio.async_drivers import gevent
######

import jinja2.ext

mixer.init() 
mixer.music.load("./assets/sounds/sonido_alarma.mp3") 

mixer.music.set_volume(0.7) 


app=Flask(__name__)
app.config["SECRET"] = "secret";
socketIO = SocketIO(app, cors_allowed_origins="*", async_mode='threading');


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

ANNOTATION_PATH = "./assets/models/custom-ssd/annotations"
CHECKPOINT_PATH = "./assets/models/custom-ssd/models"
MODEL_PATH = './assets/models/custom-ssd/models'
CHECKPOINT_NAME = 'ckpt-4'
CONFIG_PATH = './assets/models/custom-ssd/pipeline.config'


# carga del archivo pipeline 
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

# carga del modele entrenado
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# restauracion del checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(MODEL_PATH, CHECKPOINT_NAME)).expect_partial()

# archivo de categorizacion
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')



actions = np.array(['golpe_derecha', 'golpe_izquierda','caminando_izquierda','caminando_derecha'])
violent_actions = ['golpe']



def generate_model(weigths_path):
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(10,36)))
    lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(actions.shape[0], activation='softmax'))
    lstm_model.load_weights(weigths_path);

    return lstm_model;


lstm_model_1 = Sequential()
lstm_model_1.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(10,36)))
lstm_model_1.add(LSTM(64, return_sequences=False, activation='relu'))
lstm_model_1.add(Dense(64, activation='relu'))
lstm_model_1.add(Dense(32, activation='relu'))
lstm_model_1.add(Dense(actions.shape[0], activation='softmax'))
lstm_model_1.load_weights('./latest.h5');


lstm_model_2 = Sequential()
lstm_model_2.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(10,36)))
lstm_model_2.add(LSTM(64, return_sequences=False, activation='relu'))
lstm_model_2.add(Dense(64, activation='relu'))
lstm_model_2.add(Dense(32, activation='relu'))
lstm_model_2.add(Dense(actions.shape[0], activation='softmax'))
lstm_model_2.load_weights('./latest.h5');

models_conf = [
    {
        'model': lstm_model_1,
        'sequence': [],
        'predictions':[]
    }
    ,
    {
        'model': lstm_model_2,
        'sequence': [],
        'predictions':[]
    }
]


predicion_number = 1;
threshold = 1

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


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


def releaseAllCameras():
    global cap;
    if(cap):
        cap.release(); 


read_1 = True
cap = None
wired_cameras = get_wired_cameras()
available_models = ["No_model", "haar_cascade_face_detection", "criminal_behaviour_detection", "object_detection"]
connected_devices = []
selected_video = []

def is_camera_id(id):
    valid_ids = ["0","1","2","3"];
    return id in valid_ids;

def get_camera_by_id(id):
    global connected_devices;
    print(id)
    print(type(id))
    find_camera = None;
    for camera in connected_devices:
        if (camera["id"] == id):
            find_camera = camera
    return find_camera;

def add_cameras ( new_connected_devices_state ):
    global connected_devices;
    connected_devices = new_connected_devices_state["cameras"];

def update_camera(camera_id, request):
    global connected_devices;

    find_camera = None;
    for camera in connected_devices:
        if (camera["id"] == camera_id):
            find_camera = camera

    if (find_camera == None):
        raise KeyError("Ninguna camara con el id: " +  id + " esta conectada al sistema")

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

        

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold): 
    for person in keypoints_with_scores:
        pose_data = np.array(person[5:])
        normalized_points=draw_keypoints(frame, pose_data, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    filtered_keypoints = [];
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
            filtered_keypoints.append(kp);
        else:
            filtered_keypoints.append(np.zeros(3))
    return(filtered_keypoints) 



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)


def emit_notification(event, encodedImage):
    timestamp = datetime.datetime.now();
    socketIO.emit( 'event', {
        'cameraName': event["cameraName"],
        'type': event["type"],
        'message': event["message"],
        'inference': event["inference"],
        'date': timestamp.strftime("%m-%d-%Y"),
        'time': timestamp.strftime("%H:%M:%S"),
        'encodedImage': encodedImage.tolist()
    })

def draw_faces_box(detected_faces, frame):
    for (x,y,w,h) in detected_faces:
        cv2.rectangle( frame, (x,y), (x+w, y+h), (0,255,0), 2)  


def detect(camera_id):

    last_notification_time = datetime.datetime.now()

    cap = cv2.VideoCapture(int(camera_id))
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
                            event["inference"] = camera["inferencePercentage"]
                            event["cameraName"] = camera["cameraName"]
                            last_notification_time = current_time

                elif (camera["activeModel"] == "criminal_behaviour_detection"):
                    img = frame.copy()
                    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
                    input_img = tf.cast(img, dtype=tf.int32)

                    results = model(input_img)
                    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
                    loop_through_people(frame, keypoints_with_scores, EDGES, 0.2)

                    for person_index in range (0, len(models_conf)):

                        person_kp = keypoints_with_scores[person_index]
                        pose_data = np.array(person_kp[5:])
                        normalized_points=draw_keypoints(frame, pose_data, 0.5)

                        person_configuration = models_conf[person_index]; 
                        predictions = person_configuration['predictions']

                        person_configuration['sequence'].append(np.array(normalized_points).flatten())

                        if len(person_configuration['sequence']) == 10:
                            res = lstm_model_1.predict(np.expand_dims(person_configuration['sequence'], axis=0))[0]
                            predictions.append(np.argmax(res))
                            person_configuration['sequence'] = []; 
                                
                            if res[np.argmax(res)] >= threshold: 
                                if (current_time - last_notification_time).total_seconds() >= 2:
                                    event_detected = True;
                                    event["message"] = actions[np.argmax(res)].split("_")[0]
                                    event["type"] = "warning"
                                    event["inference"] = camera["inferencePercentage"]
                                    event["cameraName"] = camera["cameraName"]
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
                                min_score_thresh=0.75,
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
                                event["message"] = "Se ha detectado un " +object_detected
                                event["type"] = "warning"
                                event["inference"] = camera["inferencePercentage"]
                                event["cameraName"] = camera["cameraName"]
                                last_notification_time = current_time
                            # ToDo revisar el tiempo
                            last_notification_time = current_time

                    frame = cv2.resize(image_np_with_detections, (800, 600))

            (flag, encodedImage) = cv2.imencode(".jpg", frame);
            
            if not flag:
                continue;
            if(event_detected):
                mixer.music.play() 
                emit_notification(event, encodedImage);
                event_detected = False;

            yield( b'--frame\r\n' b'Content-Type: image\jepg\r\n\r\n' + bytearray(encodedImage) + b'\r\n' )
        else:
            cap.release(); 
            break; 
    cap.release();

def detectsss(video_path):
    last_notification_time = datetime.datetime.now()
    windows_video_path = video_path
    windows_video_path.replace("/","\\")
    cap = cv2.VideoCapture(windows_video_path )
    camera = get_camera_by_id(video_path)

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
                            event["inference"] = camera["inferencePercentage"]
                            event["cameraName"] = "Grabación"
                            last_notification_time = current_time

                elif (camera["activeModel"] == "criminal_behaviour_detection"):
                    img = frame.copy()
                    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
                    input_img = tf.cast(img, dtype=tf.int32)

                    results = model(input_img)
                    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
                    loop_through_people(frame, keypoints_with_scores, EDGES, 0.2)

                    for person_index in range (0, len(models_conf)):

                        person_kp = keypoints_with_scores[person_index]
                        pose_data = np.array(person_kp[5:])
                        normalized_points=draw_keypoints(frame, pose_data, 0.5)

                        person_configuration = models_conf[person_index]; 
                        predictions = person_configuration['predictions']

                        person_configuration['sequence'].append(np.array(normalized_points).flatten())

                        if len(person_configuration['sequence']) == 10:
                            res = lstm_model_1.predict(np.expand_dims(person_configuration['sequence'], axis=0))[0]
                            predictions.append(np.argmax(res))
                            person_configuration['sequence'] = []; 
                                
                            if res[np.argmax(res)] >= int(camera["inferencePercentage"])/100.0: 
                                behaviour_detected = actions[np.argmax(res)].split("_")[0]
                                if behaviour_detected in violent_actions:
                                    if (current_time - last_notification_time).total_seconds() >= 2:
                                        event_detected = True;
                                        event["message"] = actions[np.argmax(res)].split("_")[0]
                                        event["type"] = "warning"
                                        event["inference"] = camera["inferencePercentage"]
                                        event["cameraName"] = "Grabación"
                                        last_notification_time = current_time
                                        mixer.music.play()
                                        print(res[np.argmax(res)])
                            



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
                                min_score_thresh= int(camera["inferencePercentage"])/100.0,
                                agnostic_mode=False)
                    if(len(detections['detection_classes']) > 0):
                        if (current_time - last_notification_time).total_seconds() >= 0.4:
                            
                            max_value_index = np.argmax(detections['detection_scores'])
                            max_percentage = detections['detection_scores'][max_value_index]

            
                            object_detected = "cuchillo";

                            if(detections['detection_classes'][max_value_index] == 1): 
                                object_detected = "pistola";
                            
                            if( max_percentage >= int(camera["inferencePercentage"])/100.0 ):
                                mixer.music.play() 
                                event_detected = True;
                                event["message"] = "Se ha detectado un " +object_detected
                                event["type"] = "warning"
                                event["inference"] = str(max_percentage)
                                event["cameraName"] = "Grabación"
                                last_notification_time = current_time
                            # ToDo revisar el tiempo
                            last_notification_time = current_time

                    frame = cv2.resize(image_np_with_detections, (800, 600))

            (flag, encodedImage) = cv2.imencode(".jpg", frame);
            cv2.waitKey(10)
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
    
    if( is_camera_id(connected_devices[0]["id"]) ):
        return render_template("surveillance/one_camera_image.html", connected_cameras=wired_cameras, connected_devices=connected_devices, available_models=available_models );

    else:
        return render_template("surveillance/video_image.html", connected_cameras=wired_cameras, connected_devices=connected_devices, available_models=available_models)



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
    global selected_video
    selected_video = []
    add_cameras(request.json)
    return redirect("surveillance", 200);


@app.route("/surveillance/camera/config", methods=["PATCH"])
def updateConnectedCameras():
    args = request.args
    id = args.to_dict()["id"];
    result_code = 200
    try:
        update_camera(id, request.json)
    except KeyError:
        result_code = 404
    return redirect("surveillance", result_code)


@app.route("/surveillance/camera", methods = ["DELETE"])
def disconnectCamera():
    global connected_devices;
    args = request.args
    id = args.to_dict()["id"];
    connected_devices = [camera for camera in connected_devices if camera["id"] != id]
    return redirect("surveillance", 200);

@app.route("/reports")
def reports():
    return render_template("reports/reports.html")

@app.route("/video_feed")
def video_feed():
    args = request.args
    id = args.to_dict()["id"];
    if is_camera_id(id):
        return Response(detect((id) ), mimetype = "multipart/x-mixed-replace; boundary=frame")

    return Response(detectsss(id), mimetype = "multipart/x-mixed-replace; boundary=frame")

@socketIO.on('connect')
def connect():
    print('new client connected');


@socketIO.on("detections")
def handle_detections():
    send("mensaje del servidor", broadcast=True)


if __name__ == "__main__":
    socketIO.run(app, host = "localhost");