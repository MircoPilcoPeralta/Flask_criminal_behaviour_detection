from flask import Flask, render_template, request, jsonify, redirect, Response;
from flask_socketio import SocketIO, send
import wmi;

app=Flask(__name__)
app.config["SECRET"] = "secret";
socketIO = SocketIO(app, cors_allowed_origins="*");
 


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


def addCameras ( new_connected_devices_state ):
    global connected_devices;
    connected_devices = new_connected_devices_state["cameras"];





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

                elif (camera["activeModel"] == "multipose-criminal_behaviour_detection"):
                    # img = frame.copy()
                    # img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
                    # input_img = tf.cast(img, dtype=tf.int32)

                    # results = model(input_img)
                    # keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
                    # loop_through_people(frame, keypoints_with_scores, EDGES, 0.2)

                    emit_notification("persona moviendose", "warning")

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