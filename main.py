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

wired_cameras = get_wired_cameras();
connected_devices = [];

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
    return "one camera image"

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










if __name__ == "__main__":
    socketIO.run(app, host = "localhost");