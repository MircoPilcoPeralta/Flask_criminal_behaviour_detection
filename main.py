from flask import Flask, render_template, request, jsonify, redirect, Response;
from flask_socketio import SocketIO, send

app=Flask(__name__)
app.config["SECRET"] = "secret";
socketIO = SocketIO(app, cors_allowed_origins="*");
 
@app.route("/")
def dashboardPage():
    return "Dashboard page";

@app.route("/auth/login")
def loginPage():
    return render_template("login/login.html");

@app.route("/auth/register")
def registerPage():
    return render_template("register/register.html");

if __name__ == "__main__":
    socketIO.run(app, host = "localhost");