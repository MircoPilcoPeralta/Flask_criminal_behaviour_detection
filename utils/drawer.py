import cv2

def draw_rectangle_on_face(gray_image, frame, model):
    faces = model.detectMultiScale( gray_image, 1.3, 5);
    for (x,y,w,h) in faces:
            cv2.rectangle( frame, (x,y), (x+w, y+h), (0,255,0), 2)   