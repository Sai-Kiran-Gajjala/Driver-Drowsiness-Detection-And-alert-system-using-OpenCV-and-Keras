from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from twilio.rest import Client
from geopy.geocoders import Nominatim
import threading

app = Flask(__name__)
socketio = SocketIO(app)

mixer.init()

account_sid = 'AC62debc1eee27ae150598c625d262267d'
auth_token = '2baf86fce837f4debd2934075699a1ce'
client = Client(account_sid, auth_token)

def send_sms(message):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{current_datetime}\n{message}"
    try:
        client.messages.create(
            to='+918500286325',
            from_='+12565948486',
            body=full_message)
        print("Message sent successfully")
    except Exception as e:
        print("Error sending message:", str(e))

def get_gps_coordinates():
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode("me")
    if location:
        latitude = location.latitude
        longitude = location.longitude
        google_maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
        return google_maps_link
    return "GPS coordinates not available"

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
blink_thresh = 3
blink_duration_thresh = 1
drowsy_duration_thresh = 2
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def recommend_music(drowsiness_level):
    music_tracks = {
        'low': 'powerful-beat-121791.mp3',
        'medium': 'adrenaline-buzz-pecan-pie-main-version-01-11-3280.mp3',
        'high': 'thunder-trinity-main-version-01-56-11717.mp3'
    }
    if drowsiness_level in music_tracks:
        music_track = music_tracks[drowsiness_level]
        mixer.music.load(music_track)
        mixer.music.play(-1)
    else:
        print("No music recommendation for this drowsiness level")

def stop_music():
    mixer.music.stop()

def stop_application(event):
    if event == 27:
        print("Stopping the application...")
        stop_music()
        cv2.destroyAllWindows()
        socketio.emit('close_app')
        socketio.stop()

def calculate_drowsiness_level():
    pass

def take_readings():
    flag = 0
    last_alert_time = time.time()
    while True:
        print("Flag value:", flag)
        socketio.emit('drowsiness_level', {'level': flag})
        if time.time() - last_alert_time > 30:
            send_alert()
            last_alert_time = time.time()
        time.sleep(30)

def send_alert():
    send_sms("Driver is feeling sleepy. Please check on them.")
    socketio.emit('alert', {'message': 'Driver is feeling sleepy. Please check on them.'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    cap = cv2.VideoCapture(0)
    flag = 0
    alert_sent = False
    last_alert_time = 0
    blink_count = 0
    start_blink_time = 0
    start_drowsy_time = 0
    alert_display_time = None

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            if ear < thresh:
                blink_count += 1
                if blink_count == 1:
                    start_blink_time = time.time()
                
                if blink_count > blink_thresh and time.time() - start_blink_time < blink_duration_thresh:
                    blink_count = 0
                else:
                    if not alert_sent:
                        flag += 1
                        if flag >= frame_check:
                            cv2.putText(frame, "ALERT! Driver is getting Sleep", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "ALERT!", (10,325),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            recommend_music('high') 
                            alert_sent = True
                            start_drowsy_time = time.time()
                            socketio.emit('alert', {'message': 'Driver is feeling sleepy. Please check on them.'})
                            alert_display_time = time.time()
                    last_alert_time = time.time()
            elif alert_sent and time.time() - last_alert_time > 60:
                alert_sent = False
            else:
                blink_count = 0

        if alert_display_time and time.time() - alert_display_time > 3:
            cv2.putText(frame, "", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "", (10,325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        event = cv2.waitKey(1)
        stop_application(event)
                
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@socketio.on('connect')
def test_connect():
    print('Client connected')
    socketio.emit('alert', {'message': '', 'flag': 0})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    t = threading.Thread(target=take_readings)
    t.daemon = True
    t.start()
    socketio.run(app, debug=True)
