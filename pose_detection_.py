#importing the opencv and mediapipe modules
import cv2
import mediapipe  as mp

#creating a holistic model and also setting up the drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



#initiating the holistic model
with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:

    #webcam feed
    img = cv2.VideoCapture(0)

    while True:
        ret,frame = img.read()

        #convert the BGR to RGB

        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #making the actual detections

        results = holistic.process(image)
        print(results.face_landmarks )



        #convert the BGR to RGB

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #drawing the detected landmarks on the screen
        mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow('FEED',image)

        if cv2.waitKey(1) == ord('p'):
            break

img.release()
cv2.destroyAllWindows()


