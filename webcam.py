import requests
import os
import numpy as np
import cv2
import face_recognition


def recognize():
    encodings = os.listdir("encoding")
    images = os.listdir("images")
    # load data
    known_face_encodings = []
    known_face_names = []
    for path, enc in zip(images, encodings):
        img= face_recognition.load_image_file(f'images/{path}')
        known_face_encodings.append(face_recognition.face_encodings(img)[0])
        known_face_names.append(path.split(".")[0])
    #Reading the input image now. 
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    cam = cv2.VideoCapture(0) #index value of webcam
    face_detector1=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cam.read()
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

     # Displaying the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Drawing a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0), 1)

        cv2.imshow(frame)
        if cv2.waitKey() == 27: 
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()