#Importing libraries
import face_recognition
import os
import pickle
images = os.listdir('images')

known_images_enc = []
for image in images:
    ki = face_recognition.load_image_file("images/"+image)
    print("saving:", image)
    known_images_enc.append({
        'name' : image.replace(' ','_').split('.')[0],
        'enc': face_recognition.face_encodings(ki)[0]
    })

#Accessing know images:
known_image = face_recognition.load_image_file("bill gates.jpg")
known_image = face_recognition.load_image_file("vanshaj.jpg")
known_image = face_recognition.load_image_file("zaid kamil.jpg")

billgates_encoding = face_recognition.face_encodings(known_image)[0]
vanshaj_encoding = face_recognition.face_encodings(known_image)[0]
zaid_kamil_encoding = face_recognition.face_encodings(known_image)[0]

#saving encoding to files
for i in known_images_enc:
    with open(f"encoding/{i['name']}.enc", 'wb') as f:
        pickle.dump(i['enc'], f)
