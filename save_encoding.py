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

for i in known_images_enc:
    with open(f"encoding/{i['name']}.enc", 'wb') as f:
        pickle.dump(i['enc'], f)
