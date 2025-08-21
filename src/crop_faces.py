#chatgpt was used when writing this code
# NOTE: ERASES ALL CONTENT FROM PROCESSED PHOTOS FOLDER 

import os
import cv2
from parameters import Parameters
import shutil

param = Parameters()
class Crop_Faces():

    def __init__(self):
        global param


    def training_crop_faces(self):
        print("Cropping faces from Uncropped_Students folder...")
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if haar_cascade.empty():
            print("Haar Cascade failed to load. Check the path or OpenCV installation.")
            return

        input_dir = "../Students/Uncropped_Students/"
        output_dir = "../Students/Cropped_Students/"
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist.")
            return

        if not os.listdir(input_dir):
            print(f"Input directory {input_dir} is empty.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # DELETE ALL FILES FROM OUTPUT DIR
        """try:
            for entry in os.listdir(output_dir):
                    item = os.path.join(output_dir, entry)
                    if os.path.isfile(item) or os.path.islink(item):
                        os.remove(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item) 
            print("All files deleted successfully.")
        except OSError:
            print("Error occurred while deleting files.")"""

        for foldername in os.listdir(input_dir):
            i = 1
            #print(f"Processing file: {filename}")
            for filename in os.listdir(input_dir+foldername):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(input_dir,foldername, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Image {filename} could not be loaded. Skipping...")
                        continue

                    #print("Image loaded successfully.")
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_image = cv2.equalizeHist(gray_image)
                    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

                    #print("Converted to grayscale and applied pre-processing.")

                    faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.15, minNeighbors=10, minSize=(50, 50))
                    print(f"Found {len(faces)} face(s) in {filename}")

                    for j, face in enumerate(faces):
                        x,y,w,h = faces[j]
                        face = image[y:y+h, x:x+w]
                        resized_face = cv2.resize(face, param.dims)
                        resized_face = cv2.cvtColor(resized_face,cv2.COLOR_BGR2GRAY)
                    
                        folderOutput = foldername.replace(" ","_")
                        output_folder_path = os.path.join(output_dir, folderOutput)
                        if not os.path.exists(output_folder_path):
                            os.makedirs(output_folder_path)

                        output_path = os.path.join(output_dir, folderOutput + f"/{i}_face_.jpg")
                        i += 1
                        cv2.imwrite(output_path, resized_face)
                        print(f"Saved face to {output_path}")

    def attendance_crop_faces(self):
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if haar_cascade.empty():
            print("Haar Cascade failed to load. Check the path or OpenCV installation.")
            return

        input_dir = "../Attendance_Photos/"
        output_dir = "../Identified_Faces/"
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist.")
            return

        if not os.listdir(input_dir):
            print(f"Input directory {input_dir} is empty.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # DELETE ALL FILES FROM OUTPUT DIR
        try:
            for entry in os.listdir(output_dir):
                    item = os.path.join(output_dir, entry)
                    if os.path.isfile(item) or os.path.islink(item):
                        os.remove(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item) 
            print("All files deleted successfully.")
        except OSError:
            print("Error occurred while deleting files.")

        i=0
        for filename in os.listdir(input_dir):
            #print(f"Processing file: {filename}")
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Image {filename} could not be loaded. Skipping...")
                    continue

                #print("Image loaded successfully.")
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.equalizeHist(gray_image)
                gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

                #print("Converted to grayscale and applied pre-processing.")

                faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.15, minNeighbors=10, minSize=(50, 50))
                print(f"Found {len(faces)} face(s) in {filename}")

                for j, face in enumerate(faces):
                    x,y,w,h = faces[j]
                    face = image[y:y+h, x:x+w]
                    resized_face = cv2.resize(face, param.dims)
                    resized_face = cv2.cvtColor(resized_face,cv2.COLOR_BGR2GRAY)

                    output_path = os.path.join(output_dir, f"{i + 1}_face_.jpg")
                    i+=1
                    cv2.imwrite(output_path, resized_face)
                    print(f"Saved face to {output_path}")
