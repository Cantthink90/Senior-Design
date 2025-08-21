"""
Facial Recognition for Classroom Attendance Implementation File
Team 6: Ryan Bulharowski, Austen Leslie, Nahum Mekonnen, Van Nipper
"""

# from data import dataset
from parameters import Parameters
from SVD import SVD
from nn import MLP
from crop_faces import Crop_Faces
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import datetime

if __name__ == "__main__":
    
    # Initialize parameters from external class
    pms = Parameters()

    """ DATA PREPROCESSING """
    os.system('cls')
    print("Welcome to the Attendance MLP\n\n")
    user = ""
    while(user.lower().replace(" ","") != "y" ):
        user = input("\nHave all classroom photos been put in Attendance_Photos (y/n):")
        if(user.lower().replace(" ","") == "n"):
            print("Please put all classroom photos in the Attendance folder")
            input("Press enter when photos are in the folder")
    
    cf = Crop_Faces()
    cf.attendance_crop_faces()

    user = ""  
    while(user.lower().replace(" ","") != "y" ):
        user = input("\nPlease look through Identified_Faces. Have all non-face been removed from the folder (y/n):")
        if(user.lower().replace(" ","") == "n"):
            print("Please remove all non-face photos from the Idenfied_Faces folder")
            input("Press enter when all non-face photos are removed")
    

    """ SVD """
    svd = SVD()
    simPhotos = svd.attendanceSVD()

    """ MLP """
    # Initialize input layer with number of nodes equal to output of SVD
    nn = MLP(pms.dims[0] * pms.dims[1])

    folderpath = "../Students/Cropped_Students/"
    Classroom = []
    for people in os.listdir(folderpath):
        if os.path.isdir(folderpath+people):
            Classroom.append(people)
    Classroom.sort()


    namestring = "model" + str(pms.dims[0] * pms.dims[1])
    for layersize in pms.layers:
        namestring += '.' + str(layersize)

    # LOAD NEURAL NETWORK
    nn.load(f'models/{namestring}.npz')

    """ Take Attendance """
    excel_out = []
    attendance = []
    user = ""
    for people in simPhotos:
        output = nn.feedForward(people).flatten()
        predicted_lab = np.argmax(output) + 1
        attendance.append((Classroom[int(predicted_lab)-1],max(output)))
        
        while(user.lower().replace(" ","") != "y" and user.lower().replace(" ","") != "n" ):
            user = input("\nWould you like to see processed photos and bar graphs (y/n):")
        
        if user == "y":

            np_people = np.array(people)
            face_image = np_people.reshape(pms.dims[1], pms.dims[0])

            fig, (ax_img, ax_bar) = plt.subplots(1,2, figsize=(12,5))
            ax_img.imshow(face_image, cmap ='gray')
            ax_img.axis('off')
            ax_img.set_title(f"Predicted: {Classroom[int(predicted_lab)-1]}")

            classes = Classroom
            ax_bar.bar(classes, output, color='blue', alpha=0.7)
            ax_bar.set_xticks(classes)
            ax_bar.set_ylim([0,1])
            ax_bar.set_xlabel("Class")
            ax_bar.set_ylabel("Confidence")
            ax_bar.set_title("Output Confidence Values")

    if user == "y":
        plt.tight_layout()
        plt.show()
        
    
    #print(Classroom)
    pasted = False
    for student in Classroom:
        for face in attendance:
            if face[0] == student:
                if face[1] >= pms.confirmationRate:
                    excel_out.append((student, "Present"))
                    pasted = True
        if not pasted:
            excel_out.append((student, "Absent"))
        pasted =False
    df = pd.DataFrame(excel_out, columns=['Student Names', 'Present/Absence'])
    df.to_excel('../'+datetime.datetime.now().strftime("%m-%d-%Y")+'.xlsx', index=False)


