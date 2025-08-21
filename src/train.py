import os
import numpy as np
from parameters import Parameters
from nn import MLP
from SVD import SVD
from XY_Set import XY_Set
from crop_faces import Crop_Faces
import matplotlib.pyplot as plt

# SET UP INITIAL VALUES
pm = Parameters()
svd = SVD()
nn = MLP(pm.dims[0] * pm.dims[1])
cf = Crop_Faces()
os.system("cls")


# SVD
os.system('cls')
if input("Have you updated the Student Photos in the Uncropped_Students folder? (y/n): ") == 'y':
    cf.training_crop_faces()
    if input("Have all non-face photos been removed from the Identified_Faces folder? (y/n): ") == 'n':
        print("Please remove all non-face photos from the Identified_Faces folder.")
        input("Press enter when all non-face photos are removed.")

os.listdir("../Students/Cropped_Students/")
svd.trainingPhotoData()

# AUTOMATICALLY GET OUTPUT FILE BASED ON PARAMETERS
svd_name = f'SVD_Datasets/Batch_{pm.batch_size}+Rank_{pm.rank}+Dim_{pm.dims[0]}_{pm.dims[1]}+Output_{pm.outputNum}.txt'
batch_Size = pm.batch_size
input_Nodes = nn.layers_sizes[0]

folderpath = "../Students/Cropped_Students/"
Classroom = []
for people in os.listdir(folderpath):
    if os.path.isdir(folderpath+people):
        Classroom.append(people)

Classroom.sort()

# CREATE X AND Y SETS
# num_Batches = countBatches()
xySet = XY_Set(svd_name=svd_name, batch_Size=batch_Size, dims=pm.dims, input_Nodes=input_Nodes, Classroom=Classroom)
X_train, Y_train, X_test, Y_test = xySet.X_train, xySet.Y_train, xySet.X_test, xySet.Y_test

# CHECK IF MODEL NEEDS TO BE TRAINED OR LOADED
namestring = "model" + str(pm.dims[0] * pm.dims[1])

if input("Train or load model? (t/l): ") == 't':

    # BUILD, TRAIN AND SAVE MODEL
    for layersize in pm.layers:
        nn.addLayer(layersize)
        namestring += '.' + str(layersize)
    
    nn.train(X_train, Y_train, pm.epochs, pm.learningRate, Xtest=X_test, Ytest=Y_test)
    nn.save(f'models/{namestring}.npz')

else:

    # LOAD MODEL
    for layersize in pm.layers:
        namestring += '.' + str(layersize)

    if nn.load(f'models/{namestring}.npz') == -1:
        exit()

# EXAMINE RESULTS with MATPLOTLIB

for data, label in zip(X_test, Y_test):
        
    output = nn.feedForward(data.reshape(1, -1)).flatten()
    predicted_lab = np.argmax(output) + 1
    face_image = data.reshape(pm.dims[1], pm.dims[0])

    fig, (ax_img, ax_bar) = plt.subplots(1,2, figsize=(12,5))
    ax_img.imshow(face_image, cmap ='gray')
    ax_img.axis('off')
    ax_img.set_title(f"True: {Classroom[int(label)-1]}\nPredicted: {Classroom[int(predicted_lab)-1]}")

    classes = Classroom
    ax_bar.bar(classes, output, color='blue', alpha=0.7)
    ax_bar.set_xticks(classes)
    ax_bar.set_ylim([0,1])
    ax_bar.set_xlabel("Class")
    ax_bar.set_ylabel("Confidence")
    ax_bar.set_title("Output Confidence Values")

    plt.tight_layout()
plt.show()