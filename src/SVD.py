"""
CLASS SVD written by Ryan Bulharowski and Austen Leslie
Edited by Van Nipper
Constructor: Project parameters object
A class which is used to perform Singular Value Decomposition on photos.
Outputs the photo data as a file to be read by the neural network.
"""

import os
import random
import numpy as np
from PIL import Image
from parameters import Parameters

pm = Parameters()

class SVD():

    def __init__(self):
        """
        SVD CONTSTRUCTOR
        Input: Project parameters object
        Output: SVD Object
        Creates an SVD object with the hyperparameters as specified by the project.
        """

        # Initialize hyperparameters from parameters object
        global pm
        
    def attendanceSVD(self):
        """
        FUNCTION attendanceSVD
        Input: None
        Output: SVD images with no training labels
        Performs SVD on the photos in the Processed_Photos folder and returns
        only an object with the image data.
        """
        filepath = "../Identified_Faces/"
        svdImages = []

        # For each photo in the filepath TODO: THIS IS A FOLDER OF IMAGES
        for photo in os.listdir(filepath):
            
            # Temp list for SVD data
            tempSVD = []
            
            # Resizes photo if it doesn't match hyperparameters
            image = Image.open(filepath+photo).convert("L")
            if image.size != pm.dims:

                image = image.resize(pm.dims)

            # Converts image to numpy array
            image = np.array(image)

            # Perform SVD 
            U, S, V = np.linalg.svd(image)
            sum = np.zeros_like(image, dtype=np.float64)

            for i in range(pm.rank):

                sum += S[i]* np.outer(U[:,i],V[i,:])
            
            # Normalize the data
            sum /= 255

            # Add the data to the tempSVD list
            for element in np.nditer(sum):

                tempSVD.append(element)

            # Add the tempSVD list to the svdImages list
            svdImages.append(tempSVD)

        return svdImages

    def trainingPhotoData(self):
        """
        FUNCTION: trainingPhotoData
        Input: None
        Output: None (void)
        Navigates to the SVD_Input folder and performs SVD on every photo, saving the data
        in the SVD_Datasets folder.
        """

        # Do not perform SVD if the file already exists
        if os.path.exists(f"SVD_Datasets/Batch_{pm.batch_size}+Rank_{pm.rank}+Dim_{pm.dims[0]}_{pm.dims[1]}+Output_{pm.outputNum}.txt"):
            
            print("The SVD File Already Exists. Bypassing SVD File Creation")
            return
        
        else:

            print("Creating SVD File")

        # Folder name that contains the photo folders
        filepath = "../Students/Cropped_Students/"        
        trainingData = []
        testData = []

        maxData = 9999
        for name in os.listdir(filepath):
            if os.path.isdir(filepath+name):
                maxData = min(maxData,len(os.listdir(filepath+name)))

        # For each item in the filepath
        for name in os.listdir(filepath):

            # Check if a folder
            if os.path.isdir(filepath+name):

                # Grabs each photo in the named folder and performs SVD
                tempStore = []
                for photos in os.listdir(filepath+name):

                    tempStore.append(self.__trainingSVD(filepath+name+"/"+photos,name))
                
                # Randomly remove data until the maxData is reached
                while len(tempStore) > maxData:
                    ranNum = random.randint(0,len(tempStore)-1)
                    tempStore.pop(ranNum)


                # 20% of Data is saved to Test Dataset and then removed from tempStore
                for x in range(int(len(tempStore)*0.2)):
                    ranNum = random.randint(0,len(tempStore)-1)
                    testData.append(tempStore[ranNum])
                    tempStore.pop(ranNum)

                # Remaining data goes to Training Dataset
                for y in tempStore:
                    trainingData.append(y)
        
        """ CREATE SVD DATA FILE """

        # Create a list of empty strings to store the batches
        batches = [""] * (len(trainingData) // pm.batch_size + 1)
        testBatch = ""

        # Randomly sort the training data into batches
        modifiyTraining = trainingData.copy()
        for x in range(len(batches)):

            if len(modifiyTraining) < pm.batch_size:
                for y in range(len(modifiyTraining)):
                    ranNum = random.randint(0,len(modifiyTraining)-1)
                    batches[x] += modifiyTraining[ranNum]
                    modifiyTraining.pop(ranNum)
            else:
                for y in range(pm.batch_size):
                    ranNum = random.randint(0,len(modifiyTraining)-1)
                    batches[x] += modifiyTraining[ranNum]
                    modifiyTraining.pop(ranNum)
        
        # Add the remaining data to the last batch
        for x in testData:

            testBatch += x

        # Filename structure
        fileName = f"SVD_Datasets/Batch_{pm.batch_size}+Rank_{pm.rank}+Dim_{pm.dims[0]}_{pm.dims[1]}+Output_{pm.outputNum}.txt"

        # Create file and add training data
        try:
            file = open(fileName,"x")
            # Batches are written first
            for lines in batches:
                file.write(lines+"\n")
            # This line separates train data from test data
            file.write("###############\n")
            file.write(testBatch)
            file.close()
        except:
            print("File already exists")

    def __trainingSVD(self, photoPath, idenity):
        """
        FUNCTION: __trainingSVD
        Input: photoPath (str), idenity (str)
        Output: tempSVD (str)
        Performs SVD on the photo and returns the data in a string format.
        """
        
        # Convert photos to BW
        if photoPath[-3:] == "jpg":
            image = Image.open(photoPath).convert("L")
        else:
            return ""

        # Resize photo
        if image.size != pm.dims:

            image = image.resize(pm.dims)

        # Initial setup for the file structure
        # Each line in the outputted file is a batch.
        # Starts with the idenity of the person followed by their corresponding SVD data.
        # '&' break up identities and data, while spaces break up individual data values
        tempSVD = idenity + "&"
        
        image_array = np.array(image)
        U, S, V = np.linalg.svd(image_array)
        sum = np.zeros_like(image_array, dtype=np.float64)
        for i in range(pm.rank):

            sum += S[i]* np.outer(U[:,i],V[i,:])

        sum /= 255

        for element in np.nditer(sum):

            tempSVD += str(element) + " "

        tempSVD += "&"
        return tempSVD
