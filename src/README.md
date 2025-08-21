#src Folder Overview
This README summarizes all of files and folder in the src folder.

##### models
This folder stores the neural network model values. The storage of these neural network files allows for quick attendance checks since the model has already been trained.
##### SVD_Datasets
This folder stores the SVD values of all the photos put in the Students folder. This allows for quicker training if there are multiple training attempts with the same dataset.
##### crop_faces.py
This file focuses on identify people's faces in photos. This file is outside of the project's current scope and ChatGPT was used. This file crops and grayscales photos and stores them in either Identified_Faces folder, for when taking attendance, or stores them in Students/Cropped_Students , for when training the model.
##### main.py
This file operates the code when you want to take attendance of a classroom. It then will output the attendance of the class, based on the photos in Attendance_Photos, to an Excel file.
##### nn.py
This file is the MLP neural network file. It is an abstract file that is controlled by main.py and train.py. This file includes feed forward functions, backpropogation, save neural network to file, and loading a neural network file.
##### parameters.py
This file stores the hyperparameters for neural network. This includes the rank used for SVD, epochs, neual network structure etc.
##### setup.py
This file is used to install all the necessary libraries to run this program. To utilize this file, you have to download a python environment such as Anaconda and then run this python file when you activated the python environment.
##### SVD.py
This file performs SVD on the photos that it is passed. The SVD process reduces the complexity of each photo and then outputs the SVD results either to a list (used in main.py) or a file (used in train.py).
##### train.py
This python file controls the neural network, SVD, parameters, and XY_Set files to train a MLP neural network. The user is asked to make sure all of the photos are put in the correct place, performs SVD on all photos, and train the neural network. The file might have to be run twice after changing the dataset.
##### XY_Set.py
This file takes the SVD Dataset file and processes it into lists such that the data can be passed into the neural network. It also divides the data into lists containing the training data and lists containing the testing data.
