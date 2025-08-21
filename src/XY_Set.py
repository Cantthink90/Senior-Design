import re
import numpy as np

class XY_Set:
    def __init__(self, svd_name, batch_Size, dims, input_Nodes, Classroom):
        self.svd_name = svd_name
        self.batch_Size = batch_Size
        self.input_Nodes = input_Nodes
        self.Classroom = Classroom
        self.dims = dims
        self.num_batches = self.countBatches()
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.createXYSet()

    def countBatches(self):
        # Open file to get number of batches
        with open(self.svd_name, 'r') as file:
            num_Batches = 0

            while file.readline()[0] != '#':
                num_Batches += 1
            
            file.close()
    
        return num_Batches

    def createXYSet(self):
        # Open file to read data
        with open( self.svd_name, 'r') as file:
            
            # X and Y list to fill
            X_train = np.empty((self.num_batches, self.batch_Size, self.input_Nodes))
            Y_train = np.empty((self.num_batches, self.batch_Size))
            
            # Read all lines from file
            content = file.readlines()

            # Loop through each batch/line
            for batchNum in range(self.num_batches):

                # Temp indexing variables
                valueLoc = 0
                personLoc = -1

                # Break if first character is '#'
                if content[batchNum][0] != "#":

                    # Split at every " " or "&" and save it into a list called X_train
                    split_line = re.split(r"[ &]", content[batchNum])

                    # Remove empty strings and \n characters from list
                    split_line = [t for t in split_line if len(t) > 0]
                    split_line.pop(-1)

                    # Loop through split line
                    for p in range(len(split_line)):

                        # Checks if value is a subject num or not
                        if split_line[p] in self.Classroom:

                            
                            index = self.Classroom.index(split_line[p]) + 1
                            # Increment indexing
                            valueLoc = 0
                            personLoc += 1

                            # Adds correct values to output vector without the s on the front
                            Y_train[batchNum][personLoc] = index

                        else:

                            # Adds correct value to input vector, sets value location 
                            X_train[batchNum][personLoc][valueLoc] = float(split_line[p])
                            valueLoc += 1

            line = re.split(r"[ &]", content[-1])
            num_examples = len(line) // (self.dims[0] * self.dims[1] + 1)
            # Remove empty strings and \n characters from list
            line = [t for t in line if len(t) > 0]

            X_test = np.empty((num_examples, (self.dims[0] * self.dims[1])))
            Y_test = np.empty(num_examples)

            for i in range(num_examples):
                for j in range(self.dims[0] * self.dims[1] + 1):
                    if j == 0:
                        Y_test[i] = self.Classroom.index(line[i * (self.dims[0] * self.dims[1] + 1)])+ 1
                    else:
                        X_test[i][j - 1] = float(line[i * (self.dims[0] * self.dims[1] + 1) + j])
        
        return X_train, Y_train, X_test, Y_test