"""
Instructions for creating hyper-parameter object:
1. In your file, do from parameters import Parameters.
2. Call whatever parameter you want.
"""

import os
            
class Parameters ():

    def __init__ (self):
        self.batch_size = self.count_folders("../Students/Cropped_Students")
        self.epochs = 40
        self.rank = 23
        self.dims = (92, 112)
        self.outputNum = self.count_folders("../Students/Cropped_Students")
        self.layers = [1500, 500, self.outputNum]
        self.confirmationRate =.4   
        self.learningRate =.015   

    
    def count_folders(self,path):
        folder_count = 0
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                folder_count += 1
        return folder_count 