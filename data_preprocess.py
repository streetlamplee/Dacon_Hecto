import cv2
import numpy as np
import os

class preprocess():
    def __init__(self, path):
        self.path = path
        self.class_list = os.listdir(path)
        self.class_list.sort()
        self.data_class_pair = []
        self.preprocess()

    def preprocess(self):
        for idx, c in enumerate(self.class_list):
            iter_file_list = os.listdir(os.path.join(self.path, c))
            for iter_file in iter_file_list:
                self.data_class_pair.append([os.path.join(os.path.join(self.path, c),iter_file), idx, len(self.class_list)])


    def get_data(self):
        return self.data_class_pair