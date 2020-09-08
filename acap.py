import cv2
import numpy as np

import time
import threading

class acap:
    def __init__(self, source):
        try:
            self.s = int(source)
        except:
            self.s = source
        
        self.cap = cv2.VideoCapture(self.s)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.__run = True
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.__thread = threading.Thread(target=self.__thread_read).start()

    def read(self):
        return True, self.image
    
    def __thread_read(self):
        while self.__run:
            ret, img = self.cap.read()

            if ret:
                self.image = img
    
    def __del__(self):
        self.__run = False

cap = acap(0)

while True:
    ret, img = cap.read()
    if ret:
        cv2.imshow('test', img)

        if cv2.waitKey(1) == 113:
            del cap
            break
