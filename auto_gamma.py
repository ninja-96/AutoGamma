import numpy as np
import cv2

import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -7)

def adjust_gamma(image, gamma=1.0, maxx=255):
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return table[image]
    # return cv2.LUT(image, table)

rec = []

while(True):
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_tmp = gray.copy()
        div = None
        cor = 1
        prev = None

        for i in range(500):
            s = time.time()
            x = np.histogram([gray_tmp], list(range(256)))[0]
            y = range(len(x))
            xx = np.poly1d(np.polyfit(y, x, 1))(y)

            rt = time.time() - s
            print('hist time', rt, 1/rt)
            # print(abs(xx[0] - xx[-1]), i % 3)

            if div is None:
                prev = abs(xx[0] - xx[-1])
                # div = abs(xx[0] - xx[-1])
                div = xx[0] - xx[-1]
            else:
                if i % 2 == 0:
                    if prev == abs(xx[0] - xx[-1]):
                        break
                    else:
                        prev = abs(xx[0] - xx[-1])

            if i in range(0, 250):
                s = 0.01
            elif i in range(250, 500):
                s = 0.001
            
            if xx[0] > xx[-1]:
                cor -= s

            if xx[0] < xx[-1]:
                cor += s
            
            gray_tmp = adjust_gamma(gray.copy(), cor)

            if cor < 0.5:
                cor = 0.5
                break

            if cor > 1.5:
                cor = 1.5
                break

        print(div, cor, i)
        cv2.imshow('gray_or', gray)
        cv2.imshow('gray', adjust_gamma(gray, cor))

        k = cv2.waitKey()
        if k == 113:
            break
        elif k == 13:
            print(f'{[div, cor]} added')
            rec.append([div, cor])

np.save('test.npy', np.array(rec))

cap.release()
cv2.destroyAllWindows()
