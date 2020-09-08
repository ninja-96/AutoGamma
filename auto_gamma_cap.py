import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0, maxx=255):
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return table[image]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while(True):
    ret, frame = cap.read()
    if ret:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_ag = l.copy()
        # a_ag = a.copy()
        # b_ag = b.copy()

        x = np.histogram([l_ag], list(range(256)))[0]
        y = range(len(x))
        xx = np.poly1d(np.polyfit(y, x, 1))(y)

        div = xx[0] - xx[-1]
        cor = -1 / (1 + 2.71**(-div/500)) + 1.5

        l_ag = adjust_gamma(l_ag, cor)
        # a_ag = adjust_gamma(a_ag, cor)
        # b_ag = adjust_gamma(b_ag, cor)
        
        img_ag = cv2.merge((l_ag, a, b))
        img_ag = cv2.cvtColor(img_ag, cv2.COLOR_LAB2BGR)

        cv2.imshow('frame', frame)
        cv2.imshow('img_ag', img_ag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()