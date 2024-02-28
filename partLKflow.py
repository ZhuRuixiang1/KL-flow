import numpy as np
import cv2
import math
import sys


def even_points(xxyy,num_points):
    x1 = xxyy[0]  # 左上角 x 坐标
    y1 = xxyy[1]  # 左上角 y 坐标
    x2 = xxyy[2]  # 右下角 x 坐标
    y2 = xxyy[3]  # 右下角 y 坐标

    image_width = abs(x2 - x1)    # 计算图像宽度
    image_height = abs(y2 - y1)   # 计算图像高度
    r = image_width/image_height
    num_points_vertically = int(math.sqrt(r*num_points))
    num_points_horizontally = int(math.sqrt(num_points/r))
    horizontal_spacing = int(abs(x2 - x1) / num_points_horizontally)  # 水平间距
    vertical_spacing = int(abs(y2 - y1) / num_points_vertically)      # 垂直间距
    points = []
    for i in range(num_points_vertically):
        for j in range(num_points_horizontally):
            point_x = x1 + horizontal_spacing * j     # 当前点的 x 坐标
            point_y = y1 + vertical_spacing * i
            points.append([[float(point_x),float(point_y)]])
    points = np.array(points).astype(np.float32)
    return points

def rand_points(xxyy,num_points):
    x1 = xxyy[0]  # 左上角 x 坐标
    y1 = xxyy[1]  # 左上角 y 坐标
    x2 = xxyy[2]  # 右下角 x 坐标
    y2 = xxyy[3]  # 右下角 y 坐标
    points = []
    for _ in range(num_points):
        point_x = np.random.uniform(x1, x2)
        point_y = np.random.uniform(y1, y2)
        points.append([[float(point_x),float(point_y)]])
    point = np.array(points).astype(np.float32)
    return point

if __name__ == "__main__":
    #xxyy = [700,10,880,800]
    xxyy = [1100,300,1450,780]
    num_points = 50
    cap = cv2.VideoCapture(r"D:\project\mudslide\data\1.avi")
    lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0,255,(1000,3))
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = rand_points(xxyy, num_points)
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        if frame is None: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)#计算新的一副图像中相应的特征点额位置
        good_new = p1[st==1]
        good_old = p0[st==1]

        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel() #ravel()函数用于降维并且不产生副本
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()