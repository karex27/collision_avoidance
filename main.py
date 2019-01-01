import cv2
import numpy as np
import Tkinter
import camera
import time
import sys



# intial setting
root = Tkinter.Tk()
TYPE = "PI"
cam_setting = {"width":1280,"height":720,"frame_rate":23,"shutter":0,"iso":800}
depth_setting = {"MinDisparity":0,"BlockSize":5,"NumDisparities":1,"UniquenessRatio":0}

# the current active camera
cam_ON = None
cam_EYE = 'L'
# initialize the known distance from the camera to the object, which
KNOWN_DISTANCE = 39.5
# initialize the known object width, which in this case, the piece of
KNOWN_WIDTH = 8.5

focalLength = 0.0


# simple_distancce
def Smp_det():
    global focalLength
    global cam_ON
    # choose one camera for single detection
    cam_ON = camera.Open(TYPE,cam_EYE)
    cam_ON = camera.Setup(cam_ON,cam_setting,TYPE)
    while 1:
        # capture image
        time.sleep(0.1)
        cap_img = camera.Capture(cam_ON,TYPE)
        # cap_img = cv2.flip(cap_img,-1)
        # cv2.imshow('cap_img', cap_img)
        # key = cv2.waitKey(0)

        # find biggest contour
        gray = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        cv2.imshow('canny', edged)
        key = cv2.waitKey(20)

        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts):
            c = max(cnts, key=cv2.contourArea)
            marker = cv2.minAreaRect(c)

            if focalLength == 0.0:
                focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
                # cv2.imshow("initial_frame", cap_img)
                # cv2.waitKey(0)
            else:
                distance = (KNOWN_WIDTH * focalLength) / marker[1][0]

                # draw a bounding box around the image and display it
                box = np.int0(cv2.boxPoints(marker))
                cv2.drawContours(cap_img, [box], -1, (0, 255, 0), 2)
                cv2.putText(cap_img, "%.2fcm" % distance,
                            (20, cap_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (0, 255, 0), 3)
                cv2.imshow("distance_image", cap_img)
                key = cv2.waitKey(20)
                if key ==27:
                    cv2.destroyAllWindows()
                    cam_ON.close()
                    break


# set camera coefficients
def Cam_set():
    global cam_ON
    cam_setting['height'] =sb1.get()
    cam_setting['width'] = sb2.get()

    if not cam_ON == None:
        cam_ON = camera.Setup(cam_ON, cam_setting, TYPE)
    else:
        cam_ = camera.Open(TYPE, "L")
        cam_ = camera.Setup(cam_, cam_setting, TYPE)
        cam_.close()

        cam_ = camera.Open(TYPE, "R")
        cam_ = camera.Setup(cam_, cam_setting, TYPE)
        cam_.close()


def cal_3d_dist(depth,threeD):
    result_set =[]
    (_, cnts, _) = cv2.findContours(depth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts):
        max_coutour = max(cnts, key=cv2.contourArea)
        max_area = cv2.contourArea(max_coutour)
        for c in cnts:
            if cv2.contourArea(c) >= max_area/2:
                marker = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(marker))
                # coordinates of the AreaRect corners
                object_x = []
                object_y = []
                for point in box:
                    object_x.append(point[0])
                    object_y.append(point[1])
                object_x.sort()
                object_y.sort()
                # coordinate of the area cnetre
                avg_x = reduce(lambda x, y: x + y, object_x) / len(object_x)
                avg_y = reduce(lambda x, y: x + y, object_y) / len(object_y)
                # dist_z is the actual real world distancce
                dist_xyz = threeD[avg_y][avg_x]
                dist_z = dist_xyz[2]

                result_set.append([box,dist_z,(avg_x,avg_y)])

    return result_set

# calculate depth map
def Depth_cal(img_L,img_R,leftROI,rightROI):

    stereoMatcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # # set stereo matcher coefficients
    depth_setting['MinDisparity'] = sb3.get()
    depth_setting['NumDisparities'] = sb4.get()
    # block size can only be odd number
    b_size = sb5.get()
    if not (b_size % 2) == 0:
        depth_setting['BlockSize'] = b_size
    else:
        depth_setting['BlockSize'] = b_size + 1
    depth_setting['UniquenessRatio'] = sb6.get()

    # depth_setting['NumDisparities'] = cv2.getTrackbarPos("NumDisparities", "depth")
    # blockSize = cv2.getTrackbarPos("BlockSize", "depth")
    # if blockSize % 2 == 0:
    #     depth_setting['BlockSize'] = blockSize + 1
    # if blockSize < 5:
    #     depth_setting['BlockSize'] = 5
    # depth_setting['MinDisparity'] = cv2.getTrackbarPos("MinDisparity", "depth")
    # depth_setting['UniquenessRatio'] = cv2.getTrackbarPos("UniquenessRatio", "depth")

    stereoMatcher.setMinDisparity(depth_setting['MinDisparity'])
    stereoMatcher.setNumDisparities(depth_setting['NumDisparities']*16)
    stereoMatcher.setBlockSize(depth_setting['BlockSize'])
    stereoMatcher.setUniquenessRatio(depth_setting['UniquenessRatio'])
    stereoMatcher.setROI1(leftROI)
    stereoMatcher.setROI2(rightROI)

    if not (img_L.ndim == 2 | img_R.ndim == 2):
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

    # img_L = cv2.equalizeHist(img_L)
    # img_R = cv2.equalizeHist(img_R)

    depth = stereoMatcher.compute(img_L, img_R)
    # depth = stereoMatcher.compute(img_L, img_R)

    # Normalize the image for representation
    depth_no = cv2.normalize(depth, depth, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _,depth_tho = cv2.threshold(depth_no,200,255,cv2.THRESH_TOZERO_INV)
    # min = depth.min()
    # max = depth.max()
    # depth_no = np.uint8(255 * (depth - min) / (max - min))

    return depth, depth_no, depth_tho


# capture picture and calculate depth map
def Depth_map():
    global cam_ON, cam_EYE
    Cam_coefficient = '/home/pi/Desktop/PyCharm/Cam_coefficients.npz'

    try:
        calibration = np.load(Cam_coefficient, allow_pickle=False)
        flag = True
        print("Loading camera coefficients from cache file at {0}".format(Cam_coefficient))
        imageSize = tuple(calibration["imageSize"])
        leftMapX = calibration["leftMapX"]
        leftMapY = calibration["leftMapY"]
        leftROI = tuple(calibration["leftROI"])
        rightMapX = calibration["rightMapX"]
        rightMapY = calibration["rightMapY"]
        rightROI = tuple(calibration["rightROI"])
        Q = calibration["dispartityToDepthMap"]
    except IOError:
        print("Cache file at {0} not found".format(Cam_coefficient))
        flag = False

    if flag:
        cam_EYE = 'L'
        cam_ON = camera.Open("PI",cam_EYE)
        cam_ON = camera.Setup(cam_ON, cam_setting, TYPE)
        # capture image
        img_L = camera.Capture(cam_ON, TYPE)
        # img_L = cv2.flip(img_L, -1)
        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        cam_ON.close()

        cam_EYE = 'R'
        cam_ON = camera.Open("PI", cam_EYE)
        cam_ON = camera.Setup(cam_ON, cam_setting, TYPE)
        # capture image
        # time.sleep(0.1)
        img_R = camera.Capture(cam_ON, TYPE)
        # img_R = cv2.flip(img_L, -1)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
        cam_ON.close()

        fixedLeft = cv2.remap(gray_L, leftMapX, leftMapY, cv2.INTER_LINEAR)
        cv2.imshow('fixedLeft', fixedLeft)
        # cv2.imwrite('/home/pi/Desktop/PyCharm/fixedLeft.jpg', fixedLeft)
        fixedRight = cv2.remap(gray_R, rightMapX, rightMapY, cv2.INTER_LINEAR)
        cv2.imshow('fixedRight', fixedRight)
        # cv2.imwrite('/home/pi/Desktop/PyCharm/fixedRight.jpg', fixedRight)

        # depth: original depth map, used for calculate 3D distance
        # depth_no: normalised depth map, used to display
        # depth_tho: normalised depth map with threhold applied, used to optimise
        depth, depth_no,depth_tho = Depth_cal(fixedLeft, fixedRight,leftROI,rightROI)
        # calculate the real world distance
        threeD = cv2.reprojectImageTo3D(depth.astype(np.float32) / 16., Q)
        dis_set = cal_3d_dist(depth_tho,threeD)

        for item in dis_set:
            cv2.drawContours(depth_tho, [item[0]], -1, 255, 2)
            cv2.putText(depth_tho, "%.2fcm" % item[1],
                        (item[2][0], item[2][1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        2, 255, 3)

        cv2.imshow("depth",depth_no)
        cv2.imshow("depth_threshold", depth_tho)
        key = cv2.waitKey(-1)
        if key ==27:
            cv2.destroyAllWindows()



# choose which single camera is select for simple detect
def Cam_select():
    global cam_EYE
    if choose_L.get():
        cam_EYE = 'L'
    elif choose_R.get():
        cam_EYE = 'R'
    else:
        cam_EYE = 'L'


# select left camera
choose_L = Tkinter.IntVar()
c1 = Tkinter.Checkbutton(root, text='Left_cam', variable=choose_L, onvalue=1, offvalue=0,command=Cam_select)
c1.pack(padx=5, pady=10, side=Tkinter.LEFT)
# select right camera
choose_R = Tkinter.IntVar()
c2 = Tkinter.Checkbutton(root, text='Right_cam', variable=choose_R, onvalue=1, offvalue=0,command=Cam_select)
c2.pack(padx=5, pady=10, side=Tkinter.LEFT)
# start simple_distance
btn1 = Tkinter.Button(root,width=10,padx=20,text='simple_detect',anchor='c',borderwidth=10,relief='ridge',compound='bottom',command=Smp_det)
btn1.pack(padx=5, pady=10, side=Tkinter.LEFT)

# show depth map
btn3 = Tkinter.Button(root,width=10,padx=20,text='Depth_map',anchor='c',borderwidth=10,relief='ridge',compound='bottom',command=Depth_map)
btn3.pack()

# change camera height
sb1 = Tkinter.Scale(root, from_=0, to=1080,tickinterval=200,label='height',length=400,orient=Tkinter.HORIZONTAL)
sb1.pack()
# change camera width
sb2 = Tkinter.Scale(root, from_=0, to=1920,tickinterval=200,label='width',length=400,orient=Tkinter.HORIZONTAL)
sb2.pack()
# apply changes to camera
btn2 = Tkinter.Button(root,width=10,padx=20,text='set_camera',anchor='c',borderwidth=10,relief='ridge',compound='bottom',command=Cam_set)
btn2.pack()

 # change MinDisparity
sb3 = Tkinter.Scale(root, from_=0, to=10,tickinterval=2,label='MinDisparity',length=400,orient=Tkinter.HORIZONTAL)
sb3.pack()
# change NumDisparities
sb4 = Tkinter.Scale(root, from_=1, to=10,tickinterval=2,label='NumDisparities',length=400,orient=Tkinter.HORIZONTAL)
sb4.pack()
# change BlockSize
sb5 = Tkinter.Scale(root, from_=5, to=255,tickinterval=20,label='BlockSize',length=400,orient=Tkinter.HORIZONTAL)
sb5.pack()
# change UniquenessRatio
sb6 = Tkinter.Scale(root, from_=0, to=20,tickinterval=2,label='UniquenessRatio',length=400,orient=Tkinter.HORIZONTAL)
sb6.pack()


# cv2.createTrackbar("NumDisparities", "depth", 0, 10, lambda x: None)
# cv2.createTrackbar("BlockSize", "depth", 5, 255, lambda x: None)
# cv2.createTrackbar("MinDisparity", "depth", 0, 10, lambda x: None)
# cv2.createTrackbar("UniquenessRatio", "depth", 0, 20, lambda x: None)


# read default settings
sb1.set(cam_setting['height'])
sb2.set(cam_setting['width'] )
sb3.set(depth_setting['MinDisparity'] )
sb4.set(depth_setting['NumDisparities'] )
sb5.set(depth_setting['BlockSize'])
sb6.set(depth_setting['UniquenessRatio'] )


def on_closing():
    if not cam_ON == None:
        cam_ON.close()
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_closing)


Tkinter.mainloop()
