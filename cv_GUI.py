import cv2
import numpy as np
import camera
import sys

### camera setting ###
# camera type
TYPE = "PI"
# actived camera
cam_ON = None
cam_setting = {"width":1280,"height":720,"frame_rate":23,"shutter":0,"iso":800}
# stereo calibration matrix by OpenCV
Cam_coefficient_CV = '/home/pi/Desktop/PyCharm/Cam_coefficients.npz'
# stereo calibration matrix by MATLAB
Cam_coefficient_MATLAB = '/home/pi/Desktop/PyCharm/Cam_coefficients_MATLAB.npz'
# camera loaded status
Cam_loaded = {"Loaded":0,"Type":0}

### depth map coefficients ###
BM_setting = {"MinDisparity":0,"BlockSize":5,"NumDisparities":1,"UniquenessRatio":0}
SGBM_setting = {"MinDisparity":0,"BlockSize":5,"NumDisparities":1,"UniquenessRatio":0}

### other setting coefficients ###
# display image scale
SCALE = 0.5
# kernel size for morphology process
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
# captured image pair saving path
save_folder = '/home/pi/Desktop/PyCharm/Save/'
img_set = 1

### GUI setting ###
# function switch
size_img = np.zeros((3,350,3), np.uint8)
cv2.namedWindow("depth")
cv2.moveWindow("depth", 1500, 50)

cv2.createTrackbar("Open","switch",0, 1, lambda x: None)
Cali = 'Calibration \n0 : CV \n1 : MATLAB'
cv2.createTrackbar(Cali,"switch",0, 1, lambda x: None)
# 0: BM, 1: SGBM
Algo = 'Depth \n0 : BM \n1 : SGBM'
cv2.createTrackbar(Algo,"switch",0, 1, lambda x: None)
cv2.createTrackbar("Depth","switch",0, 1, lambda x: None)
cv2.createTrackbar("Threshold","switch",0,255,lambda x: None)
cv2.createTrackbar("Dist","switch",0, 1, lambda x: None)
cal_dist = 'Distancce \n0 : near \n1 : far'
cv2.createTrackbar(cal_dist,'switch',0, 1, lambda x: None)
obj_size = 'Object size \n0 : small \n100 : large'
cv2.createTrackbar(obj_size,"switch",0,100,lambda x: None)

# disparity coefficients
size_img2 = np.zeros((3,350,3), np.uint8)
cv2.namedWindow("switch")
cv2.moveWindow("switch", 1300, 50)

cv2.createTrackbar("min", "depth", 0, 10, lambda x: None)
cv2.createTrackbar("num", "depth", 1, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
cv2.createTrackbar("UniRatio", "depth", 0, 20, lambda x: None)


# calcualte real world distance from disparity
def cal_3d_dist(depth,threeD,object_size):
    result_set =[]
    (_, cnts, _) = cv2.findContours(depth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts):
        max_coutour = max(cnts, key=cv2.contourArea)
        max_area = cv2.contourArea(max_coutour)
        for c in cnts:
            if cv2.contourArea(c) >= (max_area*object_size)/100:
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
                # for a 30mm cell size chessboard
                dist_z = dist_xyz[2]*3

                result_set.append([box,dist_z,(avg_x,avg_y)])

    return result_set


# main loop
while(True):
    cv2.imshow("switch", size_img)
    cv2.imshow("depth", size_img2)
    cv2.waitKey(50)

    # read main switch
    open_proccess = cv2.getTrackbarPos("Open","switch")
    if open_proccess == 1:
        # choose which calibraton file to use
        cali_file = cv2.getTrackbarPos(Cali, "switch")
        if Cam_loaded["Loaded"] == 1 and Cam_loaded["Type"] == cali_file:
            # if calibration matrix already loaded
            able_to_cal = True
        else:
            print("Loading calibration matrix...")
            if cali_file == 0:
                # loading OpenCV matrix
                try:
                    calibration = np.load(Cam_coefficient_CV, allow_pickle=False)
                    able_to_cal = True
                    Cam_loaded["Loaded"] = 1
                    Cam_loaded["Type"] = 0

                    print("Loading camera coefficients from cache file at {0}".format(Cam_coefficient_CV))
                    imageSize = tuple(calibration["imageSize"])
                    leftMapX = calibration["leftMapX"]
                    leftMapY = calibration["leftMapY"]
                    # leftROI = tuple(calibration["leftROI"])
                    rightMapX = calibration["rightMapX"]
                    rightMapY = calibration["rightMapY"]
                    # rightROI = tuple(calibration["rightROI"])
                    Q = calibration["dispartityToDepthMap"]
                except IOError:
                    print("OpenCV cache file at {0} not found".format(Cam_coefficient_CV))
                    print("Exit with error")
                    able_to_cal = False

            elif cali_file == 1:
                # loading MATLAB matrix
                try:
                    calibration = np.load(Cam_coefficient_MATLAB, allow_pickle=False)
                    able_to_cal = True
                    Cam_loaded["Loaded"] = 1
                    Cam_loaded["Type"] = 1

                    print("Loading camera coefficients from cache file at {0}".format(Cam_coefficient_MATLAB))
                    imageSize = calibration["imageSize"]
                    leftMapX = calibration["leftMapX"]
                    leftMapY = calibration["leftMapY"]
                    # leftROI = tuple(calibration["leftROI"])
                    rightMapX = calibration["rightMapX"]
                    rightMapY = calibration["rightMapY"]
                    # rightROI = tuple(calibration["rightROI"])
                    Q = calibration["dispartityToDepthMap"]
                except IOError:
                    print("MATLAB cache file at {0} not found".format(Cam_coefficient_MATLAB))
                    print("Exit with error")
                    able_to_cal = False

        # if matrix loaded, display captured image
        if able_to_cal != True:
            print("Unable to load calibration matrix, terminated")
            sys.exit(1)
        else:
            print ("Start capture images...")
            cam_EYE = 'L'
            cam_ON = camera.Open(TYPE, cam_EYE)
            cam_ON = camera.Setup(cam_ON, cam_setting, TYPE)
            # capture image
            img_L = camera.Capture(cam_ON, TYPE)
            # img_L = cv2.flip(img_L, -1)
            gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
            cam_ON.close()

            cam_EYE = 'R'
            cam_ON = camera.Open(TYPE, cam_EYE)
            cam_ON = camera.Setup(cam_ON, cam_setting, TYPE)
            # capture image
            # time.sleep(0.1)
            img_R = camera.Capture(cam_ON, TYPE)
            # img_R = cv2.flip(img_L, -1)
            gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
            cam_ON.close()

            # obtain rectified image pair
            fixedLeft = cv2.remap(gray_L, leftMapX, leftMapY, cv2.INTER_LINEAR)
            fixedRight = cv2.remap(gray_R, rightMapX, rightMapY, cv2.INTER_LINEAR)

            # display rectified images
            height, width = fixedLeft.shape[:2]
            fixedLeft_s = cv2.resize(fixedLeft, (int(SCALE * width), int(SCALE * height)), interpolation=cv2.INTER_CUBIC)
            fixedRight_s = cv2.resize(fixedRight, (int(SCALE * width), int(SCALE * height)), interpolation=cv2.INTER_CUBIC)
            cv2.namedWindow("Left_fixed")
            cv2.namedWindow("Right_fixed")
            cv2.moveWindow("Left_fixed", 0, 10)
            cv2.moveWindow("Right_fixed", 650, 10)
            # draw standard line
            cv2.line(fixedLeft_s, (0, int(SCALE * height * 0.5)), (int(SCALE * width), int(SCALE * height * 0.5)), 255,
                     1)
            cv2.line(fixedLeft_s, (0, int(SCALE * height * 0.3)), (int(SCALE * width), int(SCALE * height * 0.3)), 255,
                     1)
            cv2.line(fixedLeft_s, (0, int(SCALE * height * 0.7)), (int(SCALE * width), int(SCALE * height * 0.7)), 255,
                     1)

            cv2.line(fixedRight_s, (0, int(SCALE * height * 0.5)), (int(SCALE * width), int(SCALE * height * 0.5)), 255,
                     1)
            cv2.line(fixedRight_s, (0, int(SCALE * height * 0.3)), (int(SCALE * width), int(SCALE * height * 0.3)), 255,
                     1)
            cv2.line(fixedRight_s, (0, int(SCALE * height * 0.7)), (int(SCALE * width), int(SCALE * height * 0.7)), 255,
                     1)
            cv2.imshow("Left_fixed", fixedLeft_s)
            cv2.imshow("Right_fixed", fixedRight_s)
            key = cv2.waitKey(50)
            if key == 27:
                cv2.destroyAllWindows()
                break

            # read "calculate disparity" switch
            open_depth = cv2.getTrackbarPos("Depth","switch")
            if open_depth == 1:
                print("Start obtain disparity...")
                # choose disparity calculation algorithm
                algorithm = cv2.getTrackbarPos(Algo,"switch")
                if algorithm ==0:
                    # BM
                    print("Select Block Matching to obtain disparity... ")
                    stereoMatcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                    # set stereo matcher coefficients
                    BM_setting['MinDisparity'] = cv2.getTrackbarPos("min", "depth")
                    num = cv2.getTrackbarPos("num", "depth")
                    if num == 0:
                        BM_setting['NumDisparities'] = 1
                    else:
                        BM_setting['NumDisparities'] = num
                    # block size can only be odd number
                    b_size = cv2.getTrackbarPos("blockSize", "depth")
                    if not (b_size % 2) == 0:
                        BM_setting['BlockSize'] = b_size
                    else:
                        BM_setting['BlockSize'] = b_size + 1
                    BM_setting['UniquenessRatio'] = cv2.getTrackbarPos("UniRatio", "depth")

                    stereoMatcher.setMinDisparity(BM_setting['MinDisparity'])
                    stereoMatcher.setNumDisparities(BM_setting['NumDisparities'] * 16)
                    stereoMatcher.setBlockSize(BM_setting['BlockSize'])
                    stereoMatcher.setUniquenessRatio(BM_setting['UniquenessRatio'])
                elif algorithm ==1:
                    # SGBM
                    print("Select Semi Global Block Matching to obtain disparity... ")
                    stereoMatcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                    # set stereo matcher coefficients
                    SGBM_setting['MinDisparity'] = cv2.getTrackbarPos("min", "depth")
                    num = cv2.getTrackbarPos("num", "depth")
                    if num == 0:
                        SGBM_setting['NumDisparities'] = 1
                    else:
                        SGBM_setting['NumDisparities'] = num
                    b_size = cv2.getTrackbarPos("blockSize", "depth")
                    if not (b_size % 2) == 0:
                        SGBM_setting['BlockSize'] = b_size
                    else:
                        SGBM_setting['BlockSize'] = b_size + 1
                    SGBM_setting['UniquenessRatio'] = cv2.getTrackbarPos("UniRatio", "depth")

                    stereoMatcher.setMinDisparity(BM_setting['MinDisparity'])
                    stereoMatcher.setNumDisparities(SGBM_setting['NumDisparities'] * 16)
                    stereoMatcher.setBlockSize(SGBM_setting['BlockSize'])
                    stereoMatcher.setUniquenessRatio(BM_setting['UniquenessRatio'])

                # calculate depth map
                depth = stereoMatcher.compute(fixedLeft, fixedRight)
                # normalise for display
                depth_no = cv2.normalize(depth, depth, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_8U)
                # do close process to remove gaps in depth map
                depth_no = cv2.morphologyEx(depth_no, cv2.MORPH_CLOSE, kernel)

                # get threshold number and far-near mode
                thres = cv2.getTrackbarPos("Threshold","switch")
                mode = cv2.getTrackbarPos(cal_dist,'switch')
                if mode == 0:
                    _, depth_tho = cv2.threshold(depth_no, thres, 255, cv2.THRESH_TOZERO)
                elif mode ==1:
                    _, depth_tho = cv2.threshold(depth_no, thres, 255, cv2.THRESH_TOZERO_INV)

                # read "calculate distance" switch
                open_dist = cv2.getTrackbarPos("Dist","switch")
                if open_dist == 1:
                    print("Calculate 3D distancce...")
                    # estimate real world distance
                    threeD = cv2.reprojectImageTo3D(depth.astype(np.float32) / 16., Q)
                    object_size = cv2.getTrackbarPos(obj_size,"switch")
                    dis_set = cal_3d_dist(depth_tho, threeD,object_size)
                    # plot estimated distance on img
                    for item in dis_set:
                        cv2.drawContours(depth_tho, [item[0]], -1, 255, 2)
                        cv2.putText(depth_tho, "%.2fcm" % item[1],
                                    (item[2][0], item[2][1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, 150, 3)

                # dispalay depth map image
                height, width = depth_no.shape[:2]
                depth_no_s = cv2.resize(depth_no, (int(SCALE * width), int(SCALE * height)), interpolation=cv2.INTER_CUBIC)
                depth_tho_s = cv2.resize(depth_tho, (int(SCALE * width), int(SCALE * height)), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("disparity", depth_no_s)
                cv2.moveWindow("disparity",0,400)
                cv2.imshow("depth_threshold", depth_tho_s)
                cv2.moveWindow("depth_threshold", 650, 400)
                key = cv2.waitKey(50)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key == 13:
                    # if press "enter" then save current image pairs
                    L_save = save_folder +"L_fixed_{:02d}.jpg"
                    R_save = save_folder + "R_fixed_{:02d}.jpg"
                    dep_save = save_folder + "Depth_{:02d}.jpg"
                    dist_save = save_folder + "Dist_{:02d}.jpg"
                    cv2.imwrite(L_save.format(img_set), fixedLeft)
                    cv2.imwrite(R_save.format(img_set), fixedRight)
                    cv2.imwrite(dep_save.format(img_set), depth_no)
                    cv2.imwrite(dist_save.format(img_set), depth_tho)
                    img_set += 1
                    print("Image pair saved")


