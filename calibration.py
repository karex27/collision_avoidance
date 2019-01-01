import numpy as np
import cv2
import camera
import time
import sys
import glob
import os
import random

cam_setting = {"width":1280,"height":720,"frame_rate":23,"shutter":0,"iso":800}
# capture imgs or do calibration
# "shoot" or "calibrate"
status = sys.argv[1]
# maximum samples numbers used for calculation
MAX_IMAGES = int(sys.argv[2])
# save path for left and right image pair
L_path = '/home/pi/Desktop/PyCharm/L_cali'
R_path = '/home/pi/Desktop/PyCharm/R_cali'


# calibrate single camera
def Find_chessboard_single(img_path):
    CHESSBOARD_SIZE = (9, 6)
    OBJECT_POINT_ZERO = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
                                 np.float32)
    OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                               0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)


    TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
                            0.001)

    filenames = []
    objectPoints = []
    imagePoints = []
    imageSize = None
    cacheFile = "{0}/chessboards.npz".format(img_path)
    try:
        cache = np.load(cacheFile)
        print("Loading image data from cache file at {0}".format(cacheFile))
        return (list(cache["filenames"]), list(cache["objectPoints"]),
                list(cache["imagePoints"]), tuple(cache["imageSize"]))
    except IOError:
        print("Cache file at {0} not found".format(cacheFile))


    print("Reading images at {0}".format(img_path))
    imagePaths = glob.glob("{0}/*.jpg".format(img_path))

    for imagePath in sorted(imagePaths):
        image = cv2.imread(imagePath)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        newSize = grayImage.shape[::-1]
        if imageSize != None and newSize != imageSize:
            raise ValueError(
                "Calibration image at {0} is not the same size as the others"
                    .format(imagePath))
        imageSize = newSize

        hasCorners, corners = cv2.findChessboardCorners(grayImage,
                                                        CHESSBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)

        if hasCorners:
            filenames.append(os.path.basename(imagePath))
            objectPoints.append(OBJECT_POINT_ZERO)
            cv2.cornerSubPix(grayImage, corners, (11, 11), (-1, -1),
                             TERMINATION_CRITERIA)
            imagePoints.append(corners)

        cv2.drawChessboardCorners(image, CHESSBOARD_SIZE, corners, hasCorners)
        cv2.imshow(img_path, image)
        cv2.waitKey(100)

    cv2.destroyWindow(img_path)

    print("Found corners in {0} out of {1} images"
            .format(len(imagePoints), len(imagePaths)))


    np.savez_compressed(cacheFile,
            filenames=filenames, objectPoints=objectPoints,
            imagePoints=imagePoints, imageSize=imageSize)

    return filenames, objectPoints, imagePoints, imageSize


# extract require obj, img points
def Get_require_points(requestedFilenames,allFilenames, objectPoints, imagePoints):

    requestedFilenameSet = set(requestedFilenames)
    requestedObjectPoints = []
    requestedImagePoints = []

    for index, filename in enumerate(allFilenames):
        if filename in requestedFilenameSet:
            requestedObjectPoints.append(objectPoints[index])
            requestedImagePoints.append(imagePoints[index])

    return requestedObjectPoints, requestedImagePoints


# test single camera calibration
def Test_calibration(mtx,dist,files):
    print("Reading images at {0}".format(files))
    imagePaths = glob.glob("{0}/*.jpg".format(files))

    for filename in sorted(imagePaths):
        img = cv2.imread(filename)
        # refine camera matrix
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        cv2.imshow("original",img)
        cv2.imshow('calibresult', dst)

        cv2.waitKey(50)

    cv2.destroyAllWindows()


def main():
    if status == "shoot":
        frameId = 1
        while True:
            cam_EYE = 'L'
            cam_ON = camera.Open("PI", cam_EYE)
            cam_ON = camera.Setup(cam_ON, cam_setting, "PI")
            # capture image
            # time.sleep(0.1)
            img_L = camera.Capture(cam_ON, "PI")
            # img_L = cv2.flip(img_L, -1)
            cam_ON.close()

            cam_EYE = 'R'
            cam_ON = camera.Open("PI", cam_EYE)
            cam_ON = camera.Setup(cam_ON, cam_setting, "PI")
            # capture image
            # time.sleep(0.1)
            img_R = camera.Capture(cam_ON, "PI")
            # img_R = cv2.flip(img_R, -1)
            cam_ON.close()

            gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
            gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

            height, width = gray_L.shape[:2]
            gray_L_s = cv2.resize(gray_L,(int(0.5 * width),int(0.5 * height)),interpolation=cv2.INTER_CUBIC)
            gray_R_s = cv2.resize(gray_R, (int(0.5 * width),int(0.5 * height)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('L_cam', gray_L_s)
            cv2.imshow('R_cam', gray_R_s)
            cv2.moveWindow("L_cam", 100, 400)
            cv2.moveWindow("R_cam",800,400)
            key = cv2.waitKey(50)
            if key == 27:
                break
            elif key == 13:
                # press enter
                L_save = L_path + '/{:04d}.jpg'
                R_save = R_path + '/{:04d}.jpg'
                cv2.imwrite(L_save.format(frameId), gray_L)
                cv2.imwrite(R_save.format(frameId), gray_R)
                print("Image pair {0} saved.".format(frameId))
                frameId += 1

    elif status == "calibrate":
        # refer: https://albertarmea.com/post/opencv-stereo-camera/

        TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        OPTIMIZE_ALPHA = 0

        outputFile = '/home/pi/Desktop/PyCharm/Cam_coefficients.npz'
        leftCamCali = '/home/pi/Desktop/PyCharm/L_Cam_Matrix.npz'
        rightCamCali = '/home/pi/Desktop/PyCharm/R_Cam_Matrix.npz'

        (leftFilenames, leftObjectPoints, leftImagePoints, leftSize) = Find_chessboard_single(L_path)
        (rightFilenames, rightObjectPoints, rightImagePoints, rightSize ) = Find_chessboard_single(R_path)

        if leftSize != rightSize:
            print("Camera resolutions do not match")
            sys.exit(1)
        imageSize = leftSize

        print("Choose {0} images to do calibration".format(MAX_IMAGES))
        filenames = list(set(leftFilenames) & set(rightFilenames))
        if (len(filenames) > MAX_IMAGES):
            print("Too many images to calibrate, using {0} randomly selected images"
                  .format(MAX_IMAGES))
            filenames = random.sample(filenames, MAX_IMAGES)
        filenames = sorted(filenames)
        print("Using these images:")
        print(filenames)

        leftObjectPoints, leftImagePoints = Get_require_points(filenames,
                leftFilenames, leftObjectPoints, leftImagePoints)
        rightObjectPoints, rightImagePoints = Get_require_points(filenames,
                rightFilenames, rightObjectPoints, rightImagePoints)

        # objectPoints = leftObjectPoints
        objectPoints = rightObjectPoints

        try:
            cache_L = np.load(leftCamCali)
            print("Loading left calibration data from cache file at {0}".format(leftCamCali))
            cache_R = np.load(rightCamCali)
            print("Loading left calibration data from cache file at {0}".format(rightCamCali))

            leftCameraMatrix = cache_L["leftCameraMatrix"]
            leftDistortionCoefficients = cache_L["leftDistortionCoefficients"]
            rightCameraMatrix = cache_R["rightCameraMatrix"]
            rightDistortionCoefficients = cache_R["rightDistortionCoefficients"]
        except IOError:
            print("Cache file at {0} not found".format(leftCamCali))
            print("Cache file at {0} not found".format(rightCamCali))

            print("Calibrating left camera...")
            _, leftCameraMatrix, leftDistortionCoefficients, _, _ = cv2.calibrateCamera(
                objectPoints, leftImagePoints, imageSize, None, None)
            print("Caching left camera matrix... ")
            np.savez_compressed(leftCamCali, leftCameraMatrix=leftCameraMatrix,
                                leftDistortionCoefficients=leftDistortionCoefficients)

            print("Calibrating right camera...")
            _, rightCameraMatrix, rightDistortionCoefficients, _, _ = cv2.calibrateCamera(
                objectPoints, rightImagePoints, imageSize, None, None)
            print("Caching right camera matrix... ")
            np.savez_compressed(rightCamCali, rightCameraMatrix=rightCameraMatrix,
                                rightDistortionCoefficients=rightDistortionCoefficients)


        # print("Calibration test... ")
        # Test_calibration(leftCameraMatrix, leftDistortionCoefficients, L_path)
        # Test_calibration(rightCameraMatrix, rightDistortionCoefficients, R_path)

        print("Calibrating cameras together...")
        (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
            objectPoints, leftImagePoints, rightImagePoints,
            leftCameraMatrix, leftDistortionCoefficients,
            rightCameraMatrix, rightDistortionCoefficients,
            imageSize, None, None, None, None,
            cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)

        print("Rectifying cameras...")
        (leftRectification, rightRectification, leftProjection, rightProjection,
         dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            leftCameraMatrix, leftDistortionCoefficients,
            rightCameraMatrix, rightDistortionCoefficients,
            imageSize, rotationMatrix, translationVector,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

        print("Saving calibration...")
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            leftCameraMatrix, leftDistortionCoefficients, leftRectification,
            leftProjection, imageSize, cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            rightCameraMatrix, rightDistortionCoefficients, rightRectification,
            rightProjection, imageSize, cv2.CV_32FC1)

        np.savez_compressed(outputFile, imageSize=imageSize,
                            leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
                            rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI,
                            dispartityToDepthMap=dispartityToDepthMap)

        cv2.destroyAllWindows()

main()