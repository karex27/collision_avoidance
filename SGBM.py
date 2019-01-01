import cv2
import numpy as np
import camera

# disparity settings
window_size = 5
min_disp = 64
num_disp = 256 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=num_disp,
                              blockSize=16,
                              P1=8 * 3 * window_size ** 2,
                              P2=32 * 3 * window_size ** 2,
                              disp12MaxDiff=1,
                              uniquenessRatio=10,
                              speckleWindowSize=100,
                              speckleRange=32
                              )
TYPE = "PI"
cam_setting = {"width":1280,"height":720,"frame_rate":23,"shutter":0,"iso":800}
# morphology settings
kernel = np.ones((12, 12), np.uint8)


while True:

    cam_EYE = 'L'
    cam_ON = camera.Open("PI", cam_EYE)
    cam_ON = camera.Setup(cam_ON, cam_setting, TYPE)
    # capture image
    image_left = camera.Capture(cam_ON, TYPE)
    # img_L = cv2.flip(img_L, -1)
    # gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    cam_ON.close()

    cam_EYE = 'R'
    cam_ON = camera.Open("PI", cam_EYE)
    cam_ON = camera.Setup(cam_ON, cam_setting, TYPE)
    # capture image
    # time.sleep(0.1)
    image_right = camera.Capture(cam_ON, TYPE)
    # img_R = cv2.flip(img_L, -1)
    # gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
    cam_ON.close()

    # compute disparity
    disparity = stereo.compute(image_left, image_right).astype(np.float32) / 16.0
    disparity = (disparity - min_disp) / num_disp

    # apply threshold
    threshold = cv2.threshold(disparity, 0.6, 1.0, cv2.THRESH_BINARY)[1]

    # apply morphological transformation
    morphology = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # show images
    cv2.imshow('left eye', image_left)
    cv2.imshow('right eye', image_right)
    cv2.imshow('disparity', disparity)
    # cv2.imshow('threshold', threshold)
    # cv2.imshow('morphology', morphology)
    key = cv2.waitKey(2000)
    if key == 27:
        cv2.destroyAllWindows()
        break