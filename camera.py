import cv2
import picamera
from picamera.array import PiRGBArray
import RPi.GPIO as gp

# gpio initialise
def GPIO_Cam_Sel():
    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    # layer 1
    gp.setup(7, gp.OUT)
    gp.setup(11, gp.OUT)
    gp.setup(12, gp.OUT)
    # layer 2
    gp.setup(15, gp.OUT)
    gp.setup(16, gp.OUT)
    # layer 3
    gp.setup(21, gp.OUT)
    gp.setup(22, gp.OUT)

    gp.output(11, True)
    gp.output(12, True)
    gp.output(15, True)
    gp.output(16, True)
    gp.output(21, True)
    gp.output(22, True)


# camera selection based on GPIO
def Cam_Select(type="L"):
    GPIO_Cam_Sel()
    if type == "L":
        # camera C
        gp.output(7, False)
        gp.output(11, True)
        gp.output(12, False)
    elif type =="R":
        # camera A
        gp.output(7, False)
        gp.output(11, False)
        gp.output(12, True)
    else:
        return False


# open camera
def Open(type="CV",eye="L"):
    if type == "CV":
        cam = cv2.VideoCapture(0)
        return cam
    elif type == "PI":
        Cam_Select(eye)
        cam = picamera.PiCamera()
        return cam
    else:
        return False


# write camera setting
def Setup(cam,cam_setting,type="CV"):
    if type == "CV":
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_setting['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_setting['height'])
        return cam
    elif type == "PI":
        cam.resolution = (cam_setting['width'], cam_setting['height'])
        cam.iso = cam_setting["iso"]
        return cam
    else:
        return False


# capture single frame image
def Capture(cam,type="CV"):
    if type == "CV":
        ret, img = cam.read()
        return img
    elif type == "PI":
        rawCapture = PiRGBArray(cam)
        cam.capture(rawCapture, format="bgr")
        img = rawCapture.array
        return img
    else:
        return False
