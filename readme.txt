camera.py -- camera initialise and all related functions
calibration.py -- shoot image pairs and use them to do stereo calibration
cv_GUI.py -- main program, GUIs to control all functions

Cam_coefficients.npz -- stereo matrix calculated using OpenCV
Cam_coefficients_MATLAB.npz -- stereo matrix calculated using MATLAB (not working well)

---------------------------------
If want to use current stereo matrix for all the functions, run cv_GUI.py directly.
If want to recalibrate the camera, run calibration.py with "shoot" mode first, then run it in "calibrate" mode.