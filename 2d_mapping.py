import numpy as np
import cv2

def transform_2d():
	#TODO: find out how to get the
	camera_pts = []
	fl_plan_pts = []

	#Calculate H matrix
	H,stat = cv2.findHomography(camera_pts,fl_plan_pts)

	#pt is the point to 2D map
	pt = np.array(pt_from_camera, dtype='float32')
	pt = np.array([pt])

	#get the mapping array
	cv2.perspectiveTransform(pt,H)
