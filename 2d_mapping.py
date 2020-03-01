import numpy as np
import cv2
import displayer

def transform_2d():
	#TODO: find out how to get the points
	camera_pts = np.array([[314,320],[650,868],[1264,448],[1544,326]])
	fl_plan_pts = np.array([[20,62],[81,144],[142,71],[181,60]])

	#Open camera and floor plan
	cam_img = cv2.imread('00000.jpg',cv2.IMREAD_COLOR)
	fl_plan_img = cv2.imread('msee_atrium.png',cv2.IMREAD_COLOR)

	#Calculate H matrix
	H,stat = cv2.findHomography(camera_pts,fl_plan_pts)

	#pt is the point to 2D map
	pt = np.array(camera_pts, dtype='float32')
	pt = np.array([pt])

	#get the mapping array
	req_map = cv2.perspectiveTransform(pt,H)
	circle_rad = 5
	circ_col = (255,0,0)
	thickness = -1

	mapp = req_map[0]
	print(displayer.req_point_list)

	#Draw the points on camera and floor plan
	for cord in mapp:
		center = (int(cord[0]),int(cord[1]))
		fl_plan_img = cv2.circle(fl_plan_img,center,circle_rad,circ_col,thickness)

	for pt in camera_pts:
		center = (int(pt[0]),int(pt[1]))
		cam_img = cv2.circle(cam_img,center,circle_rad,circ_col,thickness)
	cv2.imshow("fl_plan_img",fl_plan_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	transform_2d()

if __name__=="__main__":
	main()
