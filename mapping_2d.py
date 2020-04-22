import numpy as np
import cv2
import operator
from tqdm import tqdm

def coord_extraction (img):
    coord = []
	
	# Tracks each double click on the image
    def click(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDBLCLK:
            coord.append([x,y])
	
	# Open the image
    image = cv2.imread(img)
    height= image.shape[0]
    width = image.shape[1]
	# Scale the image so that it fits most monitors
    resize_factor = 1920/width if 1920/width < 1080/height else 1080/height
    dwidth = int(width*resize_factor)
    dheight= int(height*resize_factor)
    dim = (dwidth,dheight)
    resized = cv2.resize(image,dim)

	# Display the scaled image on screen
    clone = resized.copy()
    winname = 'Press ESC to exit; Press R to reload; Double Click to Select Reference Points'
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname,click)
	
	# Extract points
    while True:
        cv2.imshow(winname,resized)
        #cv2.imshow(winname,image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            image = clone.copy()
            coord= []
        elif key == 27:
            break

    cv2.destroyAllWindows()

	# Rescale the points and format for output
    coord = np.asarray(coord)
    coord = coord / resize_factor
    pts = []
    for i in coord:
        pts.append(np.around(i))
    pts = np.asarray(pts)

    return pts

def transform_2d(frame_data):
	# Camera and floor plan hard coded points
	camera_pts = coord_extraction('camera_reference.jpg')
	fl_plan_pts = coord_extraction('floor_plan.jpg')

	# Open camera and floor plan
	cam_img = cv2.imread('00000.jpg',cv2.IMREAD_COLOR)
	fl_plan_img = cv2.imread('floor_plan.jpg',cv2.IMREAD_COLOR)
	h,w,l = fl_plan_img.shape
	size = (w,h)

	# Calculate H matrix
	H,stat = cv2.findHomography(camera_pts,fl_plan_pts)

	c = 1
	img_array = []
	pts_2d = []

	for frame in tqdm(frame_data):
		pts_frame = []
		for data in frame:
			# pt is the point to 2D map
			pt = np.array([[data[0],data[1]]], dtype='float32')
			pt = np.array([pt])

			# get the mapping array
			req_map = cv2.perspectiveTransform(pt,H)
			circle_rad = 20
			circ_col = (255,0,0)
			thickness = -1

			mapp = req_map[0]
			pts_frame.append([int(mapp[0][0]),int(mapp[0][1]),data[2]])
			# Draw the points on camera and floor plan
			for cord in mapp:
				center = (int(cord[0]),int(cord[1]))
				fl_plan_img = cv2.circle(fl_plan_img,center,circle_rad,circ_col,thickness)
				fl_plan_img = cv2.putText(fl_plan_img,str(data[2]),(int(cord[0]),int(cord[1])),cv2.FONT_HERSHEY_SIMPLEX,5,color=(0, 255, 0),thickness=2)
	
		pts_2d.append(pts_frame)
		img_array.append(fl_plan_img)
		fl_plan_img = cv2.imread('floor_plan.jpg',cv2.IMREAD_COLOR)
		c=c+1
	video = cv2.VideoWriter('2d_proj_vid.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)
	
	for i in tqdm(img_array):
		video.write(i)
	video.release()

	return pts_2d

def heatmap_gen (pts_2d, interval = 1):
	
	# Determine total number of people
	count = max([len(frame) for frame in pts_2d]) - 1

	# Read the floor plan image
	fl_plan_img = cv2.imread('floor_plan.jpg',cv2.IMREAD_COLOR)

	coord_dict = {}
	color = []

	# Create a dictionary with each id as keys and coordinates as values
	for i in range (count):
		coord_dict[i] = []

	# Generate color for each id
	for i in range (count):
		r = np.random.randint(low = 0, high = 255, size = 1)
		g = np.random.randint(low = 0, high = 255, size = 1)
		b = np.random.randint(low = 0, high = 255, size = 1)
		color.append((int(r[0]),int(b[0]),int(g[0])))
	
	# Extract frame using given interval and from each frame extract points for each id
	for i in range (0, len(pts_2d), interval):
		for pt in pts_2d[i]:
			coord_dict[pt[2]].append([pt[0],pt[1]])

	# Connect the dots
	for i in range (count):
		coord_list = coord_dict[i]
		for j in range (1,len(coord_list)):
			cv2.line(fl_plan_img, tuple(coord_list[j-1]), tuple(coord_list[j]),color[i],5)
	
	x = 70
	y = 950
	for i in range(count):
		cv2.putText(fl_plan_img, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 5, color[i],3)
		y+=3

	cv2.imwrite('heatmap.png',fl_plan_img)

def main():
	transform_2d()

if __name__=="__main__":
	main()
