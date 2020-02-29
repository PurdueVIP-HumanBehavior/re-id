import cv2
import operator
import numpy as np

image = cv2.imread('/Users/ray/Desktop/OneDrive - purdue.edu/VIP/test_MSEE_atrium.jpg')
width = image.shape[1]
height= image.shape[0]
if width < 200 or height < 200:
    resize_factor = 10
    dwidth = int(width*resize_factor)
    dheight= int(height*resize_factor)
    dim = (dwidth,dheight)
    resized = cv2.resize(image,dim)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
else:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([35, 140, 60])
upper_blue = np.array([255, 255, 180])
mask = cv2.inRange(hsv, lower_blue, upper_blue)  
coord=cv2.findNonZero(mask)

if width < 200 or height < 200:
    res = cv2.bitwise_and(resized,resized, mask= mask)
else:
    res = cv2.bitwise_and(image,image, mask= mask)
#cv2.imshow('res',res)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

x_coordinate = np.array([])
y_coordinate = np.array([])


for pt in coord:
    x,y = np.split(pt[0],2)
    x_coordinate = np.append(x_coordinate,x[0])
    y_coordinate = np.append(y_coordinate,y[0])

coord_list = []
for i in range (0,x_coordinate.size):
    coord_list.append((x_coordinate[i],y_coordinate[i]))

coord_list.sort(key = operator.itemgetter(0))

x_coordinate = []
y_coordinate = []
for pt in coord_list:
    x,y = pt
    x_coordinate.append(x)
    y_coordinate.append(y)

x_coord = [[]]
x_coord[0].append(x_coordinate[0])
axis_value = 0

for i in range(1,len(x_coordinate)):
    if np.abs(x_coordinate[i] - x_coordinate[i-1]) >= 100:
        x_coord.append([])
        axis_value += 1
    x_coord[axis_value].append(x_coordinate[i])

y_coord = [[]]
y_coord[0].append(y_coordinate[0])
axis_value = 0
incr = 1

for j in range(1,len(y_coordinate)):
    if incr == len(x_coord[axis_value]):
        y_coord.append([])
        axis_value += 1
        incr = 0
    y_coord[axis_value].append(y_coordinate[j])
    incr += 1

pt_x = []
pt_y = []


for x in x_coord:
    if width < 200 or height < 200:
        mean = np.mean(x)
        mean = int(mean/10)
    else:
        mean = int(np.mean(x))
    pt_x.append(mean)

for y in y_coord:
    if width < 200 or height < 200:
        mean = np.mean(y)
        mean = int(mean/10)
    else:
        mean = int(np.mean(y))
    pt_y.append(mean)


pts = np.stack((pt_x, pt_y), axis=-1)

print(pts)

#print(image.shape)
#print(resized.shape)