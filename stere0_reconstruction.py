import cv2
import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()



imgL=cv2.imread('bike_l.png')
imgR=cv2.imread('bike_r.png')
#cv2.destroyAllWindows()

#print(image.shape)
doffs=124.343
b=193.001
f=3979.911
window_size2=5
stereo =  cv2.StereoSGBM_create(minDisparity = 23, numDisparities = 222, blockSize = 5, uniquenessRatio = 5, speckleWindowSize = 5, speckleRange = 5, disp12MaxDiff = 0, P1 = 8*3*window_size2, P2 = 32*3*window_size2)
disp_map = stereo.compute(imgL,imgR).astype(np.float32)/240
disparity = cv2.pyrDown(disp_map)

image = cv2.pyrDown(cv2.imread('bike_l.png'))
distortion=np.zeros((5,1))
k=np.array([[3979.911, 0, 1244.772] ,[0, 3979.911 ,1019.507], [0 ,0 ,1]])
T=np.zeros((3,1))
T[0][0]=b
# R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(k,distortion,k,distortion,(2964,2000),np.identity(3),T)
Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, -2000/2],[0, 0, 0, f],[0, 0, -1/b, doffs/b]])
points = cv2.reprojectImageTo3D(disparity, Q)


mask = disparity > disparity.min()
xp=points[:,:,0]
yp=points[:,:,1]
zp=points[:,:,2]
xp=xp[mask]
yp=yp[mask]
zp=zp[mask]
xp=xp.flatten().reshape(-1,1)

yp=yp.flatten().reshape(-1,1)

zp=zp.flatten().reshape(-1,1)
point3d=np.hstack((xp,yp,zp))

#print(mask[0])
colors = image
#cv2.imshow('colors', colors)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

points = points
colors = image

points = points.reshape(-1,3)
colors = colors.reshape(-1,3)
points = np.hstack([points, colors])

ply_header = '''ply
	format ascii 1.0
	element vertex %(vert_num)d
	property float x
	property float y
	property float z
	property uchar blue
	property uchar green
	property uchar red
	end_header
	'''
with open('output.ply', 'w') as f:
	f.write(ply_header %dict(vert_num = len(points)))
	np.savetxt(f, points, '%f %f %f %d %d %d')

print("DONE")

cv2.imshow('window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()