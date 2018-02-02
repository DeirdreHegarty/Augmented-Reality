import numpy as np
import cv2
import glob


def takeCalibrationPictures():
    ''' Take pictures to calibrate camera'''
    camera = cv2.VideoCapture(0)
    i=0
    while i < 14:
        key = cv2.waitKey(500) & 0xFF # wait until hit a key
        retval, im = camera.read()
        cv2.imshow('im',im)
        if key == ord(' '):
            string="testimages/string"+str(i)
            file =   string + ".png"
            cv2.imwrite(file, im)
            i += 1

def calcMeanError():
    ''' calculate the mean error '''
    mean_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print "total error: ", mean_error/len(objpoints)


# uncomment to take new photos
# takeCalibrationPictures()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 9
HEIGHT = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:WIDTH,0:HEIGHT].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# read in from testimages -> calibration pictures should be here

# images = glob.glob('borrowedimages/*.jpg')
# images = glob.glob('test/*.jpg')
# images = glob.glob('images/*.jpg')
images = glob.glob('testimages/*.png')

for fname in images:
    print fname
    img = cv2.imread(fname)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (WIDTH,HEIGHT),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        
        # change from 
        # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # and replace all corners2 with corners
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (WIDTH,HEIGHT), corners,ret)

        # CAN COMMENT IN NEXT THREE LINES (show calibration images)
        # cv2.imshow('img',img)
        # cv2.imwrite('img.png',img) 
        # cv2.waitKey(500)

# calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# print np.array(imgpoints).shape
# print np.array(objpoints).shape

# undistortion step 
# img = cv2.imread('borrowedimages/image-004.jpg')
# img = cv2.imread('test/25360602_1744054325645796_1742832554_n.jpg')
# img = cv2.imread('images/1.jpg')
img = cv2.imread('testimages/string1.png')
h,  w= img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


# undistort
dst = cv2.undistort(img, mtx, dist, 0, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

print "region of interest: ", roi

calcMeanError()


cv2.destroyAllWindows()