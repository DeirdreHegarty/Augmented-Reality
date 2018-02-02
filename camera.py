""" Example of using OpenCV API to detect and draw checkerboard pattern"""
import numpy as np
import cv2

# These two imports are for the signal handler
import signal
import sys
import calibcam # calibrate the camera

#### Some helper functions #####
def reallyDestroyWindow(windowName) :
    ''' Bug in OpenCV's destroyWindow method, so... '''
    ''' This fix is from http://stackoverflow.com/questions/6116564/ '''
    cv2.destroyWindow(windowName)
    for i in range (1,5):
        cv2.waitKey(1) 

def shutdown():
        ''' Call to shutdown camera and windows '''
        global cap
        cap.release()
        reallyDestroyWindow('img')

def signal_handler(signal, frame):
        ''' Signal handler for handling ctrl-c '''
        shutdown()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
##########

############## calibration of plane to plane 3x3 projection matrix 

def compute_homography(fp,tp):
    ''' Compute homography that takes fp to tp. 
    fp and tp should be (N,3) '''

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # create matrix for linear method, 2 rows for each correspondence pair
    num_corners = fp.shape[0]

    # construct constraint matrix
    A = np.zeros((num_corners*2,9)); 
    A[0::2,0:3] = fp
    A[1::2,3:6] = fp
    A[0::2,6:9] = fp * -np.repeat(np.expand_dims(tp[:,0],axis=1),3,axis=1)
    A[1::2,6:9] = fp * -np.repeat(np.expand_dims(tp[:,1],axis=1),3,axis=1)

    # solve using *naive* eigenvalue approach
    D,V = np.linalg.eig(A.transpose().dot(A))

    H = V[:,np.argmin(D)].reshape((3,3))
    
    # normalise and return
    return H

##############


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# YOU SHOULD SET THESE VALUES TO REFLECT THE SETUP
# OF YOUR CHECKERBOARD
HEIGHT = 6
WIDTH = 9
# CAMWIDTH = 1280
# CAMHEIGHT = 720
CAMWIDTH = calibcam.w
CAMHEIGHT = calibcam.h

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)

cap = cv2.VideoCapture(0)

## Step 0: Load the image you wish to overlay
im_src = cv2.imread('doge.jpg')
# im_src = cv2.imread('car.png')
pattern = cv2.imread('pattern.png')

ret,patterncorners = cv2.findChessboardCorners(pattern, (HEIGHT,WIDTH),None)

# linspace returns evenly spaced numbers over a specified interval
# a.k.a. splitting up my image to be projeced
wi = np.linspace(0, CAMWIDTH, WIDTH) # width of picture to project
hi = np.linspace(0, CAMHEIGHT, HEIGHT) # height of picture to project

# print(wi) # default to 50 intervals ( 0.  17.20408163   34.40816327   51.6122449 ...)
# print(hi)

# empty list
imgmat = []

# Now that I have wi and hi, create the XY coordinates for projected image
for i in wi:
    for j in hi:
        imgmat.append([i,j])

imgmat = np.matrix(imgmat)
# print(imgmat.shape) # X Y coordinates of linear space (54,2)
homoimg = np.concatenate((imgmat, np.ones(WIDTH*HEIGHT)[:,None]), axis=1)
# print(homoimg.shape) #(54,3)

while (True):
        #capture a frame
        ret, img = cap.read()

        ## IF YOU WISH TO UNDISTORT YOUR IMAGE YOU SHOULD DO IT HERE
        # calibcam.py is imported at the top
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH),None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            # check of specified criteria above is met
            # cornerSubPix refines the corner locations
            # (11,11) specifies half of side length of search window
            # (-1,-1) means no dead region in middle of search zone (chessboard)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # reproject corners (diagonal lines)
            # corners = 2d coordinates

            # (Draw lines)
            # cv2.drawChessboardCorners(img, (HEIGHT,WIDTH), corners,ret)
            # print(corners.shape) #(54, 1, 2)
            


            ## STEP 1A : Compute fp -- an Nx3 array of the 2D homogeneous coordinates of the
            ## detected checkerboard corners
            
            # SOLUTION:
            # convert the 2d coordinates for the corners to matrix
            cornermatrix = np.matrix(corners)
            # print(cornermatrix.shape) # shape = (54,2)

            # now make these coordinates homogeneous
            # WIDTH & HEIGHT are defined above

            homocorners = np.concatenate((cornermatrix, np.ones(WIDTH*HEIGHT)[:,None]), axis=1)
            # print(homocorners) # shape = (54,3)



            ## STEP 1B: Compute tp -- an Nx3 array of the 2D homogeneous coordinates of the
            ## samples of the image coordinates
            ## Note: this could be done outside of the loop either!

            # THIS IS DONE OUTSIDE THE LOOP



            ## STEP 2: Compute the homography from tp to fp
            # h, status = cv2.findHomography(homocorners,homoimg)
            hi, status = cv2.findHomography(patterncorners, homocorners)
            # print(h)


            # USE CALIBCAM TO UNDISTORT CAMERA FEED
            # print calibcam.mtx, calibcam.dist
            cv2.undistort(img, calibcam.mtx, calibcam.dist, 0, calibcam.newcameramtx)


            ## STEP 3: Compute warped mask image
            im_dst = cv2.warpPerspective(im_src, hi, (CAMWIDTH,CAMHEIGHT))
            # cv2.imshow('im_dst',im_dst) 



            ## STEP 4: Compute warped overlay image
            # creating the black and white "mask"
            gray = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY) #turn image grey
            ret,thresh1 = cv2.threshold(gray,1,255,cv2.THRESH_BINARY) #threshold 
            # cv2.imshow('thresh1',thresh1) 
            thresh1 = cv2.bitwise_not(thresh1) # swaps black and white



            # HAD AN ERROR: IMAGE WASNT MAPPING AS THE RIGHT SIZE. 
            # NEEDED TO FIND HOMOGRAPHY BETWEEN IMAGE OF CHESSBOARD
            # AND THE CAMERA FEED


            ## STEP 5: Compute final image by combining the warped frame with the captured frame
            # the black and white image is going to be an array of 1 or 255 

            # where 0 is completely black and 1 is completely white

            # ERROR (fix by changing back to RGB)
            #  (-209) The operation is neither 'array op array' (where arrays have the same size and type), 
            #  nor 'array op scalar', nor 'scalar op array' in function binary_op
            thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB) # grey > colour
            img = cv2.bitwise_and(img,thresh1)
            img = cv2.add(img,im_dst) # current error 
            # cv2.imshow('img',img)




        cv2.imshow('img',img)      
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# release everything
shutdown()