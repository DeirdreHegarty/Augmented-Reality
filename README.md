#### What does it do?

With one simple command the camera opens and takes in a camera feed. The
participant holds up a checkerboard facing the camera and an image is projected
onto the board in real time.  In order for this to work correctly the camera is
calibrated to cater for barrel distortion.

#### How does it work?

The first step is to calibrate the camera. In order to do this I took a series
of photographs of myself holding the checkerboard in front of the webcam. The
program then finds the corners of each checkerboard corner in all the
photographs and stores them in an array. The same is done for the real time
image feed of the checkerboard being held up in front of the camera. The camera
matrix is computed and then undistorted.

The next step was to capture images from the camera and find checkerboard
corners in the image. The homography between the checkerboard pattern and the
checkerboard in the undistorted image (from the previous step) is computed. A
mask is created containing the desired image to be projected into the real
world. The mask is thresholded and combined with the image feed.
