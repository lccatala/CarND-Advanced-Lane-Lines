import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sbs

test_calibration = True

def corners_unwarp(img, nx, ny, mtx, dist):
  # Use the OpenCV undistort() function to remove distortion
  undist = cv2.undistort(img, mtx, dist, None, mtx)
  # Convert undistorted image to grayscale
  gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
  # Search for corners in the grayscaled image
  ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

  if ret == True:
    # If we found corners, draw them! (just for fun)
    cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # My choice of 100 pixels is not exact, but close enough for our purpose here
    offset = 100 # offset for dst points
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                  [img_size[0]-offset, img_size[1]-offset], 
                                  [offset, img_size[1]-offset]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

  # Return the resulting image and matrix
  return warped, M

if __name__ == '__main__':
  print('Calibrating camera...')
  nx = 9
  ny = 6

  # Prepare object points like (0,0,0), (1,0,0), (2,0,0)... (7,5,0)
  objp = np.zeros((nx * ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x and y coordinates
  objpoints = [] # 3D points in real-world space
  imgpoints = [] # 2D points in image plane

  calibration_filenames = os.listdir('camera_cal')
  for name in calibration_filenames:
    img = mpimg.imread('camera_cal/' + name)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret:
      imgpoints.append(corners)
      objpoints.append(objp)

      img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

  print('Camera calibration done!')

  if test_calibration:
    print('Testing camera calibration...')
    for name in os.listdir('camera_cal'):
      img = mpimg.imread('camera_cal/' + name)
      undist = cv2.undistort(img, mtx, dist, None, mtx)
      mpimg.imsave('undistorted_images/' + name, undist)
      # sbs.set_style("dark")
      # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
      # ax1.imshow(img)
      # ax1.set_title('Original Image')
      # ax2.imshow(undist)
      # ax2.set_title('Undistorted Image')
      # f.savefig("undistorted_images/" + name)
    print('Camera calibration testing done')