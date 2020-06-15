import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

class Line():
  def __init__ (self):
    self.detected = False
    self.recent_xfitted = []
    self.bestx = None
    self.best_fit = None
    self.current_fit = [np.array([False])]
    self.radius_of_curvature = None
    self.line_base_pos = None
    self.diffs = np.array([0, 0, 0], dtype='float')
    self.allx = None
    self.ally = None

def gaussian_blur(img, kernel_size=3):
  blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
  return blurred

def warp_perspective(img, src_points, dst_points):
  src_points = np.float32(src_points)
  dst_points = np.float32(dst_points)
  img_size = (img.shape[1], img.shape[0])

  M = cv2.getPerspectiveTransform(src_points, dst_points)
  warped_image = cv2.warpPerspective(img, M, img_size)
  return warped_image, M

def calibrate_camera():
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

  return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def color_threshold(img, thresh_min=170, thresh_max=255):
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2]

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  # Sobel x
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
  abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
  scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

  # Threshold x gradient
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

  # Threshold color channel
  s_thresh_min = 170
  s_thresh_max = 255
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

  color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

  # Combine the two binary thresholds
  combined_binary = np.zeros_like(sxbinary)
  combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

  return combined_binary

def hsv_threshold(img, thresh_min=np.array([0,0,0]), thresh_max=np.array([255,255,255])):
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  mask = cv2.inRange(hsv, thresh_min, thresh_max)

  v_channel = hsv[:,:,2]
  binary_output = np.zeros_like(v_channel)
  binary_output[(mask > 0)] = 1
  return binary_output

def sobel(img, sobel_size=3, sobel_x=0, sobel_y=0, threshold=[0,255]):
  sobel = cv2.Sobel(img, cv2.CV_64F, sobel_x, sobel_y)
  abs_sobel = np.absolute(sobel)
  scaled_sobel = np.uint(255 * abs_sobel / np.max(abs_sobel))
  binary_output = np.zeros_like(img)
  binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
  return binary_output

def sobel_threshold_ls(img, sobel_size=3, threshold=[0,255]):
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  l_channel = hls[:,:,1]
  s_channel = hls[:,:,2]

  sobel_s_x = sobel(s_channel, sobel_size, 1, 0, threshold)
  sobel_l_x = sobel(l_channel, sobel_size, 1, 0, threshold)

  binary_output = np.zeros_like(l_channel)
  binary_output[(sobel_s_x == 1) | (sobel_l_x == 1)] = 1
  return binary_output

def find_lines(img, left, right, nwindows=9, margin=100, minpix=100):
  window_height = np.int(img.shape[0] / nwindows)

  nonzero = img.nonzero()
  nonzero_x = np.array(nonzero[1])
  nonzero_y = np.array(nonzero[0])

  current_left = left
  current_right = right

  left_lane_inds = []
  right_lane_inds = []

  left_rects = []
  right_rects = []

  for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = img.shape[0] - (window+1)*window_height
    win_y_high = img.shape[0] - window*window_height

    win_xleft_low = current_left - margin
    win_xleft_high = current_left + margin
    win_xright_low = current_right - margin
    win_xright_high = current_right + margin

    # TEMP
    left_rects.append([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high)])
    right_rects.append([(win_xright_low, win_y_low), (win_xright_high, win_y_high)])

    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
    (nonzero_x >= win_xleft_low) &  (nonzero_x < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
    (nonzero_x >= win_xright_low) &  (nonzero_x < win_xright_high)).nonzero()[0]

    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # If we found > minpix pixels, recenter next window
    # (`right` or `leftx_current`) on their mean position
    if len(good_left_inds) > minpix:
      current_left = np.int(np.mean(nonzero_x[good_left_inds]))
    if len(good_right_inds) > minpix:
      current_right = np.int(np.mean(nonzero_x[good_right_inds]))

  # Concatenate the arrays of indices (previously was a list of lists of pixels)
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  # Extract left and right line pixel positions
  left_x = nonzero_x[left_lane_inds]
  left_y = nonzero_y[left_lane_inds] 
  right_x = nonzero_x[right_lane_inds]
  right_y = nonzero_y[right_lane_inds]

  return left_rects, right_rects, left_x, left_y, right_x, right_y

def find_ends(img):
  hist = np.sum(img[img.shape[0]//2:,:], axis=0)

  mid = np.int(hist.shape[0]/2)
  left = np.argmax(hist[:mid])
  right = np.argmax(hist[mid:]) + mid

  return left, right

def measure_curvature_radius(ploty, leftx, rightx, y_mpp=30/720, x_mpp=3.7/700):
  y_eval = np.max(ploty)

  left_fit_cr = np.polyfit(ploty*y_mpp, leftx*x_mpp, 2)
  right_fit_cr = np.polyfit(ploty*y_mpp, rightx*x_mpp, 2)

  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*y_mpp + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*y_mpp + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

  return left_curverad, right_curverad

def process_image(img):

  # Perspective transform
  src_points = [[200, 720], # Bottom left
                [570, 470], # Top left
                [720, 470], # Top right
                [1130, 720]] # Bottom right
  dst_points = [[320, 720], # Bottom left
                [320, 0], # Top left
                [980, 0], # Top right
                [980, 720]] # Bottom right
  undist = cv2.undistort(img, mtx, dist, None, mtx)
  warped_image, M = warp_perspective(undist, src_points, dst_points)

  # Color masking
  yellow_binary = hsv_threshold(warped_image, np.array([20,100,100]), np.array([30,255,255]))
  white_binary = hsv_threshold(warped_image, np.array([0,0,223]), np.array([255,32,255]))
  sobel_binary = sobel_threshold_ls(warped_image, 5, [50, 255])

  th_image = np.zeros_like(white_binary)
  th_image[(white_binary == 1) | (yellow_binary == 1) | (sobel_binary == 1)] = 1

  blurred = gaussian_blur(th_image, 25) # Blurring for reducing noise

  # Lane detection
  ploty = np.linspace(0, warped_image[0].shape[0]-1, warped_image[0].shape[0])
  left, right = find_ends(blurred)

  left_rects, right_rects, left_x, left_y, right_x, right_y = find_lines(blurred, left, right)

  left_line = Line()
  right_line = Line()

  if left_x.size:
    left_fit = np.polyfit(left_y, left_x, 2)
    left_line.bestx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    left_line.current_fit = left_fit
    left_line.allx = left_x
    left_line.ally = left_y

  if right_x.size:
    right_fit = np.polyfit(right_y, right_x, 2)
    right_line.bestx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    right_line.current_fit = right_fit
    right_line.allx = right_x
    right_line.ally = right_y

  # Draw pixels of each line in a different color
  lines_image = (np.dstack((blurred, blurred, blurred)) * 255).astype(np.uint8).copy()
  lines_image[left_line.ally, left_line.allx] = [255,0,0]
  lines_image[right_line.ally, right_line.allx] = [0,0,255]

  # Curvature
  left_line.radius_of_curvature, right_line.radius_of_curvature = measure_curvature_radius(ploty, left_line.bestx, right_line.bestx)
  curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
  curvature = np.round(curvature, 2)

  # Find deviation from center
  left_end = left_line.current_fit[0] * warped_image.shape[0] ** 2 + left_line.current_fit[1] * warped_image.shape[1] + left_line.current_fit[2]
  right_end = right_line.current_fit[0] * warped_image.shape[0] ** 2 + right_line.current_fit[1] * warped_image.shape[1] + right_line.current_fit[2]

  center = (left_end + right_end) / 2
  distance_to_center = (center - warped_image.shape[1] / 2) / 2
  distance_to_center = np.round(distance_to_center, 2)

  # Draw data onto unwarped image
  distance_message = 'Vehicle is ' + str(np.absolute(distance_to_center)) + 'cm '
  if distance_to_center > 0:
    distance_message += 'right of center'
  else:
    distance_message += 'left of center'
  
  curvature_message = 'Radius of Curvature = ' + str(curvature) + 'm'

  zeros_image = np.zeros_like(blurred).astype(np.uint8)
  blank_slate = np.dstack((zeros_image,zeros_image,zeros_image))

  left_points = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
  right_points = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
  points = np.hstack((left_points, right_points))

  # Paint area in between lanes green
  cv2.fillPoly(blank_slate, np.int_([points]), (0,255,0))

  # Apply inverted distortion to image
  dewarped_poly = cv2.warpPerspective(blank_slate, np.linalg.inv(M), (blurred.shape[1], blurred.shape[0]))
  dewarped_lines = cv2.warpPerspective(lines_image, np.linalg.inv(M), (blurred.shape[1], blurred.shape[0]))

  output = cv2.addWeighted(undist, 1, dewarped_poly, 0.3, 0)
  output = cv2.addWeighted(output, 1, dewarped_lines, 0.3, 0.3)

  cv2.putText(output, distance_message, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
  cv2.putText(output, curvature_message, (30, 120), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)

  return output

#######################
##### ENTRY POINT #####
#######################

if __name__ == '__main__':
  print('Calibrating camera...')
  ret, mtx, dist, rvecs, tvecs = calibrate_camera()
  print('Camera calibration done!')

  output_filename = 'project_video_lines.mp4'
  input_clip = VideoFileClip('project_video.mp4')
  output_clip = input_clip.fl_image(process_image)
  output_clip.write_videofile(output_filename, audio=False)