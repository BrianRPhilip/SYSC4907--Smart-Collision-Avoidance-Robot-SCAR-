import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
from PIL import Image
import math
from collections import deque
import time
import timeit
import socket
import io
import struct
#import pickle

serverAddressPort = ("192.169.2.32", 8080)
bufferSize = 1024

# Create a UDP socket at client side
#UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

time_start = time.time()

def ServerToPiSendCMD(cmd):
  bytesToSend = str.encode(cmd)
  # Send to server using created UDP socket
  UDPClientSocket.sendto(bytesToSend, serverAddressPort)
  print("sent")

def PiToServerReceiveIMG():
  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server_socket.bind(('192.168.2.36', 8000))  # ADD IP HERE
  server_socket.listen(5)

  #file= connection.makefile('rb')
  try:
    img = None
    while True:
      # Read the length of the image as a 32-bit unsigned int. If the
      # length is zero, quit the loop
      # Accept a single connection and make a file-like object out of it
      conn, addr = server_socket.accept()
      connection = conn[0].makefile('rb')

      #data = server_socket.recv(4096000)
      #file.write(data)
      image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
      if not image_len:
        break
      # Construct a stream to hold the image data and read the image
      # data from the connection
      image_stream = io.BytesIO()
      image_stream.write(connection.read(image_len))
      # Rewind the stream, open it as an image with PIL and do some
      # processing on it
      #image_stream.seek(0)
      image = Image.open(image_stream)
      imarr = np.array(image)
      height, width, c = imarr.shape
      #im = Image.fromarray(imarr)
      print("here --------------> ", type(imarr), width, height, c)
      #run processing algo
      command = main_func(imarr)
      print(addr)
      toSend = command.encode()
      conn.send(toSend)
      print('Image is %dx%d' % image.size)
      image.verify()
      print('Image is verified')
  finally:
    connection.close()
    server_socket.close()

def main_func(image):
  IMAGE_H = 480
  IMAGE_W = 360
  img = image
  print("here")
  #plt.imshow(img, cmap='gray')
  #plt.show()
  src = np.float32([[0, 100], [500, 100], [500, 300], [0, 300]])
  dst = np.array([[0, img.shape[0]], [0, 0], [500, 0], [500, img.shape[0]]], np.float32)
  def perspTransform(src, dst):
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    return (M, Minv)

  (M, Minv) = perspTransform(src, dst)

  result = cv2.warpPerspective(img, M,(500,480), flags=cv2.INTER_LINEAR)
  rotated = imutils.rotate(result, 270)

  cropped_image = rotated[0:480, 20:480]
  img_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

  lower_blue = np.array([10,0,80]) #change values here. not good
  upper_blue = np.array([70,255,255])

  #lower_blue = np.array([130,0,0]) #change values here. not good
  #upper_blue = np.array([180,255,255])

  mask = cv2.inRange(cropped_image, lower_blue, upper_blue)
  res = cv2.bitwise_and(cropped_image, cropped_image, mask = mask)

  crop_mask = mask[0:400, 0:480]
  histogram = np.sum(crop_mask[crop_mask.shape[0]//2:, :], axis=0)


  def create_queue(length=10):
    return deque(maxlen=length)


  class LaneLine:
    def __init__(self):
      self.polynomial_coeff = None
      self.line_fit_x = None
      self.non_zero_x = []
      self.non_zero_y = []
      self.windows = []


  class LaneLineHistory:
    def __init__(self, queue_depth=2, test_points=[50, 300, 500, 700], poly_max_deviation_distance=150):
      self.lane_lines = create_queue(queue_depth)
      self.smoothed_poly = None
      self.test_points = test_points
      self.poly_max_deviation_distance = poly_max_deviation_distance

    def append(self, lane_line, force=False):
      if len(self.lane_lines) == 0 or force:
        self.lane_lines.append(lane_line)
        self.get_smoothed_polynomial()
        return True

      test_y_smooth = np.asarray(list(
        map(lambda x: self.smoothed_poly[0] * x ** 2 + self.smoothed_poly[1] * x + self.smoothed_poly[2],
            self.test_points)))
      test_y_new = np.asarray(list(map(
        lambda x: lane_line.polynomial_coeff[0] * x ** 2 + lane_line.polynomial_coeff[1] * x + lane_line.polynomial_coeff[
          2], self.test_points)))

      dist = np.absolute(test_y_smooth - test_y_new)
      max_dist = dist[np.argmax(dist)]

      if max_dist > self.poly_max_deviation_distance:
        print("**** MAX DISTANCE BREACHED, STOPPING NOW ****")
        print("y_smooth={0} - y_new={1} - distance={2} - max-distance={3}".format(test_y_smooth, test_y_new, max_dist,
                                                                                  self.poly_max_deviation_distance))
        return False

      self.lane_lines.append(lane_line)
      self.get_smoothed_polynomial()

      return True

    def get_smoothed_polynomial(self):
      all_coeffs = np.asarray(list(map(lambda lane_line: lane_line.polynomial_coeff, self.lane_lines)))
      self.smoothed_poly = np.mean(all_coeffs, axis=0)

      return self.smoothed_poly

  class LineDetector:

    def __init__(self, objpts, imgpts, src, dst, sliding_windows_per_line, 
                 sliding_window_half_width, sliding_window_recenter_thres, 
                 small_img_size=(256, 144), small_img_x_offset=20, small_img_y_offset=10,
                 img_dimensions=(720, 1280), lane_width_px=800, 
                 lane_center_px_psp=600, real_world_lane_size_cm=(32, 27.5)):
        self.objpts = objpts
        self.imgpts = imgpts
        (self.M_psp, self.M_inv_psp) = perspTransform(src, dst)

        self.sliding_windows_per_line = sliding_windows_per_line
        self.sliding_window_half_width = sliding_window_half_width
        self.sliding_window_recenter_thres = sliding_window_recenter_thres

        self.small_img_size = small_img_size
        self.small_img_x_offset = small_img_x_offset
        self.small_img_y_offset = small_img_y_offset

        self.img_dimensions = img_dimensions
        self.lane_width_px = lane_width_px
        self.lane_center_px_psp = lane_center_px_psp 
        self.real_world_lane_size_cm = real_world_lane_size_cm

        # We can pre-compute some data here
        self.ym_per_px = self.real_world_lane_size_cm[0] / self.img_dimensions[0]
        self.xm_per_px = self.real_world_lane_size_cm[1] / self.lane_width_px
        self.ploty = np.linspace(0, self.img_dimensions[0] - 1, self.img_dimensions[0])

        self.previous_left_lane_line = None
        self.previous_right_lane_line = None

        self.previous_left_lane_lines = LaneLineHistory()
        self.previous_right_lane_lines = LaneLineHistory()

        self.total_img_count = 0  

    def processImage(self, image):
      ll, rl, mid_fit = self.compute_lane_lines(image)

      print(ll)#.line_fit_x, rl.line_fit_x)
      if ll == 0  and rl == 0:
        print("needs to turn")
        return mid_fit, ll.line_fit_x, rl. line_fit_x
      drawn_lines = self.draw_lane_lines(image, ll, rl)
      plt.imshow(image[image.shape[0]//2:,:], cmap = 'gray')
      plt.show()  
      plt.imshow(drawn_lines, cmap = 'gray')
      plt.show()

      ploty = np.linspace(0, image.shape[0] - 1, image.shape[0] )
      mid_fitx = (mid_fit[0] * ploty**2 + mid_fit[1] * ploty + mid_fit[2])
      pts_mid = np.dstack((mid_fitx, ploty)).astype(np.int32)
      cv2.circle(image, (230, 250), 10, (255,255,255), -1)
      i = cv2.polylines(image, pts_mid, False,  (255, 140,0), 5)
      plt.imshow(i, cmap = 'gray')
      plt.show()
      print(mid_fit)

      return mid_fit, ll.polynomial_coeff, rl.polynomial_coeff

    def draw_lane_lines(self, warped_img, left_line, right_line):
        """
        Returns an image where the computed lane lines have been drawn on top of the original warped binary image
        """
        # Create an output image with 3 colors (RGB) from the binary warped image to draw on and  visualize the result

        out_img = np.dstack((warped_img, warped_img, warped_img))*255

        # Now draw the lines
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        pts_left = np.dstack((left_line.line_fit_x, ploty)).astype(np.int32)
        pts_right = np.dstack((right_line.line_fit_x, ploty)).astype(np.int32)

        cv2.polylines(out_img, pts_left, False,  (255, 140,0), 5)
        cv2.polylines(out_img, pts_right, False, (255, 140,0), 5)

        if not (left_line.line_fit_x[0] == 0 and left_line.line_fit_x[1] == 0 and left_line.line_fit_x[2] == 0):
          for low_pt, high_pt in left_line.windows:
            cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)
        else:
          print("not plotting left windows")

        if not (right_line.line_fit_x[0] == 0 and right_line.line_fit_x[1] == 0 and right_line.line_fit_x[2] == 0):
          for low_pt, high_pt in right_line.windows:            
              cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)           
        else:
          print("not plotting right windows")
        return out_img   

    def compute_lane_lines(self, warped_img):

        # Take a histogram of the bottom half of the image, summing pixel values column wise 
        histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines 
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint # don't forget to offset by midpoint!


        # Set height of windows
        window_height = np.int(warped_img.shape[0]//self.sliding_windows_per_line)
        # Identify the x and y positions of all nonzero pixels in the image
        # NOTE: nonzero returns a tuple of arrays in y and x directions
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        total_non_zeros = len(nonzeroy)
        non_zero_found_pct = 0.0
        one_line_detected = False

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base    


        # Set the width of the windows +/- margin
        margin = self.sliding_window_half_width
        # Set minimum number of pixels found to recenter window
        minpix = self.sliding_window_recenter_thres
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Our lane line objects we store the result of this computation
        left_line = LaneLine()
        right_line = LaneLine()

        if self.previous_left_lane_line is not None and self.previous_right_lane_line is not None:
            # We have already computed the lane lines polynomials from a previous image
            left_lane_inds = ((nonzerox > (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                           + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                           + self.previous_left_lane_line.polynomial_coeff[2] - margin)) 
                              & (nonzerox < (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                            + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy 
                                            + self.previous_left_lane_line.polynomial_coeff[2] + margin))) 

            right_lane_inds = ((nonzerox > (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                           + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                           + self.previous_right_lane_line.polynomial_coeff[2] - margin)) 
                              & (nonzerox < (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2) 
                                            + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy 
                                            + self.previous_right_lane_line.polynomial_coeff[2] + margin))) 

            non_zero_found_left = np.sum(left_lane_inds)
            non_zero_found_right = np.sum(right_lane_inds)
            non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros

            print("[Previous lane] Found pct={0}".format(non_zero_found_pct))
            #print(left_lane_inds)

        if non_zero_found_pct < 0.85:
            print("Non zeros found below thresholds, begining sliding window - pct={0}".format(non_zero_found_pct))
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(self.sliding_windows_per_line):
                # Identify window boundaries in x and y (and right and left)
                # We are moving our windows from the bottom to the top of the screen (highest to lowest y value)
                win_y_low = warped_img.shape[0] - (window + 1)* window_height
                win_y_high = warped_img.shape[0] - window * window_height

                # Defining our window's coverage in the horizontal (i.e. x) direction 
                # Notice that the window's width is twice the margin
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                left_line.windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
                right_line.windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])

                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # These are the indices that are non zero in our sliding windows
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            #print('length right side lane: ' + str(right_lane_inds))

            non_zero_found_left = np.sum(left_lane_inds)
            non_zero_found_right = np.sum(right_lane_inds)
            non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros

            print("[Sliding windows] Found pct={0}".format(non_zero_found_pct))


        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        #print("[LEFT] Number of hot pixels={0}".format(len(leftx)))
        #print("[RIGHT] Number of hot pixels={0}".format(len(rightx)))
        # Fit a second order polynomial to each
        #print(leftx, rightx)
        if len(leftx) == 0 and len(rightx)==0:
          print("empty list no lines. start turning in a circle")
          return (0, 0, 0)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        if left_fit[0] == right_fit[0] and left_fit[1] == right_fit[1] and left_fit[2] == right_fit[2]:
          one_line_detected = True
          print("only one line create function here")
        mid_fit = (left_fit + right_fit)/2
        print("Poly left {0}".format(left_fit))
        print("Poly right {0}".format(right_fit))
        print("Poly mid {0}".format(mid_fit))

        if one_line_detected:
          left_fit[0] = 0
          left_fit[1] = 0
          left_fit[2] = 0
          new_leftx = []
          for item in leftx:
            new_leftx.append(0)
          leftx = new_leftx
          left_fit[0:2] = 0
          print("left line is positive hence needs to turn right")

        left_line.polynomial_coeff = left_fit
        right_line.polynomial_coeff = right_fit

        if not self.previous_left_lane_lines.append(left_line):
            left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
            left_line.polynomial_coeff = left_fit
            self.previous_left_lane_lines.append(left_line, force=True)
            print("**** REVISED Poly left {0}".format(left_fit))            
        #else:
            #left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
            #left_line.polynomial_coeff = left_fit


        if not self.previous_right_lane_lines.append(right_line):
            right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
            right_line.polynomial_coeff = right_fit
            self.previous_right_lane_lines.append(right_line, force=True)
            print("**** REVISED Poly right {0}".format(right_fit))
        #else:
            #right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
            #right_line.polynomial_coeff = right_fit


        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0] )
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        mid_fitx = mid_fit[0] * ploty**2 + mid_fit[1] * ploty + mid_fit[2]

        left_line.polynomial_coeff = left_fit
        left_line.line_fit_x = left_fitx
        left_line.non_zero_x = leftx  
        left_line.non_zero_y = lefty

        right_line.polynomial_coeff = right_fit
        right_line.line_fit_x = right_fitx
        right_line.non_zero_x = rightx
        right_line.non_zero_y = righty


        return (left_line, right_line, mid_fit)
      
  opts = 0
  ipts = 0
  algo = LineDetector(opts, ipts, src, dst, 20, 100, 50)
  mid_fit, left_l, right_l = algo.processImage(crop_mask)
  print(mid_fit, left_l, right_l)

  forward = 1
  right = 2
  left = 3
  backward = 4
  stop = 5
  terminate_program = 7

  def followTrajectory(mid, left, right):
    if left.all() == 0 and right.all() == 0: # logic to end program in case of no lines detected
      #print("here1")
      command = send_pkt(terminate_program, 0)
    else:
      if left.all() == 0:
        command = send_pkt(left, 60)
        print("turn left")
      if right.all()==0:
        command = send_pkt(right, 60)
        print("turn right")
      if left.all() != 0 and right.all() != 0:
        command = send_pkt(forward, 50)
        print("forward")
    return command
  def send_pkt(direction, pwm):
    cmd = str(direction) + "," + str(pwm)
    print(cmd)
    return cmd
    #ServerToPiSendCMD(command)

  command = followTrajectory(mid_fit, left_l, right_l)
  return command
#elapsed_time = timeit.timeit(main_func(), number=100)/100
PiToServerReceiveIMG()
time_end = time.time()
duration = time_end - time_start
print("duration = ", duration, " seconds")
