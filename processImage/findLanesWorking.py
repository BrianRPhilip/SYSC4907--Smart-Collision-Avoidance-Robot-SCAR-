import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def skeletonize(img):
    skimage.morphology.skeletonize(img)


def equalization(img):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 255, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # Here is how we will proceed on drawing discontinued lane lines
    '''
    First find the slopes of all lines, then divide lines into left and 
    right lane bases on their slope and position. Extrapolate on right and left lines using
    using linear regrssion over lines segments to find the best fit line for
    left and right lane line.
    After finding left and right final lines, we uses the slope and intercept to
    find the end points of the lines and use them to draw the lane lines
    on the output image.
    '''

    draw_right = True
    draw_left = True

    # Find slopes of all lines
    # But only retain lines where abs(slope) > threshold
    trap_height = 0.4
    slope_threshold = 0.5
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]  # extract start and end points

        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding divide by zero
            slope = 999.  # infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)

        # Filter lines based on slopes
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
    lines = new_lines

    # Separate lines into left and right lines based on weither their
    # respective slope are positive or negative
    r_lines = []
    l_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]  # extract start and end points
        img_x_center = img.shape[1] / 2  # x coordinate of the centre of the image
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            r_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            l_lines.append(line)

    # Interpolate over all line segments to find the best fit of right and
    # left lane lines
    # Right lines
    r_lines_x = []
    r_lines_y = []

    for line in r_lines:
        x1, y1, x2, y2 = line[0]  # extract start and end points

        # x coordinates of lines segments in right lines
        r_lines_x.append(x1)
        r_lines_x.append(x2)

        # y coordinates of lines segments in right lines
        r_lines_y.append(y1)
        r_lines_y.append(y2)

    # use polyfit to run linear regrssion to construct the final line
    if len(r_lines_x) > 0:
        r_m, r_b = np.polyfit(r_lines_x, r_lines_y, 1)  # y = x*m + b
    else:
        r_m, r_b = 1, 1
        draw_right = False

    # Left lane lines
    l_lines_x = []
    l_lines_y = []

    for line in l_lines:
        x1, y1, x2, y2 = line[0]  # extract start and end points

        # x coordinates of lines segments in left lines
        l_lines_x.append(x1)
        l_lines_x.append(x2)

        # y coordinates of lines segments in left lines
        l_lines_y.append(y1)
        l_lines_y.append(y2)

    # use polyfit to run linear regrssion to construct the final line
    if len(l_lines_x) > 0:
        l_m, l_b = np.polyfit(l_lines_x, l_lines_y, 1)  # y = x*m + b
    else:
        l_m, l_b = 1, 1
        draw_left = False

    # Find 2 end points for right and left lines, used to draw the lines
    # We use line equation y = mx+b => x = (y-b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)
    # Convert points from float to int
    y1 = int(y1)
    y2 = int(y2)

    right_x1 = (y1 - r_b) / r_m
    right_x2 = (y2 - r_b) / r_m

    left_x1 = (y1 - l_b) / l_m
    left_x2 = (y2 - l_b) / l_m

    # convert to int
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90, 100, 100])
    upper_yellow = np.array([110, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    return image2


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.


def _process_image(image_in):
    # color selection
    image = filter_colors(image_in)

    # Equalization
    # equalize = equalization(image)

    # read in and grayscale the image
    gray = grayscale(image)

    # define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)

    # define our parameters for Canny and apply
    # Otu's threshold selection method
    high_threshold, thresh_im = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = 0.5 * high_threshold
    # low_threshold = 50
    # high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # this time we are defining a four sided polygon to mask
    # We want a trapezoid shape, with bottom edge at the bottom of the image
    # trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
    # trap_top_width = 0.07  # ditto for top edge of trapezoid
    # trap_height = 0.4  # height of the trapezoid expressed as percentage of image height

    imshape = image.shape
    vertices = np.array([[(100, imshape[0]), (440, 325), (550, 325), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    rho = 2
    theta = np.pi / 360
    threshold = 15
    min_line_len = 10
    max_line_gap = 200

    # run Hough on edge detected image
    # output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # drawing the lines on the edge image
    initial_image = image_in.astype('uint8')
    result = weighted_img(line_image, initial_image)
    return result


# Display an example image
annotated_image = _process_image(cv2.imread('solidWhiteRight.jpg'))
cv2.imshow(annotated_image)