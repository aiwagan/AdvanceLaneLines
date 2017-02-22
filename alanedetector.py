import numpy as np
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        #self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

class CLaneDetector():
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        self.n = 10
        self.left_line.recent_xfitted = np.zeros((10, 720))
        self.right_line.recent_xfitted = np.zeros((10, 720))
        
        self.detected = False 
        src = np.float32([ [590,450],[690,450],[270, 690],[1050,690]   ])
        dst =np.float32([ [200,0],[1000,0],[200, 720],[1000,720]   ]) 
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        self.n = 10
        self.iter_counter=0
        self.buffer_index=0
        #self.bird_view_transformer =Perspective_Transform(src, dst)
        self.yvals = np.arange(720)
        self.detected = None
        self.count = 0
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('./camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        self.ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[0:2],None,None)
            
    
    def corners_unwarp(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        img_size = (1280,720)
        warped = cv2.warpPerspective(undist, self.M, img_size)
        return warped
    
    
    # Edit this function to create your own pipeline.
    def pipeline(self, img):
        #img = np.copy(img)
        #img = self.corners_unwarp(img)

        #sxbinary= abs_sobel_thresh(img,'x',30,255)
        #smbinary = mag_thresh(img, sobel_kernel=3, mag_thresh=(100, 255))
        #ssbinary = hls_select(img, thresh=(100, 255))
        #shbinary = h_select(img, thresh=(10, 25))
        #fbinary = np.multiply(ssbinary,shbinary)
        #fbinary = fbinary+sxbinary+smbinary   #+ssbinary+shbinary
        #fbinary[(fbinary > 0)] = 255  #(fbinary/scale_factor).astype(np.uint8) 
        #return fbinary
        img = np.copy(img)
        img = self.corners_unwarp(img)
        #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        r_channel = img[:, :, 0]
        r_thresh = (200, 255)
        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
 
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        scharrx = cv2.Scharr(v_channel, cv2.CV_64F, 1, 0)
        abs_scharrx = np.absolute(scharrx)
        scaled_scharr = np.uint8(255 * abs_scharrx / np.max(abs_scharrx))
        thresh_min = 10
        thresh_max = 255
        scharr_x_binary = np.zeros_like(scaled_scharr)
        scharr_x_binary[(scaled_scharr >= thresh_min) & (scaled_scharr <= thresh_max)] = 1
        s_binary = np.zeros_like(s_channel)
        s_thresh_min = 100
        s_thresh_max = 255
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        v_binary = np.zeros_like(v_channel)
        v_thresh_min = 200
        v_thresh_max = 255
        v_binary[(v_channel >= v_thresh_min) & (v_channel <= v_thresh_max)] = 1
        binary = np.zeros_like(scharr_x_binary)
        binary[((v_binary == 1) & (s_binary == 1) | (r_binary == 1)  | (scharr_x_binary == 1))] = 255
        
        #binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
        return binary
    
    def line_detection(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.left_line.allx = nonzerox[left_lane_inds]
        self.left_line.ally = nonzeroy[left_lane_inds] 
        self.right_line.allx = nonzerox[right_lane_inds]
        self.right_line.ally = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        self.left_line.current_fit = np.polyfit(self.left_line.ally, self.left_line.allx, 2)
        self.right_line.current_fit = np.polyfit(self.right_line.ally, self.right_line.allx, 2)
        self.detected = True
        

    def fast_line_detection(self, binary_warped):
        """

        :param binary_warped:
        :return:
        """
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100

        left_lane_inds = (
            (nonzerox > (
                self.left_line.current_fit[0] * (nonzeroy ** 2) + self.left_line.current_fit[1] * nonzeroy + self.left_line.current_fit[2] - margin)) & (
                nonzerox < (
                    self.left_line.current_fit[0] * (nonzeroy ** 2) + self.left_line.current_fit[1] * nonzeroy + self.left_line.current_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (
                self.right_line.current_fit[0] * (nonzeroy ** 2) + self.right_line.current_fit[1] * nonzeroy + self.right_line.current_fit[2] - margin)) & (
                nonzerox < (
                    self.right_line.current_fit[0] * (nonzeroy ** 2) + self.right_line.current_fit[1] * nonzeroy + self.right_line.current_fit[2] + margin)))
        
         # Extract left and right line pixel positions
        self.left_line.allx = nonzerox[left_lane_inds]
        self.left_line.ally = nonzeroy[left_lane_inds] 
        self.right_line.allx = nonzerox[right_lane_inds]
        self.right_line.ally = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        self.left_line.current_fit = np.polyfit(self.left_line.ally, self.left_line.allx, 2)
        self.right_line.current_fit = np.polyfit(self.right_line.ally, self.right_line.allx, 2)
        
        #return left_fitx, right_fitx

    def calculate_road_info(self, image_size, left_x, right_x):
        """
        This method calculates left and right road curvature and off of the vehicle from the center
        of the lane

        :param image_size:
            Size of the image

        :param left_x:
            X coordinated of left lane pixels

        :param right_x:
            X coordinated of right lane pixels

        :return:
            Left and right curvatures of the lane and off of the vehicle from the center of the lane
        """
        # first we calculate the intercept points at the bottom of our image
        left_intercept = self.left_line.current_fit[0] * image_size[0] ** 2 + self.left_line.current_fit[1] * image_size[0] + self.left_line.current_fit[2]
        right_intercept = self.right_line.current_fit[0] * image_size[0] ** 2 + self.right_line.current_fit[1] * image_size[0] + self.right_line.current_fit[2]

        # Next take the difference in pixels between left and right interceptor points
        road_width_in_pixels = right_intercept - left_intercept
        #assert road_width_in_pixels > 0, 'Road width in pixel can not be negative'

        # Since average highway lane line width in US is about 3.7m
        # Source: https://en.wikipedia.org/wiki/Lane#Lane_width
        # we calculate length per pixel in meters
        meters_per_pixel_x_dir = 3.7 / road_width_in_pixels
        meters_per_pixel_y_dir = 30 / road_width_in_pixels

        # Recalculate road curvature in X-Y space
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, left_x * meters_per_pixel_x_dir, 2)
        right_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, right_x * meters_per_pixel_x_dir, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * meters_per_pixel_y_dir + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * meters_per_pixel_y_dir + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Next, we can lane deviation
        calculated_center = (left_intercept + right_intercept) / 2.0
        lane_deviation = (calculated_center - image_size[1] / 2.0) * meters_per_pixel_x_dir

        return left_curverad, right_curverad, lane_deviation
    
    def draw_poly(self, img):
        warped=self.pipeline(img)
        if self.detected:
            self.fast_line_detection(warped)
        else:
            self.line_detection(warped)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = self.left_line.current_fit[0]*ploty**2 + self.left_line.current_fit[1]*ploty + self.left_line.current_fit[2]
        right_fitx =  self.right_line.current_fit[0]*ploty**2 +  self.right_line.current_fit[1]*ploty +  self.right_line.current_fit[2]
        
        
        self.left_line.recent_xfitted[self.buffer_index] = left_fitx
        self.right_line.recent_xfitted[self.buffer_index] = right_fitx

        self.buffer_index += 1
        self.buffer_index %= self.n

        if self.iter_counter < self.n:
            self.iter_counter += 1
            self.left_line.bestx = np.sum(self.left_line.recent_xfitted, axis=0) / self.iter_counter
            self.right_line.bestx = np.sum(self.right_line.recent_xfitted, axis=0) / self.iter_counter
        else:
            self.left_line.bestx = np.average(self.left_line.recent_xfitted, axis=0)
            self.right_line.bestx = np.average(self.right_line.recent_xfitted, axis=0)

        
        

            
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (100,255, 10))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.M_inv, (img.shape[1], img.shape[0])) 

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        
        left_curvature, right_curvature, calculated_deviation = self.calculate_road_info(img.shape, self.left_line.bestx,
                                                                                         self.right_line.bestx)
        curvature_text = 'Left Curvature: {:.2f} m    Right Curvature: {:.2f} m'.format(left_curvature, right_curvature)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(result, curvature_text, (50, 50), font, 1, (221, 128, 11), 2)

        deviation_info = 'Lane Deviation: {:.3f} m'.format(calculated_deviation)
        cv2.putText(result, deviation_info, (50, 90), font, 1, (221, 128, 11), 2)

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Lane Detector.')
    parser.add_argument('-i', '--input', help = 'Input file', nargs='+')
    parser.add_argument('-o', '--output', help = 'Output file', nargs='+')
    args = parser.parse_args()
    #outputfile = parser.parse_args(['--output'])
    #print(args.output[0])
    
    #output = './output.mp4'
    output = args.output[0]
    #clip1 = VideoFileClip('./project_video.mp4')
    
    clip1 = VideoFileClip(args.input[0])
    
    ld = CLaneDetector()
    white_clip = clip1.fl_image(ld.draw_poly)
    white_clip.write_videofile(output, audio=False)
    print("process Finished. Please view %s" % output)
