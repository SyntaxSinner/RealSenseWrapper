import pyrealsense2 as rs
import numpy as np 
import cv2 
import math
from matplotlib import pyplot as plt
import time
import uuid
import os

"""
This class is a wrapper for the realsense camera. It contains methods for : 
1)Perform RGBD streaming through the rgbd_stream method
2)Perform RGBD shot taking through the rgbd_shot method, each shot is stored in the history list
*)Headless RGBD shot without windowing the output of the camera for Raspberry pi use 
3)Perform depth_filtered RGBD stream through the depth_filter_stream method
4)Calculate the distance between two points in the image through the calculate_distance method
5)clear the history of the wrapper through the clear_history method
6)Perform filtering over 8 consecutive frames through the processing_pipeline method
"""


class RealsenseWrapper:
    
    #Constructor for the RealSenseWrapper class, through it you can set the resolution and FPS of the RGB and Depth streams
    def __init__(self,color_resolution = (1280, 720), depth_resolution = (1280, 720), depth_fps = 30, color_fps = 30):
        
        #Default depth resolution was chosen to be 848x480 because it was the one recommended in the official documentation
        self.pipeline = rs.pipeline()
        
        #align object to align depth and color frames
        self.align = rs.align(rs.stream.color)
        
        #retrieve the intrinsics of the camera to perform calculations later
        self.intrinsics = rs.intrinsics()
        
        #config object to configure the pipeline and set up the RGB resolution, Depth, Depth format and FPS
        self.config = rs.config()
        #Set up the config for the depth stream
        self.config.enable_stream(rs.stream.depth,
                                  depth_resolution[0],
                                  depth_resolution[1],
                                  rs.format.z16,
                                  depth_fps)
        
        #Set up the config for the color stream
        self.config.enable_stream(rs.stream.color,
                                  color_resolution[0],
                                  color_resolution[1],
                                  rs.format.bgr8,
                                  color_fps)
        
        #align object to use for aligning depth to color frames
        self.align = rs.align(rs.stream.color)
        
        #history list to store the images taken by the rgbd_shot method
        self.history = []
        
        #1)Spatial filter
        self.spatial_magnitude = 3
        self.spatial_smooth_alpha = 0.5
        self.spatial_smooth_delta = 20
        self.spatial_holes_fill = 2
        
        #2)Temporal filter
        self.temporal_persistency_index = 3
        
        #3)Hole filling filter
        self.hole_filling = 2
        
        #Defining disparity and depth domain transforms
        
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        #Define the number of frames to consider for the temporal filter when filtering the depth stream
        self.frame_set_size = 8
        
        self.enable_spatial=True, 
        self.enable_temporal=True, 
        self.enable_hole_filling=False
        
    #Define filters 
    def define_set_spatial(self):
            
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, self.spatial_magnitude)
        self.spatial.set_option(rs.option.filter_smooth_alpha, self.spatial_smooth_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta, self.spatial_smooth_delta)
        self.spatial.set_option(rs.option.holes_fill, self.spatial_holes_fill)    
        
    def define_set_temporal(self):
        
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.holes_fill, self.temporal_persistency_index)
        
    def define_set_hole_filler(self):
        
        self.hole_filler =  rs.hole_filling_filter()
        self.hole_filler.set_option(rs.option.holes_fill, self.hole_filling)
        
    def define_set_filters(self):
        
        if self.enable_hole_filling:
            self.define_set_hole_filler()
            
        if self.enable_spatial:
            self.define_set_spatial()
            
        if self.enable_temporal:
            self.define_set_temporal()

    def clear_history(self):
        #clear the history of the wrapper from taken images to free up space
        del self.history[:]
        
    def rgbd_stream(self, enable_filters=True):
        self.define_set_filters()
        # Start streaming
        self.pipeline.start(self.config)
        print('Streaming started')
        
        try:
            while True:
                key = cv2.waitKey(1)
                color_frame = None
                depth_frame = None
                
                # Perform filtering over 8 consecutive frames
                for _ in range(self.frame_set_size):
                    
                    frameset = self.pipeline.wait_for_frames()
                    
                    if enable_filters:
                        frameset = self.processing_pipeline(frameset)
                        
                    aligned_frameset = self.align.process(frameset)
                    
                    depth_frame = aligned_frameset.get_depth_frame()
                    color_frame = aligned_frameset.get_color_frame()
                
                #Continue to the next iteration if either depth or color frame is missing
                if not depth_frame or not color_frame:
                    print("No depth or color frame")
                    continue

                #Convert depth and color images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                #Apply colormap on depth image (convert to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_PARULA)

                #Check the dimensions of depth and color images
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                #If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:

                    resized_color_image = cv2.resize(color_image,
                                                     dsize=(depth_colormap_dim[1],
                                                            depth_colormap_dim[0]),
                                                     interpolation=cv2.INTER_AREA
                                                     )
                    images = np.hstack((resized_color_image,
                                        depth_colormap)
                                       )

                else:

                    images = np.hstack((color_image,
                                        depth_colormap)
                                       )

                # Show the combined color and depth images in a window named 'RealSense'
                window_name = 'RealSense'
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(window_name, images)

                # Wait for a key event
                key = cv2.waitKey(1)

                # Check if the 'Esc' key (ASCII value 27) is pressed
                if key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
        # Stop streaming
            self.pipeline.stop()
            print('Streaming stopped')
        #starts a video stream of the color and depth cameras


    def rgbd_shot(self, enable_cache=True, enable_filters=True):
        self.define_set_filters()
        i=0
        
        self.pipeline.start(self.config)
        
        color_image = None
        depth_image = None
        frameset = None

        while True:
            
            key = cv2.waitKey(1)
            color_frame = None
            depth_frame = None
                
            # Perform filtering over 8 consecutive frames
            for _ in range(self.frame_set_size):
                    
                frameset = self.pipeline.wait_for_frames()
                    
                if enable_filters:
                    frameset = self.processing_pipeline(frameset)
                        
                aligned_frameset = self.align.process(frameset)
                    
                depth_frame = aligned_frameset.get_depth_frame()
                color_frame = aligned_frameset.get_color_frame()

            if color_frame and depth_frame:
                # Convert the color frame to a numpy array
                color_image = np.asanyarray(color_frame.get_data())
                # Convert the depth frame to a numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                # Apply colormap on depth image (convert to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_PARULA)

                # Check the dimensions of depth and color images
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                # Show the combined color and depth images in a window named 'Color/Depth'
                cv2.namedWindow('Color/Depth', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Color/Depth', images)

            if key == ord('p'):
                
                print('Shots taken: ',i)
                i+=1
                #Store the RGB And depth images in the history as a tupple for later inspection if enable cache is set to true
                if enable_cache:
                    self.history.append((color_frame, depth_frame))

            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.pipeline.stop()
        
        #Store the RGB And depth images in the history as a tupple for later inspection
        
    def headless_shot(self, enable_filters=True, ttt_shot=10):
        
        self.define_set_filters()
        self.pipeline.start(self.config)

        color_frame = None
        depth_frame = None
        frameset = None

        while True:
            
            color_frame = None
            depth_frame = None
            start_time=time.time()
            
            
            # Perform filtering over 8 consecutive frames
            for _ in range(self.frame_set_size):
                
                frameset = self.pipeline.wait_for_frames()

                if enable_filters:
                    frameset = self.processing_pipeline(frameset)

                aligned_frameset = self.align.process(frameset)

                depth_frame = aligned_frameset.get_depth_frame()
                color_frame = aligned_frameset.get_color_frame()

                current_time=time.time()
                elapsed_time = current_time - start_time
            
                if elapsed_time >= ttt_shot :
                    rgb_array = np.asanyarray(color_frame.get_data())
                    depth_array = np.asanyarray(depth_frame.get_data())
                    self.pipeline.stop()
                    unique_id = str(uuid.uuid4())
                    rgb_filename = f"rgb_{unique_id}.png"
                    depth_filename = f"depth_{unique_id}.npy"

                    if not os.path.exists("rgb"):
                        os.makedirs("rgb")
                    if not os.path.exists("depth"):
                        os.makedirs("depth")

                    cv2.imwrite(os.path.join("rgb", rgb_filename), rgb_array)
                    np.save(os.path.join("depth", depth_filename), depth_array)


                    return(unique_id)
                    
                    
                
                
    
    """
    starts a video stream of the color and depth cameras with the background removed
    you specify the distance in meters that you want to be in focus, every object
    located at a farther distance will get removed from the picture 
    """
    
    def depth_filter_stream(self, enable_filters = True, clipping_distance_in_meters = 1):
        self.define_set_filters()
        i=0
        #start the pipeline and get the profile
        profile=self.pipeline.start(self.config)
        
        color_image = None
        depth_image = None
        frameset = None
        
        # Getting the depth sensor's depth scale 
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        
        # We will be removing the background of objects more than
        # Clipping_distance_in_meters meters away
        clipping_distance = clipping_distance_in_meters / depth_scale
        

        
        while True:

            key = cv2.waitKey(1)
                
                
            color_frame = None
            depth_frame = None
                
            #Perform filtering over 8 consecutive frames
            
            for _ in range(self.frame_set_size):
                    
                frameset = self.pipeline.wait_for_frames()
                    
                if enable_filters:
                    frameset = self.processing_pipeline(frameset)
                        
                aligned_frameset = self.align.process(frameset)
                    
                depth_frame = aligned_frameset.get_depth_frame()
                color_frame = aligned_frameset.get_color_frame()
                
                
                
            if color_frame and depth_frame:
                # Convert the color frame to a numpy array
                color_image = np.asanyarray(color_frame.get_data())
                # Convert the depth frame to a numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            
                # Apply colormap on depth image (convert to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Check the dimensions of depth and color images
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = bg_removed.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(bg_removed, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((bg_removed, depth_colormap))

                # Show the combined color and depth images in a window named 'Color/Depth'
                cv2.namedWindow('Color/Depth', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Color/Depth', images)

            if key == ord('p'):
                
                print('Shots taken: ',i)
                i+=1
                #Store the RGB And depth images in the history as a tupple for later inspection
                
                self.history.append((color_image,depth_image))

            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.pipeline.stop()

    #this function starts a stream with the purpose of taking a picture to measure the object dimensions 

    def calculate_distance(self, list_of_2_points):

        #calculating the distance between 2 points 
        #extracting points
        point1, point2 = list_of_2_points
        #extracting coordinates
        x1, y1 = point1
        x2, y2 = point2
        
        #Retrieving the latest depth frame from the history of the wrapper
        depth_frame = self.history[-1][1]

        depth1=depth_frame.get_distance(x1,y1)
        depth2=depth_frame.get_distance(x2,y2)


        point1 = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x1, y1], depth1)
        point2 = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x2, y2], depth2)

        dist = math.sqrt(
            math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(
                point1[2] - point2[2], 2))
        
        #print 'distance: '+ str(dist)
        return dist
    
    def processing_pipeline(self, frame):
        
        """
        When setting the filtering pipeline might as well consider either activating the temporal filter or the hole_filling filter since both perform the same 
        task and having both of them activated will only slow down the process and duplicate their work, the temporal filter would not even be needed if Hole-Filling 
        can fill the holes on its own.
        """
        
        """
        Filters are applied according to the order mentioned in the official documentation 
        https://dev.intelrealsense.com/docs/post-processing-filters
        """
        
        #Defining filters only when 
        
        
        frame = self.depth_to_disparity.process(frame)
        
        if self.enable_spatial:
            frame = self.spatial.process(frame)
            
            
        if self.enable_temporal:
            frame = self.temporal.process(frame)
            
            
        if self.enable_hole_filling:
            frame = self.hole_filler.process(frame)
            
            
        frame = self.disparity_to_depth.process(frame).as_frameset()
        
        return frame

    @staticmethod
    def display_rgb_and_depth(rgb_array, depth_array):
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_array)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')

        if depth_array.ndim == 2:
            # Display depth as a heatmap with 'hot' colormap
            axes[1].imshow(depth_array, cmap='hot', interpolation='nearest')
        else:
            # If the depth array has multiple channels (e.g., RGBD), we'll use the first channel for visualization
            axes[1].imshow(depth_array[:, :, 0], cmap='hot', interpolation='nearest')
        axes[1].set_title('Depth Heatmap')
        axes[1].axis('off')

        plt.show()