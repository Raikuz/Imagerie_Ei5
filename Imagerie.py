import os
os.system('python -m pip install -r requirements.txt')

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import pykinect2
import ctypes
import _ctypes
import pygame
import sys
import numpy as np
import scipy
import cv2
from scipy import spatial
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from threading import Thread
from PIL import Image
import pylab
from matplotlib.backends.backend_agg import FigureCanvasAgg

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

hullSizeRock = 0
hullSizePaper = 0
hullSizeScisors = 0

perimeterRock = 0
perimeterPaper = 0
perimeterScisors = 0
	
class depthRuntime(object):
	def __init__(self):
			pygame.init()

    # Used to manage how fast the screen updates
			self._clock = pygame.time.Clock()

    # Loop until the user clicks the close button.
			self._done = False

    # Used to manage how fast the screen updates
			self._clock = pygame.time.Clock()

    # Kinect runtime object, we want only color and body frames 
			self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

    # back buffer surface for getting Kinect depth frames, 8bit grey, width and height equal to the Kinect color frame size
			self._frame_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
    # here we will store skeleton data 
			self._bodies = None
    
    # Set the width and height of the screen [width, height]
			self._infoObject = pygame.display.Info()
			self._screen = pygame.display.set_mode((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

			pygame.display.set_caption("Kinect for Windows v2 depth")

	def diplayingPlot(self, ax):
			global hullSizeRock
			global hullSizePaper
			global hullSizeScisors 

			global perimeterRock 
			global perimeterPaper 
			global perimeterScisors
			
			filelistCisor = glob.glob('EchantillonCiseaux/*.png')
			filelistPaper = glob.glob('EchantillonPapier/*.png')
			filelistRock = glob.glob('EchantillonPierre/*.png')

			npArrayCisor = np.array([np.array(Image.open(fname)) for fname in filelistCisor])
			npArrayPaper = np.array([np.array(Image.open(fname)) for fname in filelistPaper])
			npArrayRock = np.array([np.array(Image.open(fname)) for fname in filelistRock])
			
			listHullRock  = []
			listHullPaper  = []
			listHullScisors = []
			listPeriRock  = []
			listPeriPaper  = []
			listPeriScisors = []
			
			print("-----CHARGEMENT PIERRE-----")
			for arrayrock in npArrayRock:
				hullSize, perimeter = self.get_feature(arrayrock)
				listHullRock.append(hullSize)
				listPeriRock.append(perimeter)
				ax.scatter(hullSize,perimeter,color="Grey")
				
			print("-----CHARGEMENT CISEAUX-----")
			for arrayscisors in npArrayCisor:
				hullSize, perimeter = self.get_feature(arrayscisors)
				listHullScisors.append(hullSize)
				listPeriScisors.append(perimeter)
				ax.scatter(hullSize,perimeter,color="Red")
				
			print("-----CHARGEMENT PAPIER-----")
			for arraypaper in npArrayPaper:
				hullSize, perimeter  = self.get_feature(arraypaper)
				listHullPaper.append(hullSize)
				listPeriPaper.append(perimeter)
				ax.scatter(hullSize,perimeter,color="Green")
				
			hullSizeRock = np.mean(listHullRock)
			hullSizePaper = np.mean(listHullPaper)
			hullSizeScisors = np.mean(listHullScisors)

			perimeterRock = np.mean(listPeriRock)
			perimeterPaper = np.mean(listPeriPaper)
			perimeterScisors = np.mean(listPeriScisors)
				

	def get_feature(self, image):
			imageClean = image.flatten()
			_ ,contours, _ = cv2.findContours(imageClean,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
			hull = [cv2.convexHull(cnt) for cnt in contours]
			perimeter = [cv2.arcLength(cnt,True) for cnt in contours]
			return len(hull), np.sum(perimeter)
			
            
	def draw_depth_frame(self, frame, target_surface):
			if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
					return
			target_surface.lock()
			f8 = np.uint8(frame.clip(1,4000)/16.)
			new_f8 = [np.uint8(255) if x > np.uint8(20) and x < np.uint8(50) else np.uint8(0) for x in f8]
			image = np.uint8(new_f8)
			#self.classifier_homemade(new_f8)
			frame8bit=np.dstack((image,image,image))
			address = self._kinect.surface_as_array(target_surface.get_buffer())
			ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
			del address
			target_surface.unlock()
            

		
	def run(self):
    # -------- Main Program Loop -----------
			fig = pylab.figure(figsize=[4, 4],dpi=100,)
			ax = fig.gca()
			self.diplayingPlot(ax)
			canvas = FigureCanvasAgg(fig)
			canvas.draw()
			renderer = canvas.get_renderer()
			raw_data = renderer.tostring_rgb()
			size = canvas.get_width_height()
			while not self._done:
        # --- Main event loop
					for event in pygame.event.get(): # User did something
							if event.type == pygame.QUIT: # If user clicked close
									self._done = True # Flag that we are done so we exit this loop

							elif event.type == pygame.VIDEORESIZE: # window resized
									self._screen = pygame.display.set_mode((1000,400), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        # --- Getting frames and drawing  
					if self._kinect.has_new_depth_frame():
							frame = self._kinect.get_last_depth_frame()
							self.draw_depth_frame(frame, self._frame_surface)
							frame = None

					self._screen.blit(self._frame_surface, (0,0))
					surf = pygame.image.fromstring(raw_data, size, "RGB")
					self._screen.blit(surf, (610,0))
					pygame.display.update()

        # --- Go ahead and update the screen with what we've drawn.
					pygame.display.flip()

        # --- Limit to 60 frames per second
					self._clock.tick(60)

    # Close our Kinect sensor, close the window and quit.
			self._kinect.close()
			pygame.quit()


__main__ = "Kinect v2 depth"
game = depthRuntime();
game.run();


