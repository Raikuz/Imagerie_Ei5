import os
os.system('pip install -r requirements.txt')

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
import math
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from PIL import Image
import pylab
import copy
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

			pygame.display.set_caption("JanKenPon Recognition !!!")

	# Function that create the database and plot it in a graphic
	def diplayingPlot(self, ax):
			global hullSizeRock
			global hullSizePaper
			global hullSizeScisors 

			global perimeterRock 
			global perimeterPaper 
			global perimeterScisors
			
			print("-----CHARGEMENT DE LA BASE DE DONNEES-----")
			
			# Files opening to create our database
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
			
			# Centroid calculation for each image and graphic generation
			for arrayrock in npArrayRock:
				hullSize, perimeter = self.get_feature(arrayrock)
				listHullRock.append(hullSize)
				listPeriRock.append(perimeter)				
				ax.scatter(hullSize,perimeter,color="Grey")
			
			for arrayscisors in npArrayCisor:
				hullSize, perimeter = self.get_feature(arrayscisors)
				listHullScisors.append(hullSize)
				listPeriScisors.append(perimeter)
				ax.scatter(hullSize,perimeter,color="Red")
				
			for arraypaper in npArrayPaper:
				hullSize, perimeter  = self.get_feature(arraypaper)
				listHullPaper.append(hullSize)
				listPeriPaper.append(perimeter)
				ax.scatter(hullSize,perimeter,color="Green")
				
			# Those variables estimate the centroid of each shape
			hullSizeRock = np.mean(listHullRock)
			hullSizePaper = np.mean(listHullPaper)
			hullSizeScisors = np.mean(listHullScisors)
			perimeterRock = np.mean(listPeriRock)
			perimeterPaper = np.mean(listPeriPaper)
			perimeterScisors = np.mean(listPeriScisors)
			
			
			print("-----CHARGEMENT TERMINE-----")
			
			# Here are displayed the centroids
			ax.scatter(hullSizeRock,perimeterRock,color="Grey",edgecolor='b')
			ax.scatter(hullSizePaper,perimeterPaper,color="Green",edgecolor='b')
			ax.scatter(hullSizeScisors,perimeterScisors,color="Red",edgecolor='b')
			
			
				
	# This function calculate two features for each image
	def get_feature(self, image):
			img_gray = cv2.cvtColor(np.uint8(image),cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(img_gray, 127, 255,0)
			immg,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
			hull = []
			perimeter = 0
			try:
				cnt = max(contours, key = cv2.contourArea)
				try:
					#Calculation of the hull size
					hull = cv2.convexHull(cnt,returnPoints = False)
					perimeter = cv2.arcLength(cnt,True)
				except IndexError:
					pass
			except ValueError:
				pass			
			return len(hull), perimeter
		
		
		
			
    # Frame processing    
	def draw_depth_frame(self, frame, target_surface):
			if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
					return
			target_surface.lock()
			f8 = np.uint8(frame.clip(1,4000)/16.)
			new_f8 = [np.uint8(255) if x > np.uint8(20) and x < np.uint8(50) else np.uint8(0) for x in f8]
			image = np.uint8(new_f8)
			
			# This create a third dimension to our image
			frame8bit=np.dstack((image,image,image))
			address = self._kinect.surface_as_array(target_surface.get_buffer())
			ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
			del address
			target_surface.unlock()
			return self.get_current_centroid(frame8bit)
       
	# This function calculate the nearest centroid from our current image
	def get_current_centroid(self, frame):
		image = np.reshape(frame,(424,512,3))
		x,y = self.get_feature(image)
		
		# Distance calculation from each centroid
		scisorsComp = math.sqrt( math.pow( x - hullSizeScisors,2 ) + math.pow( y - perimeterScisors,2) )
		rockComp = math.sqrt( math.pow( x - hullSizeRock ,2) + math.pow( y - perimeterRock,2) )
		paperComp = math.sqrt( math.pow( x - hullSizePaper ,2) + math.pow( y - perimeterPaper,2) )

		# Result interpretation
		if x > 10 and y > 10:
			if scisorsComp < rockComp and scisorsComp < paperComp:
				shape = "Ciseaux"
			elif rockComp < scisorsComp and rockComp < paperComp:
				shape = "Pierre"
			elif paperComp < rockComp and paperComp < scisorsComp:
				shape = "Papier"
		else:
			shape = "Defaut"	
		
		
		return shape, x, y
		
	def run(self):
    # -------- Main Program Loop -----------
			myfont = pygame.font.SysFont("monospace", 30)
			myfont.set_bold(True)
			fig = pylab.figure(figsize=[4, 4],dpi=100,)
			ax = fig.gca()
			self.diplayingPlot(ax)
			canvas = FigureCanvasAgg(fig)
			canvas.draw()
			
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
							shape, x, y = self.draw_depth_frame(frame, self._frame_surface)
							frame = None

					self._screen.blit(self._frame_surface, (0,0))
					try:
						point.remove()
					except:
						pass
					
					if x > 10 and y > 10:
						point = ax.scatter(x,y,color="Yellow")
					canvas.draw()
					renderer = canvas.get_renderer()
					raw_data = renderer.tostring_rgb()
					surf = pygame.image.fromstring(raw_data, size, "RGB")
					self._screen.blit(surf, (610,0))
					label = myfont.render(shape, 1, (255,255,255))
					self._screen.blit(label, (0, 0))
					pygame.display.update()

        # --- Go ahead and update the screen with what we've drawn.
					pygame.display.flip()

        # --- Limit to 60 frames per second
					self._clock.tick(60)

    # Close our Kinect sensor, close the window and quit.
			self._kinect.close()
			pygame.quit()


__main__ = "JanKenPon Recognition !!!"
game = depthRuntime();
game.run();


