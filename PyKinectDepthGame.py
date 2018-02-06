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

#import classifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import glob

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                pygame.color.THECOLORS["blue"], 
                pygame.color.THECOLORS["green"],
                pygame.color.THECOLORS["orange"], 
                pygame.color.THECOLORS["purple"], 
                pygame.color.THECOLORS["yellow"], 
                pygame.color.THECOLORS["violet"]]


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
            
	def classifier_homemade(self, frameArray):
			filelistCisor = glob.glob('EchantillonCiseaux/*.png')
			filelistPaper = glob.glob('EchantillonPapier/*.png')
			filelistRock = glob.glob('EchantillonPierre/*.png')

			npArrayCisor = np.array([np.array(Image.open(fname)) for fname in filelistCisor])
			npArrayPaper = np.array([np.array(Image.open(fname)) for fname in filelistPaper])
			npArrayRock = np.array([np.array(Image.open(fname)) for fname in filelistRock])

			clfCisor = NearestCentroid(metric='euclidean', shrink_threshold=None)
			clfCisor.fit(filelistCisor, frameArray)
			print("Cisor : "+clf.predict([[-0.8, -1]]))

			clfPaper = NearestCentroid(metric='euclidean', shrink_threshold=None)
			clfPaper.fit(filelistPaper, frameArray)
			print("Paper : "+clf.predict([[-0.8, -1]]))

			clfRock = NearestCentroid(metric='euclidean', shrink_threshold=None)
			clfRock.fit(filelistRock, frameArray)
			print("Rock : "+clf.predict([[-0.8, -1]]))
            
	def draw_depth_frame(self, frame, target_surface):
			if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
					return
			target_surface.lock()
			f8 = np.uint8(frame.clip(1,4000)/16.)
			new_f8 = [np.uint8(250) if x > np.uint8(40) and x < np.uint8(70) else np.uint8(0) for x in f8]
			self.classifier_homemade(new_f8)
			frame8bit=np.dstack((new_f8,new_f8,new_f8))
			address = self._kinect.surface_as_array(target_surface.get_buffer())
			ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
			del address
			target_surface.unlock()
            

	def run(self):
    # -------- Main Program Loop -----------
			while not self._done:
        # --- Main event loop
					for event in pygame.event.get(): # User did something
							if event.type == pygame.QUIT: # If user clicked close
									self._done = True # Flag that we are done so we exit this loop

							elif event.type == pygame.VIDEORESIZE: # window resized
									self._screen = pygame.display.set_mode(event.dict['size'], pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        # --- Getting frames and drawing  
					if self._kinect.has_new_depth_frame():
							frame = self._kinect.get_last_depth_frame()
							self.draw_depth_frame(frame, self._frame_surface)
							frame = None

					self._screen.blit(self._frame_surface, (0,0))
					pygame.display.update()

        # --- Go ahead and update the screen with what we've drawn.
					pygame.display.flip()

        # --- Limit to 60 frames per second
					self._clock.tick(60)

    # Close our Kinect sensor, close the window and quit.
			self._kinect.close()
			pygame.quit()


__main__ = "Kinect v2 depth"
game =depthRuntime();
game.run();

