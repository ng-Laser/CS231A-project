import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os 
import cv2 
import pickle

def animateIrisTracking(moviePath, data, frames, fps):
  try:
    vid = imageio.get_reader(moviePath,  'ffmpeg')
  except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    print('Could not open video {0}'.format(moviePath))
    return None

  def getData(i):
    image_data = vid.get_data(int(frames[i])) # later check out why frames think's it's not an int
    for circle in data[i,:]: # left circle, right circle 
      if(-1 not in circle):
        print('circle {0}'.format(circle))
        # draw the outer circle
        cv2.circle(image_data,(circle[0], circle[1]),int(circle[2]),(0, 255,0),2) # BGR
        # draw the center of the circle
        cv2.circle(image_data,(circle[0], circle[1]),2,(0,255,0),3)
    return image_data

  # function to update figure
  def updatefig(j):
      im.set_array(getData(j))
      # return the artists set
      return [im]

  numImages = len(frames) 
  videoName = moviePath[:-4] + "_iris" + ".mp4"

  fig1 = plt.figure() # make figure
  im = plt.imshow(getData(0))
  line_ani = animation.FuncAnimation(fig1, updatefig, numImages,
                                     interval=int(1000/fps))
  line_ani.save(videoName, fps=fps)


# Expects one argument which is the path to directory of .p files"
# containing object with fields 'data':nxpx2 matrix, and 'fps' - frames "
# n is the number of frames, p is the number of points and 2 is x,y"
# result is saving corresponding animations in current directory
if __name__ == '__main__':
  if len(sys.argv) < 3:
      print(
       "Expects atleast 3 argument\n"+
       "Arg 1 is path to video you want to do eye tracking on\n" + 
       "Arg2 is the path to video you have 86 features extracted from"
       # " Arg 2 can be a directory where you want all of the images"
      )
      exit()

  moviePath = sys.argv[1]
  movieDataPath = sys.argv[2]

  a = pickle.load(open(movieDataPath, 'rb'))
  data = a['data']
  frames = a['frames']
  fps = a['fps']

  animateIrisTracking(moviePath, data, frames, fps)
