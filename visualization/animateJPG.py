import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import sys
import os 

def animateFromImages(dirPath, fileTemplate, fps):
  def generateImageName(i):
    return os.path.join(dirPath, fileTemplate) + "{0}.jpg".format(i)

  # function to update figure
  def updatefig(j):
      # set the data in the axesimage object
      data = mpimg.imread(generateImageName(j))
      im.set_array(data)
      # return the artists set
      return [im]

  numImages = len([ f for f in os.listdir(dirPath) if f.startswith(fileTemplate)])
  videoName = fileTemplate + ".mp4"

  fig1 = plt.figure() # make figure
  im = plt.imshow( mpimg.imread(generateImageName(0)))
  line_ani = animation.FuncAnimation(fig1, updatefig, numImages,
                                     interval=(1000/fps))
  line_ani.save(videoName, fps=fps)


# Expects one argument which is the path to directory of .p files"
# containing object with fields 'data':nxpx2 matrix, and 'fps' - frames "
# n is the number of frames, p is the number of points and 2 is x,y"
# result is saving corresponding animations in current directory
if __name__ == '__main__':
  if len(sys.argv) < 3:
      print(
       "Expects atleast 2 argument\n"+
       "Arg 1 is path to directory containing sequence of images\n" + 
       "Arg 2 is template of file name (with frames numbered starting at '0')\n" + 
       "    All frames will be called '<template><number>.jpg'\n" + 
       "Arg 3 argument can be FPS, default is 20"
      )
      exit()

  dirPath = sys.argv[1]
  fileTemplate = sys.argv[2]

  FPS = 20
  if(len(sys.argv) > 3):
    FPS = sys.argv[3]

  animateFromImages(dirPath, fileTemplate, FPS)
