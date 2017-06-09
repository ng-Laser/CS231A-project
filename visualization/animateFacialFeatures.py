import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os 
import cv2 
import pickle

def generateFrameNumbers(meta_data):
  numframes = meta_data["nframes"]
  fps = meta_data["fps"]
  print('video fps {0}'.format(fps))
  numFrames = int(meta_data['duration']*20) # used 20 in original script
  print('numframes {0}'.format(numframes))
  frames = np.zeros(numFrames)
  for f in range(numFrames):
  #    frames[f] = int((f*fps)/20)
     frames[f] = int((f*20)/fps)
  frames = frames.astype(np.int32)
  print(frames)
  return frames

def getData(indx, frameNum, vid):
  image_data = vid.get_data(frameNum) # later check out why frames think's it's not an int
  for point in data[indx,:]: # 68 points for frame number i 
    # draw the center of the circle
    cv2.circle(image_data,(point[0], point[1]),2,(0,255,0),3)
  return image_data

# function to update figure
def updatefig(j, im, frames, vid):
  if(j % 50) == 0:
    print('frame {0}'.format(j))
  image_data = getData(j, frames[j],vid )
  im.set_array(image_data)
  # return the artists set
  return [im]

def animateIrisTracking(moviePath, data, fps, vid):
  assert(len(frames) == data.shape[0])
  numImages = len(frames) 
  videoName = moviePath[:-4] + "_facePoints" + ".mp4"
  fig1 = plt.figure() # make figure
  im = plt.imshow(getData(0, frames[0], vid))
  line_ani = animation.FuncAnimation(fig1, updatefig, numImages,fargs=(im, frames, vid),
                                     interval=int(1000/fps))
  line_ani.save(videoName, fps=fps)


# Expects one argument which is the path to directory of .p files"
# containing object with fields 'data':nxpx2 matrix, and 'fps' - frames "
# n is the number of frames, p is the number of points and 2 is x,y"
# result is saving corresponding animations in current directory
if __name__ == '__main__':
  if len(sys.argv) < 3:
      print(
       "Expects atleast 2 argument\n"+
       "Arg 1 is path to video you want to do eye tracking on\n" + 
       "Arg2 is the path to video you have 86 features extracted from"
       # " Arg 2 can be a directory where you want all of the images"
      )
      exit()

  moviePath = sys.argv[1]
  movieDataPath = sys.argv[2]
  
  vid = None
  try:
    vid = imageio.get_reader(moviePath,  'ffmpeg')
  except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    print('Could not open video {0}'.format(moviePath))
    exit()

  a = pickle.load(open(movieDataPath, 'rb'))

  frames = None
  if(a.get('frames') is None):
   print('Old and buggy kind')
   meta_data = vid.get_meta_data()
   frames = generateFrameNumbers(meta_data)
  else:
   frames = a['frames'].astype(np.int32)

  data = a['data']
  data = data.astype(np.int32) # for plotting purposes 
  assert(data.shape[1] == 68)
  assert(data.shape[2] == 2)
  fps = a['fps']

  animateIrisTracking(moviePath, data, fps, vid)
