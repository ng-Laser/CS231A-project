'''
Code copied from
http://dlib.net/face_landmark_detection.py.html
'''
# imports from frame extraction
# import cv2
import imageio
import numpy as np
import scipy
from  scipy import misc

# imports from dlib test 
import sys
import os
import dlib

import pickle
import cv2

#TODO: delete
import matplotlib.pyplot as plt

CROP_BUFFER = 8

def _printOutError(videoName, message):
  print('For video {0}: {1}'.format(videoName, message))


def getDarkestCircle(image, circles):
   def getAvgColor(i):
     center_x,center_y = (i[0],i[1])
     min_x, max_x = center_x - i[2], center_x + i[2]
     min_y, max_y = center_x - i[2], center_x + i[2]
     # return np.mean(image[min_y:max_y, min_x:max_x,:])
     a = image[min_y:max_y, min_x:max_x,:]
     b = np.zeros(a.shape)
     b[a < 180] = 1.0
     return np.sum(b)/b.size

   print([getAvgColor(i) for i in circles[0,:]])
   indx = np.nanargmax([getAvgColor(i) for i in circles[0,:]])
   print(circles)
   print(circles[0,indx,:])
   return circles[0, indx, :]

def drawCirclesOnImages(image, eye, circles, offset_x, offset_y):
   print('num circles {0}'.format( circles.shape))
   circles = np.uint16(np.around(circles)) # around rounds
   darkestCircle = getDarkestCircle(eye, circles)

   circles[0,:,0] = circles[0,:,0] + offset_x
   circles[0,:,1] = circles[0,:,1] + offset_y
   print(darkestCircle)
   # darkestCircle = darkestCircle + np.array([offset_x, offset_y, 0])
   print(circles[0,:])
   print(darkestCircle)
   for i in circles[0,:]:
       # draw the outer circle
       cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2) # BGR
       # draw the center of the circle
       cv2.circle(image,(i[0],i[1]),2,(255,0,0),3)

   # draw the outer circle
   cv2.circle(image,(darkestCircle[0], darkestCircle[1]),darkestCircle[2],(0,0,255),2) # BGR
   # draw the center of the circle
   cv2.circle(image,(darkestCircle[0],darkestCircle[1]),2,(0,0,255),3)


# expects data to be slice relevant to this frame 
# returns eye, top left corner ofsets 
def cropLeftEye(img, data):
  minX, maxX = (data[36, 0] - CROP_BUFFER, data[39, 0] + CROP_BUFFER)
  minY, maxY = (data[38, 1] - CROP_BUFFER, data[41, 1] + CROP_BUFFER)
  eye = img[minY:maxY,minX:maxX ]
  # cv2.imshow('leftEye', eye)
  return (eye, minX, minY)

  
# expects data to be slice relevant to this frame 
# returns eye, top left corner ofsets 
def cropRightEye(img, data):
  minX, maxX = (data[42, 0] - CROP_BUFFER, data[45, 0] + CROP_BUFFER)
  minY, maxY = (data[44, 1] - CROP_BUFFER, data[46, 1] + CROP_BUFFER)
  eye = img[minY:maxY,minX:maxX ]
  # cv2.imshow('rightEye', eye)
  return (eye, minX, minY)
 
def extractIrisForEachFrame(videoPath, dataPath):
  a = pickle.load(open( dataPath, "rb" )) 
  data = a['data']

  desiredFPS = 20 # in frames per second
  # start of model initial set up 
  # open video/ get metadata
  try:
    vid = imageio.get_reader(videoPath,  'ffmpeg')
  except:
    _printOutError(videoPath,'Could not open video')
    return None

  meta_data = vid.get_meta_data()
  numframes = meta_data["nframes"]
  fps = meta_data["fps"]
  if(fps < desiredFPS):
    _printOutError(videoPath,'Frame rate of original video too low')
    return None
  numFrames = int(meta_data['duration']*desiredFPS)
  
  ptsFromFrames = np.array([]) # will eventually be a nx2x3 : n frames, 2 eyes, c_x, c_y, r 
  for f in range(100, numFrames):
  # for f in range(10):
     # print(int((f*fps)/desiredFPS))
     print('FRAME {0}'.format(f))
     image = vid.get_data(int((f*desiredFPS)/fps))
     # Ask the detector to find the bounding boxes of each face. The 1 in the
     # second argument indicates that we should upsample the image 1 time. This
     # will make everything bigger and allow us to detect more faces.
     eyeLeft,  minX_l, minY_l = cropLeftEye(image, data[f,:,:])
     eyeRight, minX_r, minY_r = cropRightEye(image, data[f,:,:])

     grayEye =  cv2.cvtColor(eyeLeft, cv2.COLOR_BGR2GRAY)
     grayEye = cv2.medianBlur(grayEye,3)
     # print('grayEye shape {0}'.format(grayEye.shape))
     # cv2.imwrite('testEyePlain.jpg', grayEye)
     circlesLeft = cv2.HoughCircles(grayEye,cv2.HOUGH_GRADIENT,1,int(grayEye.shape[0]*.25),
               param1=30,param2=15, minRadius=int(grayEye.shape[0]*.25), maxRadius=int(grayEye.shape[0]*.6))
     # previously used 60,20
     if(circlesLeft != None):
       try:
         drawCirclesOnImages(image,eyeLeft, circlesLeft, minX_l, minY_l)
       except:
         print("could not draw Left eye for frame {0}".format(f))
     grayEye =  cv2.cvtColor(eyeRight, cv2.COLOR_BGR2GRAY)
     grayEye = cv2.medianBlur(grayEye,3)
     #cv2.imwrite('testEyePlain.jpg', grayEye)
     circlesRight = cv2.HoughCircles(grayEye,cv2.HOUGH_GRADIENT,1,int(grayEye.shape[0]*.25),
               param1=30,param2=15, minRadius=int(grayEye.shape[0]*.25), maxRadius=int(grayEye.shape[0]*.6))
     # print(circlesRight)

     if(circlesRight != None):
       try:
         drawCirclesOnImages(image,eyeRight, circlesRight, minX_r, minY_r)
       except:
         print("could not draw right eye for frame {0}".format(f))
     cv2.imwrite('testEye{0}.jpg'.format(f), image)
     # if ptsFromFrames.size == 0:
     #  ptsFromFrames =  xy
     # else:
     #   ptsFromFrames =  np.concatenate((ptsFromFrames, xy), axis=0) 

  # result = {
  #  'data': ptsFromFrames,
  # 'fps': ptsFromFrames.shape[0]/meta_data['duration']
  #}
  #return result

if __name__ == '__main__':
  moviePath = '/home/noa_glaser/data/train-5-6/videos/Yj36y7ELRZE.000.mp4'
  movieDataPath = '/home/noa_glaser/data/train-5-6/extractedFacialFeatures/Yj36y7ELRZE.000.p'
 
  extractIrisForEachFrame(moviePath, movieDataPath)
