# imports from frame extraction
# import cv2
import imageio
import numpy as np
import scipy
from  scipy import misc
import matplotlib.pyplot as plt

# imports from dlib test 
import sys
import os

import pickle
import cv2

CROP_BUFFER = 0

def reportPercentMissingEye(data):
  b_right = [1 for i in range(data.shape[0]) if -1 in data[i,0,:]]
  b_left = [1 for i in range(data.shape[0]) if -1 in data[i,1,:]]
  return (len(b_left)*1.0/data.shape[0], len(b_right)*1.0/data.shape[0])

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
  return frames

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

   indx = np.nanargmax([getAvgColor(i) for i in circles[0,:]])
   return circles[0, indx, :]

def drawCirclesOnImages(image, eye, circles, offset_x, offset_y):
   # darkestCircle = getDarkestCircle(eye, circles)

   circles[0,:,0] = circles[0,:,0] + offset_x
   circles[0,:,1] = circles[0,:,1] + offset_y
   # print(darkestCircle)
   # darkestCircle = darkestCircle + np.array([offset_x, offset_y, 0])
   # print(darkestCircle)
   for i in circles[0,:]:
       # draw the outer circle
       cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2) # BGR
       # draw the center of the circle
       cv2.circle(image,(i[0],i[1]),2,(255,0,0),3)

def cropEye(eye, roi_corners):
  assert(eye is not None)
  grayEye =  cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
  if(grayEye is None):
    print eye
  assert(grayEye is not None)
  grayEye = cv2.medianBlur(grayEye,3)
  mask = np.zeros(grayEye.shape, dtype=np.uint8)
  cv2.fillPoly(mask, roi_corners, 255)
  return  cv2.bitwise_and(grayEye, mask)

# expects data to be slice relevant to this frame 
# returns eye, top left corner ofsets 
def cropLeftEye(img, data):
  minX, maxX = (data[36, 0] - CROP_BUFFER, data[39, 0] + CROP_BUFFER)
  minY, maxY = (data[38, 1] - CROP_BUFFER, data[41, 1] + CROP_BUFFER)
  eye = img[minY:maxY,minX:maxX ]
  if(eye.shape[0] == 0): # eye is Closed completely
    return (None, -1, -1)
  roi_corners = np.array([data[range(36,42), :]], dtype=np.int32)
  roi_corners =  roi_corners - np.array([ minX, minY], dtype='int32')
  grayEye = cropEye(eye, roi_corners)
  return (grayEye, minX, minY)
   
# expects data to be slice relevant to this frame 
# returns eye, top left corner ofsets 
def cropRightEye(img, data):
  minX, maxX = (data[42, 0] - CROP_BUFFER, data[45, 0] + CROP_BUFFER)
  minY, maxY = (data[44, 1] - CROP_BUFFER, data[46, 1] + CROP_BUFFER)
  eye = img[minY:maxY,minX:maxX ]
  if(eye.shape[0] == 0): # eye is Closed completely
    return (None, -1, -1)
  roi_corners = np.array([data[range(42, 48), :]], dtype=np.int32)
  roi_corners =  roi_corners - np.array([ minX, minY], dtype='int32')
  grayEye = cropEye(eye, roi_corners)
  return (grayEye, minX, minY)

def _drawLeftEye(img, data): # debug
  minX, maxX = (data[36, 0] - CROP_BUFFER, data[39, 0] + CROP_BUFFER)
  minY, maxY = (data[38, 1] - CROP_BUFFER, data[41, 1] + CROP_BUFFER)
  roi_corners = np.array([data[range(36,42), :]], dtype=np.int32)
  cv2.rectangle(img, (minX, maxY), (maxX, minY),(0,255,0),3) 
  for point in roi_corners[0,:]: # 68 points for frame number i 
    cv2.circle(img,(point[0], point[1]),2,(0,255,0),3)
  return img

def _drawRightEye(img, data): # debug
  minX, maxX = (data[42, 0] - CROP_BUFFER, data[45, 0] + CROP_BUFFER)
  minY, maxY = (data[44, 1] - CROP_BUFFER, data[46, 1] + CROP_BUFFER)
  roi_corners = np.array([data[range(42, 48), :]], dtype=np.int32)
  cv2.rectangle(img, (minX, maxY), (maxX, minY),(0,255,0),3) 
  for point in roi_corners[0,:]: # 68 points for frame number i 
    cv2.circle(img,(point[0], point[1]),2,(0,255,0),3)
  return img

def extractCirclesDraw(eye, image, draw, minX, minY):
  # cv2.imwrite('testEyePlain.jpg', grayEye)
  circles = cv2.HoughCircles(eye,cv2.HOUGH_GRADIENT, 1,int(eye.shape[1]),
         param1=30,param2=10, minRadius=int(eye.shape[0]*.1))
  if(circles is not None):
    circles = np.uint16(np.around(circles)) # around rounds
    try:
      drawCirclesOnImages(image,eye, circles, minX, minY)
    except Exception as inst:
      print(type(inst))    # the exception instance
      print(inst.args)     # arguments stored in .args
      print("could not draw eye")
    return circles[0,0,:]
  return np.array([-1, -1, -1])

 
def extractIrisForEachFrame(videoPath, dataPath, drawOutFrames=False):
  a = pickle.load(open( dataPath, "rb" )) 
  data = a['data']

  # open video/ get metadata
  try:
    vid = imageio.get_reader(videoPath,  'ffmpeg')
  except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    _printOutError(videoPath,'Could not open video')
    return None

  meta_data = vid.get_meta_data()
  if(a.get('frames') is None):
    frames = generateFrameNumbers(meta_data) 
  else:
    frames = a['frames'].astype(np.int32)
  ptsFromFrames = np.array([]) # will eventually be a nx2x3 : n frames, 2 eyes, c_x, c_y, r 
  for f in range(len(frames)):
     if(f %50 == 0):
       print('FRAME {0}'.format(f))
     image = vid.get_data(frames[f])
     eyeLeft,  minX_l, minY_l = cropLeftEye(image, data[f,:,:])
     eyeRight, minX_r, minY_r = cropRightEye(image, data[f,:,:])
     # image = _drawLeftEye(image, data[f,:,:])
     # image = _drawRightEye(image, data[f,:,:])
     # cv2.imwrite('testDrawEye{0}.jpg'.format(f),image[min(minY_l, minY_r)-50:minY_l+100,minX_l - 50:minX_r+100 ])
     # continue

     if(eyeLeft is None): # currently will be none if height is zero, i.e. eye is full closed
        leftCircle = np.array([-1,-1,-1])
     else:
        leftCircle  = extractCirclesDraw(eyeLeft, image, drawOutFrames,  minX_l, minY_l)
     if(eyeRight is None):
        rightCircle = np.array([-1,-1,-1])
     else:
        rightCircle = extractCirclesDraw(eyeRight, image, drawOutFrames,minX_r, minY_r)

     if(drawOutFrames):
        cv2.imwrite('testEye{0}.jpg'.format(f), image)

     xy =  np.concatenate((leftCircle[np.newaxis,:], rightCircle[np.newaxis,:]),axis=0)
     if ptsFromFrames.size == 0:
      ptsFromFrames =  xy[np.newaxis, :]
     else:
       ptsFromFrames =  np.concatenate((ptsFromFrames, xy[np.newaxis,:]), axis=0) # hopefully this works  

  print(ptsFromFrames.shape)
  print('percent frames missing an eye {0}'.format(reportPercentMissingEye(ptsFromFrames)))
  result = {
    'data': ptsFromFrames, # order is left, right
    'fps': ptsFromFrames.shape[0]/meta_data['duration'],
    'frames': frames
  }
  return result

def forAllFilesInDir(pathData, pathMovie):
    i = 0
    numSuccess = 0
    listing = os.listdir(pathData)
    for f in listing:
        if f.endswith(".p") and '_' not in f:
            print(f)
            print(f[:-2])
            if(f[:-2] + "_iris.p" in listing):
              print('Skipping processing {0}, output already exists'.format(f))
              continue 
            outFileName = os.path.join(pathData, f[:-2]) + "_iris.p"
            movieName =  os.path.join(pathMovie, f[:-2]) + ".mp4"
            dataName = os.path.join(pathData, f)
            # if f in listDirSet:
            #    _printOutError(outFileName,'Skipping because output exists')
            #    continue 
            print(f) # print which video we are processing

            extracted = extractIrisForEachFrame(movieName, dataName, drawOutFrames=False)
            '''
             try:
              extracted = extractIrisForEachFrame(movieName, dataName, drawOutFrames=False)
            except Exception as inst:
              print(type(inst))    # the exception instance
              print(inst.args)     # arguments stored in .args
              print("Extraction for movie {0} failed".format(movieName))
              continue # should probably printn something here  
            '''
            if(extracted is not  None):
              pickle.dump(extracted,  open( outFileName , "wb" ) )
              numSuccess = numSuccess + 1
              print('So far outputed {0} files'.format(numSuccess))


if __name__ == '__main__':
  if len(sys.argv) < 3:
      print(
       "Expects atleast 3 argument\n"+
       "Arg 1 is path to video dir you want to do eye tracking on\n" + 
       "Arg2 is the path to .p dir you have 86 features extracted from"
       # " Arg 2 can be a directory where you want all of the images"
      )
      exit()

  pathMovie = sys.argv[1]
  pathData = sys.argv[2]
  drawOutFrames=False

  forAllFilesInDir(pathData, pathMovie)
  # a = extractIrisForEachFrame(moviePath, movieDataPath, drawOutFrames=drawOutFrames)
  #outFileName = moviePath[:-4] + "_iris.p"
  # pickle.dump(a,  open( outFileName , "wb" ) )

