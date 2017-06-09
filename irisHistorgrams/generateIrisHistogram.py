from gridClass import Historgram
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("/home/noa_glaser/CS231A-project/visualization")
import animatePoints
import pickle
import cv2

IRIS_HIST_ENDING = '_irisHst.p'
'''
TODO as an extention,
align the face to absolute forward
1) use optimization to find the homography that maximizes
   the criterion for the first face
2) align to a prior that is the structure of the face
'''
def _findMostForwardFace(featuresData):
  maxNoseLength = np.max(featuresData[:,30,1] - featuresData[:,27,1])
  # maxFaceWidth = np.max(featuresData[:,16,0] - featuresData[:,0,0])
  maxDistBtnEyes =  np.max(featuresData[:,42,0] - featuresData[:,39,0])

  def generateCriterion(featuresFrame):
     c = 0 # criterion
     # nose length as fraction of max length (align face pitch)
     c = c + (featuresFrame[30,1] - featuresFrame[27,1])*1.0
     # skewness of nose - along with above align face pitch, and roll 
     c = c - np.abs(featuresFrame[30,0] - featuresFrame[27,0])*7
     # difference between distances bridge to eye corner 
     c = c - np.abs((featuresFrame[27,0] - featuresFrame[36,0]) - (featuresFrame[45,0] - featuresFrame[27,0]))
     # consider normalizing by sizeof face?
     return c

  # try to minimize criterion to get most aligned face 
  criterions = [generateCriterion(featuresData[i,:]) \
       for i in range(featuresData.shape[0])]
  # return criterions
  indx = np.argmax(criterions)
  referenceFace = featuresData[indx,:]
  return referenceFace 

'''
Returns data from 500 frames where the specified iris was tracked
TODO as an extention: more intelligent guesses about where the irises are
   i.e. noise rejection, averages across subsequent frames

Inputs:
   numFrames, number of frames to extract 
returns None if not enough num frames

'''
def _randomlySelectFrames(irisTrackingPoints, frames, numFrames=200):
  validIndexes = np.array([i for i in range(irisTrackingPoints.shape[0]) if -1 not in irisTrackingPoints[i]])
  if(len(validIndexes) < numFrames):
    return None

  validIndexes = validIndexes[np.random.choice(validIndexes.shape[0], size=numFrames, replace=False).astype(np.uint32)]
  validIndexes.sort()
  return (irisTrackingPoints[validIndexes,:], frames[validIndexes])

'''
  Mostly a debugging tool, outputs a video where each frame is annotated with the criterion
'''
def _animateCriterion(videoName, data, fps):
  criterion = _findMostForwardFace(data)
  fps = 2
  select = [i for i in range(len(criterion)) if criterion[i] > 50 ]
  annotation = np.array(criterion)[select]
  animatePoints.animateFromData(videoName, data[select,:], fps, annotation=annotation)

'''
Assumes 'data' is not homogenized 
'''
def perspectiveTransform(data, h):
  data = np.concatenate((data, np.ones((data.shape[0],1))), axis=1)
  proj = h.dot(data.T) # result is unormalized and transposed
  proj = (proj/proj[2,:]).T
  return proj[:,:2]

def alignAllToHomography(referenceFace, data, irisPts):
  def _getPointsForHomography(face):
    relevantPoints = np.array([0,1,2,3,36,39,27,28,29,30,42,45,14,15,16])
    return face[relevantPoints,:]

  referenceFaceHPoints = _getPointsForHomography(referenceFace)
  alignedFaces = np.zeros(data.shape)
  alignedIris = np.zeros(irisPts.shape)
  for i in range(data.shape[0]):
    face = data[i,:]
    faceHpoints = _getPointsForHomography(face)
    h = cv2.findHomography(faceHpoints, referenceFaceHPoints)[0]
    alignedFaces[i,:,:] =  perspectiveTransform(face, h)
    alignedIris[i,:] = perspectiveTransform(irisPts[i,:][np.newaxis, :], h)
  return alignedIris, alignedFaces

'''
Generates a histogram for all _iris.p it finds in the file
assumes that path is a directory with both the .p and the _iris.p files
'''
def forAllInDir(path): 
  i = 0
  numSuccess = 0
  listing = os.listdir(path)
  for f in listing:
     if f.endswith(".p") and not(f.endswith("_iris.p")):
       print('So far outputed {0} files'.format(numSuccess))
       print(f)
       print(f[:-2])
       if not(f[:-2] + "_iris.p" in listing):
         print('Skipping processing {0}, no corresponding _iris file'.format(f))
         continue 
       if (f[:-2] + IRIS_HIST_ENDING in listing):
         print('Skipping processing {0}, output already created'.format(f))
         continue

       irisFile = os.path.join(path, f[:-2] + "_iris.p")
       featureFile = os.path.join(path, f)

       iris = pickle.load(open( irisFile, "rb" ))
       irisData = iris['data'][:,:,:2] # don't care about radius
       if(leftEye):
         irisData = irisData[:,0,:]
       else:
         irisData = irisData[:,1,:]

       features = pickle.load(open( featureFile, "rb" ))
       featureData = features['data']
       fps = features['fps']
       assert(np.abs(fps - iris['fps']) < .001)

       # find reference face   
       referenceFace = _findMostForwardFace(featureData)
 
       # start picking iris data frames we wantn to plot
       selected =  _randomlySelectFrames(irisData, iris['frames'], numFrames=170)
       if(selected is None): # not enough valid frames 
         continue
       irisTrackingPoints, irisFrames = selected
       assert(features.get('frames') is not None)
       featureFrames = features['frames']
       featureFrameIndexes = [i for i in range(featureFrames.shape[0]) if featureFrames[i] in irisFrames]
       featureDataSelected = featureData[featureFrameIndexes]
       assert(featureDataSelected.shape[0] == irisTrackingPoints.shape[0])
       # animatePoints.animateFromData(featureFile[:-2] + '_bothSelected.mp4', \
       #   np.concatenate((featureDataSelected, irisTrackingPoints[:,np.newaxis,:2]), axis=1),\
       #   fps, None)

       # align the iris frames to the reference face
       alignedIrisPts, alignedFaces = alignAllToHomography(referenceFace, featureDataSelected, irisTrackingPoints)
       print('aligned')
       # animatePoints.animateFromData(featureFile[:-2] + '_bothAligned.mp4', \
       #  np.concatenate((alignedFaces, alignedIrisPts[:,np.newaxis,:]), axis=1),3, None)


       # generate Histogram
       hist = Historgram(10,14) # choosing 10 arbitrarily widthPartitions, hightPartitions
       hist.insertEyeDataCollectionToHistogram(alignedIrisPts, alignedFaces, eyeLeft=True)
       print hist.data

       histFileName = featureFile[:-2] + IRIS_HIST_ENDING 
       pickle.dump(hist.data,  open(histFileName, "wb" ) )
       numSuccess = numSuccess + 1 

if __name__ == '__main__':
  if len(sys.argv) != 2:
      print(
       "Expects 1 arguments:"+
       "Path to file with .p and _iris.p"
      )
      exit()

  leftEye = True # which eye we're creating the histogram for 
  path = sys.argv[1]
  forAllInDir(path)
