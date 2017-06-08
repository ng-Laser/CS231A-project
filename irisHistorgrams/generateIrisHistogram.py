from gridClass import Historgram
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../visualization")
import animatePoints
import pickle

'''
TODO as an extention,
align the face to absolute forward
1) use optimization to find the homography that maximizes
   the criterion for the first face
2) align to a prior that is the structure of the face
'''
# untested
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
   IrisTrackingData, the *_iris.p data including frames and the data 
   leftEye - bool, if true looks at left eye, if false looks at right eye 
      (w.r.t image not anatomically)
   numFrames, number of frames to extract 
'''
# untested
def _randomlySelectFrames(irisTrackingData, frames, leftEye=True, numFrames=200):
  irisTrackingPoints = None
  if(leftEye):
    irisTrackingPoints = irisTrackingData[:,0,:]
  else:
    irisTrackingPoints = irisTrackingData[:,1,:]
  validIndexes = np.array([i for i in range(irisTrackingData.shape[0]) if -1 not in irisTrackingData[i]])
  assert(len(validIndexes) >= numFrames)

  validIndexes = validIndexes[np.random.choice(validIndexes.shape[0], size=numFrames, replace=False).astype(np.uint32)]
  validIndexes.sort()
  irisTrackingPoints[validIndexes]
  frames[validIndexes]
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

def alignAllToHomography(referenceFace):
  pass 
 
if __name__ == '__main__':
  print 'hello' 
  if len(sys.argv) != 3:
      print(
       "Expects two arguments:"
       " Arg1 is featureFrame .p file"
       " Arg2 is _iris .p file"
      )
      exit()

  featureFile = sys.argv[1]
  features = pickle.load(open( featureFile, "rb" ))
  featureData = features['data']
  fps = features['fps']

  irisFile = sys.argv[2]
  iris = pickle.load(open( irisFile, "rb" ))
  assert(np.abs(fps - iris['fps']) < .001)
  irisData = iris['data'][:,:2] # don't care about radius
  
  # referenceFace = _findMostForwardFace(featureData)
  irisTrackingPoints, irisFrames =  _randomlySelectFrames(irisData, iris['frames'], leftEye=True, numFrames=170)
  
  assert(features.get('frames') is not None)
  featureFrames = features['frames']
  featureFrameIndexes = [i for i in range(featureFrames.shape[0]) if featureFrames[i] in irisFrames]
  featureDataSelected = featureData[featureFrameIndexes]

  assert(featureDataSelected.shape[0] == irisTrackingPoints.shape[0])
  print('featureDataSelected shape {0}'.format(featureDataSelected.shape))
  print('irisTrackingPoints shape {0}'.format(irisTrackingPoints.shape))

  # animatePoints.animateFromData(featureFile[:-2] + '_bothSelected.mp4', \
  #   np.concatenate((featureDataSelected, irisTrackingPoints[:,np.newaxis,:2]), axis=1),\
  #   fps, None)

  hist = Historgram(10,10) # choosing 10 arbitrarily
  hist.insertEyeDataCollectionToHistogram(irisTrackingPoints, featureDataSelected, eyeLeft=True)
  print hist.data

  histFileName = featureFile[:-2] + '_irisHst.p'
  pickle.dump(hist.data,  open(histFileName, "wb" ) )
