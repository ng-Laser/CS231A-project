import numpy as np
class Historgram():
  widthPartitions = 0
  hightPartitions = 0
  data = None
  assumeEyeHeightWidthRatio = 0
 
  SIDEWAYS_BUFFER = 10 
  UP_DOWN_BUFFER = 5 
  '''
  assumeEyeHalfHeightWidthRatio conservative ration of eye height radius to width
  '''
  def __init__(self, widthPartitions, hightPartitions, assumeEyeHeightWidthRatio=.6):
     self.widthPartitions = widthPartitions 
     self.hightPartitions = hightPartitions 
     self.data = np.zeros((self.hightPartitions,  self.widthPartitions)) 
     self.assumeEyeHeightWidthRatio = assumeEyeHeightWidthRatio

  '''
  on left eye, left corner is indx 36, right corner is 39
  note that eye direction is with respect that it apears in the image,
   not anatomically in the person
  '''
  def insertEyeDataToHistorgram(self,irisXY, facialFeatures, eyeLeft=True):
    eyeWidth = facialFeatures[39,0] - facialFeatures[36,0]
    eyeHeight = self.assumeEyeHeightWidthRatio * eyeWidth
    if(not eyeLeft):
      eyeWidth = facialFeatures[45,0] - facialFeatures[42,0]
    eyeWidth =  eyeWidth  + 2*self.SIDEWAYS_BUFFER
    eyeHeight = eyeHeight + 2*self.UP_DOWN_BUFFER

    eyeCornerHightDiff =  facialFeatures[36,1] - facialFeatures[39,1]
    topLeftCorner = facialFeatures[36,:] - np.array([0,eyeHeight/2]) + eyeCornerHightDiff/2
    irisInBox = irisXY - topLeftCorner

    gridCellX  =  int((self.widthPartitions*irisInBox[0])/eyeWidth)
    gridCellY  =  int((self.hightPartitions*irisInBox[1])/eyeHeight)

    def capBounds(val, lower, upper):
      if val < lower:
         return lower
      if val > upper:
         return upper
      return val 

    gridCellX = capBounds(gridCellX, 0, self.widthPartitions -1)
    gridCellY = capBounds(gridCellY, 0, self.hightPartitions -1)
    self.data[gridCellY, gridCellX] = self.data[gridCellY, gridCellX] + 1

  def insertEyeDataCollectionToHistogram(self,irisData, facialFeatureData, eyeLeft=True):
    # eye is rotated in face, slightly towards nose (especially in women)
    # Need to rotate so that left, right points are straight
    # tested this out on people, no need Half

    for i in range(irisData.shape[0]):
      self.insertEyeDataToHistorgram(irisData[i,:2], facialFeatureData[i,:], eyeLeft)
