import numpy as np
class Historgram():
  widthPartitions = 0
  hightPartitions = 0
  data = None
  assumeEyeHalfHeight = 0
 
  SIDEWAYS_BUFFER = 20 
   
  def __init__(self, widthPartitions, hightPartitions, assumeEyeHalfHeight=50):
     self.widthPartitions = widthPartitions 
     self.hightPartitions = hightPartitions 
     self.data = np.zeros((self.widthPartitions,  self.hightPartitions)) 
     self.assumeEyeHalfHeight = assumeEyeHalfHeight

  '''
  on left eye, left corner is indx 36, right corner is 39
  note that eye direction is with respect that it apears in the image,
   not anatomically in the person
  '''
  def insertEyeDataToHistorgram(self,irisXY, facialFeatures, eyeLeft=True):
    eyeHeight = self.assumeEyeHalfHeight * 2
    eyeWidth = facialFeatures[39,0] - facialFeatures[36,0]
    print( facialFeatures[39,0])
    print(facialFeatures[36,0])
    if(not eyeLeft):
      eyeWidth = facialFeatures[45,0] - facialFeatures[42,0]
    # eyeWidth = eyeWidth + 2*SIDEWAYS_BUFFER
    topLeftCorner = facialFeatures[36,:] - np.array([0,self.assumeEyeHalfHeight])
    irisInBox = irisXY - topLeftCorner
    print('irisInBox {0}'.format(irisInBox))
    print('eyeWidth {0}'.format(eyeWidth))

    gridCellX  =  (self.widthPartitions*irisInBox[0])//eyeWidth 
    gridCellY  =  (self.hightPartitions*irisInBox[1])//eyeHeight
    print(gridCellX)
    print(gridCellY)

    self.data[gridCellY, gridCellX] = self.data[gridCellY, gridCellX] + 1

  def insertEyeDataCollectionToHistogram(self,irisData, facialFeatureData, eyeLeft=True):
    for i in range(irisData.shape[0]):
      self.insertEyeDataToHistorgram(irisData[i,:2], facialFeatureData[i,:], eyeLeft)
