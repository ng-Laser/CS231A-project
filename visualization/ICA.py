import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import pickle 
from sklearn.decomposition import FastICA
from animatePoints import animateFromData

def ica(data):
  # center data 
  meanFace = np.sum(data, axis=0)/data.shape[0]
  data = data - meanFace
  dataReshaped = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
  # Compute ICA
  ica = FastICA(n_components=NUM_COMPONENTS)
  S_ = ica.fit_transform(dataReshaped)  # Reconstruct signals
  A_ = ica.mixing_  # Get estimated mixing matrix
  print("S shape {0}".format(S_.shape))
  print("A shape {0}".format(A_.shape))
  return (S_, A_, meanFace)


def visualizeICAComp(S_ , A_, meanFace, componentNum, outputFileNamePre, fps):
  time_series = S_[:, componentNum][:,np.newaxis].dot(A_[:,componentNum][np.newaxis,:])
  newShape = (time_series.shape[0], time_series.shape[1]/2,2)
  time_series = np.reshape(time_series, newShape)
  print("time series shape {0}".format(time_series.shape))
  time_series = time_series + meanFace
  print("meanFace shape {0}".format(meanFace.shape))

  fps = a['fps']
  videoName = outputFileNamePre + "_comp_" + str(componentNum) + '.mp4'
  animateFromData(videoName, time_series, fps)
  

if __name__ == '__main__':
  NUM_COMPONENTS = 5
  '''
  if len(sys.argv) != 2:
      print(
          "Expects one argument which is the path to a .p file"
          " containing object with fields 'data':nxpx2 matrix, and 'fps' - frames "
          " n is the number of frames, p is the number of points and 2 is x,y"
          )
      exit()
  fileName = sys.argv[1] 
  ''' 
  fileName = '/home/noa_glaser/data/train-5-6/extractedFacialFeatures/Yj36y7ELRZE.000.p' 

  a = pickle.load(open( fileName, "rb" )) 
  data = a['data']
  S_, A_, meanFace = ica(data)
  for i in range(S_.shape[1]):
    visualizeICAComp(S_ , A_, meanFace, i, fileName[:-2] + 'test5', a['fps'])
  
  # newFileName = fileName[:-2] + '_decomposed.p'
  # print(newFileName)
  # pickle.dump(extracted,  open( newFileName , "wb" ) )
  
