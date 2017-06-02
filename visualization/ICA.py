import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys

from sklearn.decomposition import FastICA
from animatePoints import animateFromData

def ica_file(fileName):
  a = pickle.load(open( fileName, "rb" )) 
  data = a['data']
  # center data 
  meanFace = np.sum(data, axis=0)
  data = data - meanFace
  dataReshaped = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
  # Compute ICA
  ica = FastICA(n_components=NUM_COMPONENTS)
  S_ = ica.fit_transform(data)  # Reconstruct signals
  A_ = ica.mixing_  # Get estimated mixing matrix
  return (S_, A_)


# def visualizeICAComp(S_ , componentNumber):


if __name__ == '__main__':
  NUM_COMPONENTS = 5

  if len(sys.argv) != 2:
      print(
          "Expects one argument which is the path to a .p file"
          " containing object with fields 'data':nxpx2 matrix, and 'fps' - frames "
          " n is the number of frames, p is the number of points and 2 is x,y"
          )
      exit()
  fileName = sys.argv[1] 
  fileName = 'data/train-5-6/extractedFacialFeatures/UD-8YYU7GZs.003.p' 

  
  newFileName = fileName[:-2] + '_decomposed.p'
  print(newFileName)
  # pickle.dump(extracted,  open( newFileName , "wb" ) )
  
