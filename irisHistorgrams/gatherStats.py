import sys
import os
import numpy as np
import pickle


annotationsPath = '/home/noa_glaser/data/annotation_training.pkl'
IRIS_HIST_ENDING = '_irisHst.p'

def concatenateAllHistograms(filesToAnalyse):
  numWidthCells = 10 # assuming this for now
  numHeightCells = 14 # assuming this for now
  allData = np.zeros((len(filesToAnalyse), numHeightCells,numWidthCells))
  for i in range(len(filesToAnalyse)):
    f = filesToAnalyse[i] 
    hist = pickle.load(open(f, 'rb'))
    allData[i,:,:] = hist
  return allData

if __name__ == '__main__':
  if len(sys.argv) != 2:
      print(
       "Expects one arguments:" + 
       " Arg1 is directory containing _irishist.p files that you want analysed"
      )
      exit()

  annotations = pickle.load( open(annotationsPath ) )
  dataPath = sys.argv[1]
  filesToAnalyse = [f for f in os.listdir(dataPath) if f.endswith(IRIS_HIST_ENDING)]
  movieNames = [f[:-1*len(IRIS_HIST_ENDING)] + '.mp4' for f in filesToAnalyse]

  dataFileFullPath = [os.path.join(dataPath, f) for f in filesToAnalyse]
  allHistograms = concatenateAllHistograms(dataFileFullPath)
  allHistsFileName = os.path.join(dataPath, 'dirHistograms.p')
  # pickle.dump(allHistograms, open(allHistsFileName, 'wb'))

  extraversion = [annotations['extraversion'][m] for m in movieNames]
  neuroticism =  [annotations[''][m] for m in movieNames]
  agreeableness = [annotations['agreeableness'][m] for m in movieNames]
  conscientiousness = [annotations['conscientiousness'][m] for m in movieNames]
  openness = [annotations['openness'][m] for m in movieNames]


  # print dict['extraversion'][dict['extraversion'].keys()[0]]
  
