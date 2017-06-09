import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn

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

def getStandardDev(allHistograms):
  return [np.std(allHistograms[i,:]) for i in range(allHistograms.shape[0])]

def fitToData(allHistograms, trait):
  regr = sklearn.linear_model.LinearRegression()
  sz = allHistograms.shape
  regr.fit(flat, trait)
  ypred = flat.dot(regr.coef_) + regr.intercept_
  r_Sqrd = sklearn.metrics.r2_score(extraversion, ypred)
  return regr, r_Sqrd

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
  neuroticism =  [annotations['neuroticism'][m] for m in movieNames]
  agreeableness = [annotations['agreeableness'][m] for m in movieNames]
  conscientiousness = [annotations['conscientiousness'][m] for m in movieNames]
  openness = [annotations['openness'][m] for m in movieNames]

  # std = getStandardDev(allHistograms) 
  # numSubFigWidth = 5 
  # numSubFigHeight = 1

  # fig_size = plt.rcParams["figure.figsize"]
  # # Set figure width to 12 and height to 9
  # fig_size[0] = numSubFigWidth*5
  # fig_size[1] = numSubFigHeight*5

  # print('Now Plotting')
  # plt.subplot(numSubFigHeight, numSubFigWidth, 1)
  # # plt.hist(extraversion)
  # plt.plot(std, extraversion, 'o')
  # plt.subplot(numSubFigHeight, numSubFigWidth, 2)
  # # plt.hist(neuroticism)
  # plt.plot(std, neuroticism, 'o')
  # plt.subplot(numSubFigHeight, numSubFigWidth, 3)
  # # plt.hist(agreeableness)
  # plt.plot(std, agreeableness, 'o')
  # plt.subplot(numSubFigHeight, numSubFigWidth, 4)
  # # plt.hist(conscientiousness)
  # plt.plot(std, conscientiousness, 'o')
  # plt.subplot(numSubFigHeight, numSubFigWidth, 5)
  # # plt.hist(openness)
  # plt.plot(std, openness, 'o')

  # plt.tight_layout()

  # # plt.show()
  # plt.savefig('personalityHistograms.png')
  # # print dict['extraversion'][dict['extraversion'].keys()[0]]
  
