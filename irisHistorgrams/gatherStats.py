import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model 
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

def getPercentile(allHistograms,q):
  return np.array([np.percentile(allHistograms[i,:], q) for i in range(allHistograms.shape[0])])

def getSparcity(allHistograms, cuttoff):
  numFrames = np.sum(allHistograms[0,:])
  counters = np.zeros(allHistograms.shape)
  counters[allHistograms > numFrames*cuttoff] = 1
  return np.sum(counters, axis=(1,2))

def fitToData(flat, trait):
  print('flat shape {0}'.format(flat.shape))
  print('trait shape {0}'.format(trait.shape))

  regr = linear_model.LinearRegression()
  regr.fit(flat, trait)
  ypred = flat.dot(regr.coef_) + regr.intercept_
  r_Sqrd = sklearn.metrics.r2_score(trait, ypred)
  return regr, r_Sqrd

def splitIntoTestValidation(allHistograms,scores, ratio):
  testIndexes = np.random.choice(scores.shape[0], size=int(scores.shape[0]*ratio), replace=False).astype(np.uint32)
  valIndexes = np.array([i for i in range(scores.shape[0]) if i not in testIndexes])
  # would probably be much faster to randomize order then split but whatever 
  return (allHistograms[testIndexes], scores[testIndexes],allHistograms[valIndexes], scores[valIndexes])
  
def getL1(reg, xTest, yTest):
  ypred = xTest.dot(reg.coef_) + reg.intercept_
  return np.mean(np.abs(ypred - yTest))

def subsample(allHistograms, h_factor=2, w_factor=2):
  sz = allHistograms.shape
  subsampled = np.zeros((sz[0], int(sz[1]/h_factor), int(sz[2]/w_factor)))
  for i in range(subsampled.shape[1]):
    for j in range(subsampled.shape[2]):
      subsampled[:,i,j] = np.sum(
          allHistograms[:,i*h_factor:(i+1)*h_factor, j*w_factor:(j+1)*w_factor],  axis=(1,2))
  return subsampled 

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

  # subsampled = subsample(allHistograms)
  # this was a bad idea 

  # another idea, look for the center of mass, weights around the center of mass 
  # people's webcams may be different
 
  # can also do a measure of sparcity

  extraversion =      np.array([annotations['extraversion'][m] for m in movieNames])
  neuroticism =       np.array([annotations['neuroticism'][m] for m in movieNames])
  agreeableness =     np.array([annotations['agreeableness'][m] for m in movieNames])
  conscientiousness = np.array([annotations['conscientiousness'][m] for m in movieNames])
  openness =          np.array([annotations['openness'][m] for m in movieNames])

  scores = np.concatenate((extraversion[:,np.newaxis],
                           neuroticism[:,np.newaxis],
                           agreeableness[:,np.newaxis],
                           conscientiousness[:,np.newaxis],
                           openness[:,np.newaxis]
    ), axis=1)

  allHistogramsTrain, scoresTrain,allHistogramsVal, scoresVal = splitIntoTestValidation(allHistograms,scores, .8) 

  sz = allHistogramsTrain.shape
  flatTrain = allHistogramsTrain.reshape(sz[0], sz[1]*sz[2])
  print(flatTrain.shape) 
  print(scoresTrain.shape) 
  sz = allHistogramsVal.shape
  flatVal  = allHistogramsVal.reshape(sz[0], sz[1]*sz[2])

   # denoising attempts
  histSum = np.sum(allHistograms, axis=0)
  sumFlat = histSum.reshape(histSum.size)
  totalSum = np.sum(sumFlat)
  coeffs = [i for i in range(sumFlat.shape[0]) if (sumFlat[i]/totalSum > .007)] #  and (sumFlat[i]/totalSum < .70)]
  '''
  E L1 0.109213140772
  N L1 0.108951022737
  A L1 0.0951383417182
  C L1 0.110011570983
  O L1 0.101678775147
  '''
  # coeffs = [i for i in range(flatTrain.shape[1])]
  print('coeffs {0}'.format(coeffs))
  regrE, r_SqrdE = fitToData(flatTrain[:, coeffs], scoresTrain[:,0])
  regrN, r_SqrdN = fitToData(flatTrain[:, coeffs], scoresTrain[:,1]) 
  regrA, r_SqrdA = fitToData(flatTrain[:, coeffs], scoresTrain[:,2]) 
  regrC, r_SqrdC = fitToData(flatTrain[:, coeffs], scoresTrain[:,3]) 
  regrO, r_SqrdO = fitToData(flatTrain[:, coeffs], scoresTrain[:,4]) 
  
  print('E L1 {0}'.format(getL1(regrE,flatVal[:, coeffs] ,scoresVal[:,0])))
  print('N L1 {0}'.format(getL1(regrN,flatVal[:, coeffs] ,scoresVal[:,1])))
  print('A L1 {0}'.format(getL1(regrA,flatVal[:, coeffs] ,scoresVal[:,2])))
  print('C L1 {0}'.format(getL1(regrC,flatVal[:, coeffs] ,scoresVal[:,3])))
  print('O L1 {0}'.format(getL1(regrO,flatVal[:, coeffs] ,scoresVal[:,4])))

   # std = getStandardDev(allHistograms) 

  print('r squared E {0}'.format(r_SqrdE))
  print('r squared N {0}'.format(r_SqrdN))
  print('r squared A {0}'.format(r_SqrdA))
  print('r squared C {0}'.format(r_SqrdC))
  print('r squared O {0}'.format(r_SqrdO))
  # '''
  # vanila
  # 0.237827361139
  # 0.0419504486882
  # -0.101826399973
  # -0.0747322889778
  # -0.276171921602
  # '''

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
  # sparcity  = getPercentile(allHistograms, .83)[:,np.newaxis]
  '''
  Sparcity as percentile .9
  E L1 0.116232178962
  N L1 0.117049106848
  A L1 0.098086492792
  C L1 0.121762390368
  O L1 0.11006841166
  ''' 
  '''
  Sparcity as percentile .83
  E L1 0.106532728195
  N L1 0.10814030587
  A L1 0.0865272938444
  C L1 0.119689120168
  O L1 0.112574635909
  '''
  # sparcity  = getSparcity(allHistograms, .15)[:,np.newaxis]
  # sparcityTrain, scoresTrain,sparcityVal,scoresVal = splitIntoTestValidation(sparcity,scores, .8) 
  # print('hello')
  # print(allHistograms.shape)
  # print(sparcity.shape)
  # print(sparcityTrain.shape)
  # print(scoresTrain.shape)

  # regrE, r_SqrdE = fitToData(sparcityTrain, scoresTrain[:,0])
  # regrN, r_SqrdN = fitToData(sparcityTrain, scoresTrain[:,1]) 
  # regrA, r_SqrdA = fitToData(sparcityTrain, scoresTrain[:,2]) 
  # regrC, r_SqrdC = fitToData(sparcityTrain, scoresTrain[:,3]) 
  # regrO, r_SqrdO = fitToData(sparcityTrain, scoresTrain[:,4]) 
  # 
  # print('E L1 {0}'.format(getL1(regrE,sparcityVal ,scoresVal[:,0])))
  # print('N L1 {0}'.format(getL1(regrN,sparcityVal ,scoresVal[:,1])))
  # print('A L1 {0}'.format(getL1(regrA,sparcityVal ,scoresVal[:,2])))
  # print('C L1 {0}'.format(getL1(regrC,sparcityVal ,scoresVal[:,3])))
  # print('O L1 {0}'.format(getL1(regrO,sparcityVal ,scoresVal[:,4])))


  # plt.show()
  # plt.savefig('personalityHistograms.png')
  # print dict['extraversion'][dict['extraversion'].keys()[0]]
  
