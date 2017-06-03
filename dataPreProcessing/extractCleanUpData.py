import sys 
import random
import os
import pickle

destFile = 'train-1/'
extractPrefix = 'training80'

def splitIntoTrainVal(directory, ratio_of_train):
  dirListing = os.listdir(directory)
  random.shuffle(dirListing)
  numTrain = int(len(dirListing) * ratio_of_train)
  
  trainFolder = os.path.join(directory, 'train')
  valFolder = os.path.join(directory,'val')
  os.system('mkdir ' +  trainFolder)
  os.system('mkdir ' +  valFolder)
  for f in dirListing[:numTrain]:
    print os.path.join(directory, f),os.path.join( trainFolder, f)
    os.rename(os.path.join(directory, f),os.path.join( trainFolder, f))
  for f in dirListing[numTrain:]:
    os.rename(os.path.join(directory, f),os.path.join( valFolder, f))

def splitNWay(directory, numBatches):
  dirListing = os.listdir(directory)
  random.shuffle(dirListing)
  sizeBatch = int(len(dirListing) /numBatches)

  for i in numBatches:   
    batchFolder = os.path.join(directory, 'batch-{0}'.format(i))
    os.system('mkdir ' +  batchFolder)
    for f in dirListing[sizeBatch*(i):sizeBatch*(i+1)]:
      print os.path.join(directory, f),os.path.join( batchFolder, f)
      os.rename(os.path.join(directory, f),os.path.join( batchFolder, f))


def resetNamesBack(dr):
  fileEnding ='_50uniform' #TODO: figure out how to make more general

  annotation_filename = dr + "/annotation_training.pkl"
  with open(annotation_filename, 'rb') as f:
        label_dicts = pickle.load(f) 

  for fileName in os.listdir(dr):
    fileName = fileName.replace(fileEnding,'.mp4')
    if(label_dicts.get(fileName)== None):
        print fileName, 'not in dictionary'
        fileNameCp  = fileName
	for i in range(3):
          fileNameCp  = '-' + fileNameCp
    	  if(label_dicts.get(fileNameCp) != None):
            print 'renaming', os.path.join(dr, fileName),os.path.join( dr, fileNameCp)
            # os.rename(os.path.join(dr, fileName),os.path.join( dr, fileNameCp))
            pass 

def extractFromZipMove():
  destFile = 'train-1/'
  extractPrefix = 'train80'
  for fileName in os.listdir(path):
    if fileName.startswith(extractPrefix):
      os.system('unzip ' + fileName)

def unzipAllFlsModNameMove():
  for fileName in os.listdir('./'):
   if fileName.startswith(extractPrefix):
     os.system('unzip ' + fileName)

  for fileName in os.listdir('./'):
    if(fileName.startswith('train-')):
    	continue
    # os.system('mv ./' + fileName + ' ' + destFile + fileName)


if __name__ == '__main__': 
  sourceFile = sys.argv[1]
  splitIntoTrainVal( sourceFile, .5)

