import sys 
import os
import pickle

def forAllFilesInDir(dr):
    numDeleted = 0
    for f in os.listdir(dr):
      if not f.endswith('.p'):
         continue
      fileName = os.path.join(dr, f) 
      print(fileName)
      a = pickle.load(open(fileName, 'rb'))
      if(a['data'].shape[0] != a['frames'].shape[0]):
        print('Deleting {0}'.format(fileName))
        os.system('rm ' +  fileName)
        numDeleted = numDeleted + 1
    print('Num deleted {0}'.format(numDeleted))
 

if __name__ == '__main__': 
  if len(sys.argv) != 2:
      print(
       "Expecting 1 arguments:\n" + 
        "Path to directory with .p files"
      )
      exit()
  sourceFile = sys.argv[1]
  forAllFilesInDir(sourceFile)
