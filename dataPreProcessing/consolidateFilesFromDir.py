import sys 
import os
import string 

def moveByPrefix(sourceDir, destDir):
  def getPrefix(str):
   parts = str.split('.')
   return string.join(parts[:-1],'.')

  prefixes = [getPrefix(f) for f in os.listdir(destDir)]
  filesToMove = [f for f in os.listdir(sourceDir) if getPrefix(f) in prefixes]
  print('prefixes {0}'.format(prefixes))
  print('files to move {0}'.format(filesToMove))
  for f in filesToMove:
     originalName = os.path.join(sourceDir, f) 
     newName = os.path.join(destDir, f)
     print('moving {0} to {1}'.format(originalName, newName)) 
     os.system('mv ' +  originalName + ' ' + newName)
   
if __name__ == '__main__': 
  if len(sys.argv) != 3:
      print(
        "Assumes that destDir has files of a specific ending, i.e. .p" +
        " and want to merge with files of a differnet file ending, i.e. .mp4\n" + 
        "Expecting 2 arguments:\n" + 
        "Path to directory sourceDir\n"
        "Path to directory destDir\n"
      )
      exit()
  sourceDir = sys.argv[1]
  destDir = sys.argv[2]
  moveByPrefix(sourceDir, destDir)
  
