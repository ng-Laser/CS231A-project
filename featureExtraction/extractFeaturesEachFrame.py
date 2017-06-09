'''
Code copied from
http://dlib.net/face_landmark_detection.py.html
'''
# imports from frame extraction
# import cv2
import imageio
import numpy as np
import scipy
from  scipy import misc

# imports from dlib test 
import sys
import os
import dlib

import pickle


def _printOutError(videoName, message):
  print('For video {0}: {1}'.format(videoName, message))

# attempts to 
# returns an object with fields
#     data = array of nx68x2
#     fps = real fps achieved
def extractFeaturesForEachFrame(videoPath):
  desiredFPS = 20 # in frames per second
 
  # start of model initial set up 
  predictor_path = '/home/noa_glaser/CS231A-project/featureExtraction/shape_predictor_68_face_landmarks.dat'
  print(os.path.exists(predictor_path))
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(predictor_path)

  # open video/ get metadata
  try:
    vid = imageio.get_reader(videoPath,  'ffmpeg')

  except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    _printOutError(videoPath,'Could not open video')
    return None
  meta_data = vid.get_meta_data()
  numframes = meta_data["nframes"]
  MAX_CORRUPT_FRAMES = numframes*.1
  fps = meta_data["fps"]
  print(fps)
  ''' Example of metadata
   {'ffmpeg_version '<>', 'plugin': '<>', 'source_size': (1280, 720), 
   'nframes': 460, 'fps': 30.0, 'duration': 15.32, 'size': (1280, 720)}
  '''
  if(fps < desiredFPS):
    _printOutError(videoPath,'Frame rate of original video too low')
    return None
  # assert(fps >= desiredFPS)
  numFrames = int(meta_data['duration']*desiredFPS)
  # sampleEveryN = int(fps/desiredFPS)
  
  ptsFromFrames = np.array([]) # will eventually be a nx68x2 array where 
                               # n is the number of frames extracted, as per desiredFPS
                               # the 68 dim corresponds the the 68 feature points
                               # the 2 dim corresponds x,y 
  # for f in range(0,numframes, sampleEveryN):
  corruptFrames = 0
  frames = np.zeros(numFrames)
  for f in range(numFrames):
  # for f in range(10):
     frame = int((f*fps)/desiredFPS)
     print(frame)
     image = vid.get_data(frame)
     # Ask the detector to find the bounding boxes of each face. The 1 in the
     # second argument indicates that we should upsample the image 1 time. This
     # will make everything bigger and allow us to detect more faces.
     dets = detector(image, 1)
     if(len(dets) != 1):
       _printOutError(videoPath,'Didn\'t find a face on frame {0}'.format(int((f*fps)/desiredFPS)))
       corruptFrames = corruptFrames + 1 
       if corruptFrames > MAX_CORRUPT_FRAMES:
         return None
       else: 
         continue
     # assert(len(dets) == 1) # detected exactly one face 
     # TODO: figure out better way to handle above 
     shape = predictor(image, dets[0])
     try:
       shape.part(67) # all the parts are there , TODO : actually improve
     except:
       _printOutError(videoPath,'Doesn\'t contain all the parts, skipping')
       corruptFrames = corruptFrames + 1 
       if corruptFrames > MAX_CORRUPT_FRAMES:
         return None
       else: 
         continue

     xy = [[shape.part(i).x, shape.part(i).y] for i in range(68) ]
     xy = np.array(xy)[np.newaxis,:,:]
     if(xy.shape[1] != 68 or xy.shape[2] != 2):
       _printOutError(videoPath,'Frame {0} has wrong number of components'.format(int((f*fps)/desiredFPS)))
       corruptFrames = corruptFrames + 1 
       if corruptFrames > MAX_CORRUPT_FRAMES:
         return None
       else: 
         continue

     frames[f-corruptFrames] = frame
     if ptsFromFrames.size == 0:
      ptsFromFrames =  xy
     else:
       ptsFromFrames =  np.concatenate((ptsFromFrames, xy), axis=0) 
  result = {
   'data': ptsFromFrames,
   'frames': frames[:ptsFromFrames.shape[0]], 
   'fps': ptsFromFrames.shape[0]/meta_data['duration']
  }
  return result

def forAllFilesInDir(pathSource, pathDest):
    i = 0
    numSuccess = 0
    listDirSet = set(os.listdir(pathDest))
    for f in os.listdir(pathSource):
        if f.endswith(".mp4"):
            outFileName = os.path.join(pathDest, f[:-4]) + ".p"  # excluding .mp4 endign
            if outFileName in listDirSet:
               _printOutError(outFileName,'Skipping because output exists')
               continue 
            print(f) # print which video we are processing
            vidName = os.path.join(pathSource, f)
            print(vidName)
            extracted = extractFeaturesForEachFrame(vidName)
            if(extracted is None):
               continue 

            pickle.dump(extracted,  open( outFileName , "wb" ) )
            numSuccess = numSuccess + 1
            print('So far outputed {0} files'.format(numSuccess))
            # np.save(extracedFileName , extracted)

if __name__ == '__main__':
  # note/TODO: for visualization could also use the dlib visualizer?
  # TODO: later could consider saving in memory a black list of files so we know not to waste time in trying those again
  if len(sys.argv) != 3:
      print(
       "Expecting 2 arguments:\n" + 
        "First arg: Path to directory containing videos to extract features from\n" +  
        "Secondn arg:  Path to output to"
      )
      exit()
  
  forAllFilesInDir(sys.argv[1], sys.argv[2])
