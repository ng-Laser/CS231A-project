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


def extractFeaturesForEachFrame(videoPath):
  desiredFPS = 10 # in frames per second

  # start of model initial set up 
  predictor_path = '/home/noa_glaser/CS231A-project/featureExtraction/shape_predictor_68_face_landmarks.dat'
  print(os.path.exists(predictor_path))
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(predictor_path)

  # open video/ get metadata
  vid = imageio.get_reader(videoPath,  'ffmpeg')

  meta_data = vid.get_meta_data()
  numframes = meta_data["nframes"]
  fps = meta_data["fps"]
  print(fps)
  ''' Example of metadata
   {'ffmpeg_version '<>', 'plugin': '<>', 'source_size': (1280, 720), 
   'nframes': 460, 'fps': 30.0, 'duration': 15.32, 'size': (1280, 720)}
  '''
  assert(fps >= desiredFPS)
  sampleEveryN = int(fps/desiredFPS)
  
  ptsFromFrames = np.array([]) # will eventually be a nx68x2 array where 
                               # n is the number of frames extracted, as per desiredFPS
                               # the 68 dim corresponds the the 68 feature points
                               # the 2 dim corresponds x,y 
  # for i in range(0,numframes, sampleEveryN):
  for f in range(0,100, sampleEveryN):
     print(f)
     image = vid.get_data(f)
     # Ask the detector to find the bounding boxes of each face. The 1 in the
     # second argument indicates that we should upsample the image 1 time. This
     # will make everything bigger and allow us to detect more faces.
     dets = detector(image, 1)
     assert(len(dets) == 1) # detected exactly one face 
     # TODO: figure out better way to handle above 
     shape = predictor(image, dets[0])
     xy = [[shape.part(i).x, shape.part(i).y] for i in range(68) ]
     if f == 0:
      print('hello') 
      ptsFromFrames =  np.array(xy)[np.newaxis, :,:]
     else:
       ptsFromFrames =  np.concatenate((ptsFromFrames, np.array(xy)[np.newaxis, :,:]), axis=0) 
  return ptsFromFrames

if __name__ == '__main__':
  # note/TODO: for visualization could also use the dlib visualizer?
  if len(sys.argv) != 2:
      print(
          "Give the path to the trained shape predictor model as the first "
          "argument and then the directory containing the facial images.\n"
          "For example, if you are in the python_examples folder then "
          "execute this program by running:\n"
          "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
          )
      exit()
  
  extracted = extractFeaturesForEachFrame(sys.argv[1])
  extracedFileName = 'outFile'
  np.save(extracedFileName , extracted)
  print(extracted)

