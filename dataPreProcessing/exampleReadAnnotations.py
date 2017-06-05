'''
Format of annotation_training.pkl
{
extraversion: {
  videoName1: value1,
  videoName2: value2,
  etc...
},
neuroticism:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
agreeableness:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
conscientiousness:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
interview:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
openness:{
  videoName1: value1,
  videoName2: value2,
  etc...
}
}
''' 
# Example python blurb to read annotations:

import pickle

dict = pickle.load( open( "annotation_training.pkl" ) )
print dict['extraversion'][dict['extraversion'].keys()[0]]

