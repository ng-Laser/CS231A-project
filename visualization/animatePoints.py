import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import pickle
import os 

# expects as an argument a .npy file with nxpx2 data
# where n is the number of frames, p is the number of points and 2 is x,y

def update_line(num, data, line, annotationStuff):
    # line.set_data(data[..., :num])
    # return np.random.randint(1, 30, 30), np.random.randint(1, 30, 30)
    line.set_data(data[num, :,0], data[num,:,1])
    if(annotationStuff is not None):
      annotation, plt, annotationLocation = annotationStuff
      print(annotationLocation)
      print(annotation[num])
      plt.title(str(annotation[num]))
      # ax.annotate(s=str(annotation[num]), xy=annotationLocation,  xytext=annotationLocation)
    return line

def animateFromData(videoName, data, fps, annotation=None):
  fig1 = plt.figure()
  l, = plt.plot([], [], "o")
  pad = 5
  x_vars = data[:,:,0]
  x_vars = x_vars.reshape(x_vars.size)
  y_vars = data[:,:,1]
  y_vars = y_vars.reshape(y_vars.size)
  data[:, :, 1] = -data[:, :, 1]
  y_min, y_max = (np.min(y_vars) - pad, np.max(y_vars) + pad)
  plt.xlim(np.min(x_vars) - pad, np.max(x_vars) + pad)
  plt.ylim(y_min, y_max)


  annotationStuff = None
  if annotation is not None:
    ax = fig1.add_subplot(1,1,1)
    annotationLocation = (y_min + 30, np.min(x_vars)+50)
    annotationStuff = (annotation, plt, annotationLocation)
  line_ani = animation.FuncAnimation(fig1, update_line, data.shape[0], fargs=(data, l, annotationStuff),
                                     interval=(1000/fps))
  line_ani.save(videoName, fps=fps)


def extractFramesFromFile(fileName):
  a = pickle.load(open( fileName, "rb" )) 
  data = a['data']
  fps = a['fps']
  videoName = fileName[:-2] + '.mp4'
  animateFromData(videoName, data, fps)


def forAllFilesInDir(path):
    i = 0
    numSuccess = 0
    for f in os.listdir(path):
        if f.endswith(".p"):
            outFileName = os.path.join(path, f[:-2]) + ".p"  # excluding .mp4 endign
            # if(outFileName in os.listdir(path)): # later , probably make faster, like convert to map first
            #   continue 
            print(f) # print which video we are processing
            extractFramesFromFile(outFileName)


# Expects one argument which is the path to directory of .p files"
# containing object with fields 'data':nxpx2 matrix, and 'fps' - frames "
# n is the number of frames, p is the number of points and 2 is x,y"
# result is saving corresponding animations in current directory
if __name__ == '__main__':
  if len(sys.argv) != 2:
      print(
       "Expects one argument which is the path to directory of .p files"
       " containing object with fields 'data':nxpx2 matrix, and 'fps' - frames "
       " n is the number of frames, p is the number of points and 2 is x,y"
      )
      exit()
  forAllFilesInDir(sys.argv[1])
