import cv2
import os
import sys
import imageio
import numpy as np
import scipy
from  scipy import misc
'''
VideoCapture variables of interest
    CV_CAP_PROP_FPS Frame rate.
    CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
    CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
'''

'''
More videocapture documentation at
http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
'''


'''
a function that captures numCaptureFPS Frames per second
from a video given by vidCap - a cv2.VideoCapture object
'''
'''
def getfps_windows(vidname, numcapturefps, saveto='./'):
    vidcap = cv2.videocapture(vidname)
    filenmtemplate = os.path.join(saveto, 'frame%d.jpg')

    numframes = vidcap.get(cv2.cap_prop_frame_count)
    print('numframes {0}'.format(numframes))
    videofps = vidcap.get(cv2.cap_prop_fps)
    print('videofps {0}'.format(videofps))
    videolength = numframes /videofps
    print('videolength {0}'.format(videolength))
    vidcap.set(cv2.cap_prop_pos_frames, 0)

    # todo: remove line below
    videolength = 1 # just for now for debugging
    for i in range(videolength*numcapturefps):
       curframe = i*videofps/numcapturefps
       vidcap.set(cv2.cap_prop_pos_frames, curframe)
       success,image = vidcap.read()
       print 'read a new frame: ', success
       cv2.imwrite(filenmtemplate % i, image)     # save frame as jpeg file
'''



'''
a function that captures numCaptureFPS Frames per second
from a video given by vidCap - a cv2.VideoCapture object
created after it looks like VideoCapture doesn't work on ubuntu
'''
'''
# math.floor def getfps_ubuntu(vidname, numcapturefps, saveto='./'):
def getfps_ubuntu(vidname, numcapturefps, saveto):
    vid = imageio.get_reader(vidname,  'ffmpeg')
    filenmtemplate = os.path.join(saveto, 'frame%d.jpg')
 
    meta_data = vid.get_meta_data()
    numframes = meta_data["nframes"]
    print 'numframes ' + str(numframes)
    videofps = meta_data["fps"]
    print 'videofps ' + str(videofps)
    videolength = meta_data["duration"]
    print 'videolength ' + str(videolength)

    for i in range(int(videolength*numcapturefps)):
        curframe = i*int(videofps/numcapturefps)
        image = vid.get_data(curframe)
        imageio.imwrite(filenmtemplate % i, image) # save frame as jpeg file
'''

'''
a function that captures numCaptureFPS Frames per second
from a video given by vidCap - a cv2.VideoCapture object
created after it looks like VideoCapture doesn't work on ubuntu
'''
# math.floor def getfps_ubuntu(vidname, numcapturefps, saveto='./'):
def getRegSpacing_ubuntu(vidname, numcapture, saveto):
    vid = imageio.get_reader(vidname,  'ffmpeg')
    filenmtemplate = os.path.join(saveto, 'frame%d.jpg')
 
    meta_data = vid.get_meta_data()
    numframes = meta_data["nframes"]
    print('numframes '.format(numframes))

    for i in range(numcapture):
        partition_len = int(numframes/numcapture)
        randomIndx = np.random.randint(i*partition_len, (i+1)*partition_len )
        image = vid.get_data(randomIndx)
        # image = _resize_image(image, 256) 
        imageio.imwrite(filenmtemplate % i, image) # save frame as jpeg file

def _resize_image(img, smaller_dim):
    # another way: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
    img_shape_min = np.min(img.shape[0:2])
    ratio = (1.0*img_shape_min)/smaller_dim
    newSize = np.concatenate((np.array(img.shape[0:2])*(1/ratio), [3])).astype(int)
    return scipy.misc.imresize(img, newSize)
    
'''
def deleteMoviesWithFramesExtracted(dirName):
    dirListing = os.listdir(dirName)
    dirListing.sort()
    print dirListing[0:50]
    for i in range(len(dirListing)):
        if dirListing[i].endswith(".mp4"):
            # print 'dirListing ' + dirListing[i] + ' prev ' + dirListing[i-1] 
            if dirListing[i + 1] == dirListing[i][:-4] + "_10uniform":
                print'deleting' +  dirListing[i]
                os.system('rm ./' +  dirListing[i])
'''


'''
Performes the specified function on all mp4 files in a directory specified by path
Right now there is only getFPS, an make function template later
'''
def doToAllMoviesInDir(path):
    i = 0
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            if(i%50 == 0):
                print(i)
            vidName = os.path.join(path, file)
            newFileName = os.path.join(path, file[:-4]) # excluding .mp4 endign
            newFileName = newFileName + '_50uniform' 
            print('newFileName {0}'.format(newFileName))
            os.system('sudo mkdir ' + newFileName)
            print('vidname {0}'.format(vidName))
            numcapture = 10
            getRegSpacing_ubuntu(vidName, numcapture, newFileName)
            # print 'deleting ' + vidName
            # os.system('rm ./' +  vidName)
            i = i + 1

if __name__ == "__main__":
    sourceFile = sys.argv[1]
    # print 'sourceFile ' + sourceFile
    doToAllMoviesInDir(sourceFile)
