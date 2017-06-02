import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# expects as an argument a .npy file with nxpx2 data
# where n is the number of frames, p is the number of points and 2 is x,y


def update_line(num, data, line):
    # line.set_data(data[..., :num])
    # return np.random.randint(1, 30, 30), np.random.randint(1, 30, 30)
    line.set_data(data[num, :,0], data[num,:,1])
    return line,

if __name__ == '__main__':
  if len(sys.argv) != 2:
      print(
          "Expects one argument which is the path to a .npy file"
          " containing nxpx2 data"
          " n is the number of frames, p is the number of points and 2 is x,y"
          )
      exit()
  fig1 = plt.figure()
  
  data = np.load(sys.argv[1]) 
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

  plt.xlabel('x')
  plt.title('y')

  # for plotting, y at top left corner is 0 
  
  line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                     interval=50, blit=True)
  
  line_ani.save('testVideo.mp4', fps=10)
  # plt.show()
