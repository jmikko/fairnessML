import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm
import matplotlib.patches as mpatches


def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

def plot_countour(plt, x,y,z, colormap, label):
    # define grid.
    xi = np.linspace(-5.1, 5.1, 1000)
    yi = np.linspace(-5.1, 5.1, 1000)
    # grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels, label=label)
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=colormap, levels=levels, label=label)



# make up some randomly distributed data
seed(1234)
npts = 5000
x = uniform(-10, 10, npts)
y = uniform(-10, 10, npts)

MINI_SIZE = 20
SMALL_SIZE = 25
MEDIUM_SIZE = 25
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MINI_SIZE)   # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

plt.figure(1)


colormap = 'Blues'
label = 'Group a, Label +1'
z = gauss(x, y, Sigma=np.asarray([[.8, .0], [.0, .8]]), mu=np.asarray([-1., -1.]))
plot_countour(plt, x, y, z, colormap, label)

colormap = 'Reds'
label = 'Group a, Label -1'
z = gauss(x, y, Sigma=np.asarray([[.8, .0], [.0, .8]]), mu=np.asarray([1., 1.]))
plot_countour(plt, x, y, z, colormap, label)

colormap = 'Greens'
label = 'Group b, Label +1'
z = gauss(x, y, Sigma=np.asarray([[.5, .0], [.0, .5]]), mu=np.asarray([.5, -.5]))
plot_countour(plt, x, y, z, colormap, label)

colormap = 'YlOrRd'
colormap = 'RdPu'
label = 'Group b, Label -1'
z = gauss(x, y, Sigma=np.asarray([[.5, .0], [.0, .5]]), mu=np.asarray([.5, .5]))
plot_countour(plt, x, y, z, colormap, label)

# plt.colorbar()  # draw colorbar
# plot data points.
# plt.scatter(x, y, marker='o', c='b', s=5)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title('Toy dataset')

green_patch = mpatches.Patch(color='green', label='Group b, Label +1')
blue_patch = mpatches.Patch(color='blue', label='Group a, Label +1')
red_patch = mpatches.Patch(color='red', label='Group a, Label -1')
purple_patch = mpatches.Patch(color='purple', label='Group b, Label -1')
plt.legend(loc='upper left', handles=[blue_patch, green_patch, red_patch, purple_patch])

plt.savefig('Toytest_plot')
plt.show()