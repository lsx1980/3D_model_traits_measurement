"""
Version: 1.0

Summary: compute the graph from simplified cross section image sequence 

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python3 graph_compute.py -p /home/suxingliu/3scan-skeleton/test/ -ft png


argument:
("-p", "--path", required = True,    help = "path to image file")
("-ft", "--filetype", required = False, default = 'png',   help = "Image filetype")

"""

import glob, os, sys
import argparse

import numpy as np

from skimage.morphology import skeletonize
import sknw
import matplotlib.pyplot as plt
from mayavi import mlab
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import cv2

#pip install imagesize, progressbar2
import imagesize 
import progressbar
from time import sleep


def load_image(image_file):
    
    path, filename = os.path.split(image_file)

    #base_name = os.path.splitext(os.path.basename(filename))[0]

    im_gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    #Obtain the threshold image using OTSU adaptive filter
    thresh = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return int(thresh/255)
        

def get_cmap(n, name = 'hsv'):
    """get the color mapping"""
    
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    
    return plt.cm.get_cmap(name, n)

def mayavi_visualize(graph):
    
    mlab.figure(1, size = (500, 500))
    
    cmap = get_cmap(len(graph.edges()))
    
    #for (s,e) in graph.edges():
        
    for idx, (s,e) in enumerate(graph.edges()):
        
         #generate random different colors for different traces
        color_rgb = cmap(idx)[:len(cmap(idx))-1]
        
        pst = graph[s][e]['pts']

        mlab.plot3d(pst[:,0], pst[:,1], pst[:,2], tube_radius = 0.025, color = color_rgb)
    
    nodes = graph.nodes()
    
    ps = np.array([nodes[i]['o'] for i in nodes])
    
    mlab.points3d(ps[:,0], ps[:,1], ps[:,2], color=(1.0, 0.0, 0.0), scale_factor = 2.75)
    
    mlab.show() 



def plt_visualize(graph):
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection = '3d')
    
    for (s,e) in graph.edges():
        
        pst = graph[s][e]['pts']
        
        ax.plot(pst[:,0], pst[:,1], pst[:,2], c = 'green')
        
    nodes = graph.nodes()
    
    ps = np.array([nodes[i]['o'] for i in nodes])

    ax.scatter(ps[:,0], ps[:,1], ps[:,2], c = 'r')

    plt.title('Build Graph')
    
    plt.show()
    
    
    
if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-ft", "--filetype", required = False, default = 'png',   help = "Image filetype")
    args = vars(ap.parse_args())

    #global file_path, save_path_ac, save_path_label, parent_path, pattern_id, count, whorl_dis_array, save_path_convex
    
    # setting path to cross section image files
    file_path = args["path"]
    ext = args['filetype']
     
    #accquire image file list
    filetype = '*.' + ext
    image_file_path = file_path + filetype

    #accquire image file list
    imgList = sorted(glob.glob(image_file_path))
    
    #print(imgList)
    
    
    n_samples = len(imgList)
    
    if n_samples > 0 :
        
        width, height = imagesize.get(imgList[0])
        #print(width, height)
    
    else:
        
        print("Empty image folder, abort!")
        sys.exit(0)
    
   
    #bar = progressbar.ProgressBar(maxval = n_samples, widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    #progress bar display
    bar = progressbar.ProgressBar(maxval = n_samples)
    
    #bar.start()
    
    print("Loading images...")
   
    #initialize empty numpy array
    image_chunk = np.empty(shape = (n_samples, height, width), dtype = np.float32)
   
    #fill 3d image chunk with binary image data
    for file_idx, image_file in enumerate(imgList):
        
        image_chunk[file_idx, :, :] = load_image(image_file)
        
        bar.update(file_idx+1)
        
        sleep(0.1)

    bar.finish()
    
    print("image chunk size : {0}".format(str(image_chunk.shape)))
    
    #count number and frequency of 1 and 0
    uniqueValues, occurCount = np.unique(image_chunk, return_counts=True)
        
    print("Unique Values : " , uniqueValues)
        
    print("Occurrence Count : ", occurCount)
    

    
    
    #skeletonize
    ske = skeletonize(image_chunk)

    # build graph from skeleton
    graph = sknw.build_sknw(ske)
    
    mayavi_visualize(graph)
    
    
    
    
