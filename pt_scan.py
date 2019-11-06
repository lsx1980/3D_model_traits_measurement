"""
Version: 1.5

Summary: compute the cross section plane based on 3d model

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python pt_scan.py -p /home/suxingliu/model-scan/model-data/ -m surface.ply -i 3000 -de 1 -r 1

python pt_scan.py -p /home/suxingliu/model-scan/model-data/ -m root.ply -i 1000 -de 1 -r 0

argument:
("-p", "--path", required=True,    help="path to *.ply model file")
("-m", "--model", required=True,    help="file name")
("-i", "--interval", required=True,    type = int, help="intervals along sweeping plane")
("-d", "--direction", required=True,    type = int, help="direction of sweeping plane, X=0, Y=1, Z=2")
("-r", "--reverse", required=True,    type = int, help="Reverse model top_down, 1 for Ture, 0 for False")

"""
#!/usr/bin/env python


#import matplotlib
#matplotlib.use('Qt4Agg')

# import the necessary packages
from plyfile import PlyData, PlyElement
import numpy as np
import argparse

from sklearn import preprocessing

from matplotlib import pyplot as plt
import matplotlib.cm as cm

#from xvfbwrapper import Xvfb
        
from mayavi import mlab
from mayavi.core.ui.mayavi_scene import MayaviScene

from operator import itemgetter
import os

import warnings
warnings.filterwarnings("ignore")

import pylab as pl


def mkdir(path):
    """Create result folder"""
    
    # remove space at the beginning
    path=path.strip()
    # remove slash at the end
    path=path.rstrip("\\")
 
    # path exist?   # True  # False
    isExists=os.path.exists(path)
 
    # process
    if not isExists:
        # construct the path and folder
        print (path + ' folder constructed!')
        # make dir
        os.makedirs(path)
        return True
    else:
        # if exists, return 
        print (path +' path exists!')
        return False


def get_world_to_view_matrix(mlab_scene):
    """returns the 4x4 matrix that is a concatenation of the modelview transform and
    perspective transform. Takes as input an mlab scene object."""

    if not isinstance(mlab_scene, MayaviScene):
        raise TypeError('argument must be an instance of MayaviScene')


    # The VTK method needs the aspect ratio and near and far clipping planes
    # in order to return the proper transform. So we query the current scene
    # object to get the parameters we need.
    scene_size = tuple(mlab_scene.get_size())
    clip_range = mlab_scene.camera.clipping_range
    aspect_ratio = float(scene_size[0])/float(scene_size[1])

    # this actually just gets a vtk matrix object, we can't really do anything with it yet
    vtk_comb_trans_mat = mlab_scene.camera.get_composite_projection_transform_matrix(
                                aspect_ratio, clip_range[0], clip_range[1])

     # get the vtk mat as a numpy array
    np_comb_trans_mat = vtk_comb_trans_mat.to_array()

    return np_comb_trans_mat


def get_view_to_display_matrix(mlab_scene):
    """ this function returns a 4x4 matrix that will convert normalized
        view coordinates to display coordinates. It's assumed that the view should
        take up the entire window and that the origin of the window is in the
        upper left corner"""

    if not (isinstance(mlab_scene, MayaviScene)):
        raise TypeError('argument must be an instance of MayaviScene')

    # this gets the client size of the window
    x, y = tuple(mlab_scene.get_size())

    # normalized view coordinates have the origin in the middle of the space
    # so we need to scale by width and height of the display window and shift
    # by half width and half height. The matrix accomplishes that.
    view_to_disp_mat = np.array([[x/2.0,      0.,   0.,   x/2.0],
                                 [   0.,  -y/2.0,   0.,   y/2.0],
                                 [   0.,      0.,   1.,      0.],
                                 [   0.,      0.,   0.,      1.]])

    return view_to_disp_mat


def apply_transform_to_points(points, trans_mat):
    """a function that applies a 4x4 transformation matrix to an of
        homogeneous points. The array of points should have shape Nx4"""

    if not trans_mat.shape == (4, 4):
        raise ValueError('transform matrix must be 4x4')

    if not points.shape[1] == 4:
        raise ValueError('point array must have shape Nx4')

    return np.dot(trans_mat, points.T).T


if __name__ == '__main__':
        
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to *.ply model file")
    ap.add_argument("-m", "--model", required = True, help = "model file name")
    ap.add_argument("-i", "--interval", required = False, default = '1000',  type = int, help= "intervals along sweeping plane")
    ap.add_argument("-de", "--direction", required = False, default = '1',   type = int, help = "direction of sweeping plane, X=0, Y=1, Z=2")
    ap.add_argument("-r", "--reverse", required = False, default = '1', type = int, help = "Reverse model top_down, 1 for Ture, 0 for False")
    args = vars(ap.parse_args())


    # setting path to model file 
    current_path = args["path"]
    filename = args["model"]
    file_path = current_path + filename

    # make the folder to store the results
    #mkpath = current_path + str(filename[0:-4]) + '_' + str(args["interval"]) + '_' + str(args["direction"])
    mkpath = current_path + "cross_section_scan"
    mkdir(mkpath)
    save_path = mkpath + '/'
    print ("results_folder: " + save_path)


    
    # load the model file
    try:
        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)
            num_vertex = plydata.elements[0].count
            
            print("Ply data structure: \n")
            print(plydata)
            print
            print("Number of 3D points in current model: {0} \n".format(num_vertex))
        
    except:
        print("Model file not exist!")
        sys.exit(0)
    
        

    #Parse the ply format file and Extract the data
    Data_array = np.zeros((num_vertex, len(plydata.elements[0].properties)))
    
    
    
    
    '''
   #Normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    point_normalized = min_max_scaler.fit_transform(Data_array)

    #point_normalized = preprocessing.scale(Data_array)
    #robust_scaler = preprocessing.RobustScaler()
    #point_normalized = robust_scaler.fit_transform(Data_array)
    #point_normalized = robust_scaler.transform(Data_array)
    
    
    #accquire data range
    min_x = Data_array[:, 0].min()
    max_x = Data_array[:, 0].max()
    min_y = Data_array[:, 1].min()
    max_y = Data_array[:, 1].max()
    
    range_data = int(max(max_x - min_x, max_y - min_y)*1.2)
    
    range_data_x = int((max_x - min_x)*1.2)
    
    range_data_y = int((max_y - min_y)*1.2)
    
    print "range_data_x, range_data_y"
    print range_data, range_data_x, range_data_y
    '''

    #Extract property list
    property_list = list(plydata.elements[0].properties)

    for index, item in enumerate(plydata.elements[0].properties, start = 0):
        Data_array[:,index] = plydata['vertex'].data[item.name]
        property_list[index] = (item.name,item.val_dtype)

    #print(Data_array)
    print(Data_array.shape)


    # scanning direction 
    Axis_sweep = args["direction"]

    if args["reverse"] == 1:
        flag_reverse = True
    else:
        flag_reverse = False
  

   
    # sort points according to z value increasing order
    Sorted_point = np.asarray(sorted(Data_array, key = itemgetter(Axis_sweep), reverse = flag_reverse))

    # calcute the number of sweeping layers based on defined intervals 
    if num_vertex % args["interval"] == 0: #even 
        interval = args["interval"]
    else: #odd
        interval = args["interval"] + 1
    
    num_layer = int(num_vertex/interval)


    #Assign datatype
    mydtype = property_list

    # Divide the 3D points data into chunks 
    sub_point = np.array_split(Sorted_point,num_layer)
    print("Number of sweeping plane: {0} \n".format(len(sub_point)))

    #calculate the interval based on user selection 
    if args["interval"] == 0: #even 
        interval = args["interval"]
    else: #odd
        interval = args["interval"] + 1

    '''
    disp_coords_rec = []
    disp_coords_rec_range_x = []
    disp_coords_rec_range_y = []
    
    print ("Scanning 3D model file with moving depth plane...\n")
    
    angle_azimuth = Axis_sweep*90
    angle_elevation = (Axis_sweep + 1)*45
    angle_roll = (1-Axis_sweep)*180
    
    
    

    # save the cross section plane 
    for index, item in enumerate(sub_point, start = 0):
        
        # extract coordinates values
        X =  item[:,0]
        Y =  item[:,1]
        Z =  item[:,2]
        
        N = len(item)
        
    
        #mlab.options.offscreen = True
        #mayavi.engine.current_scene.scene.off_screen_rendering = True
        
        # initialize the mayavi figure handle
        f = mlab.figure()
        
        # setup vewing transformation matrix to align the model in world coordinates 
        mlab.view(azimuth = angle_azimuth, elevation = angle_elevation,roll = angle_roll, distance = None, focalpoint= None, reset_roll = True)
        
        # plot the points with mlab
        pts = mlab.points3d(X, Y, Z)

        # create a single N x 4 array of points
        # adding a fourth column of ones expresses the world points in homogenous coordinates
        W = np.ones(X.shape)
        hmgns_world_coords = np.column_stack((X, Y, Z, W))

        # applying world_to_view_matrix transform to get 'unnormalized' view coordinates 
        comb_trans_mat = get_world_to_view_matrix(f.scene)
        #get the transform matrix for the current scene view
        view_coords = apply_transform_to_points(hmgns_world_coords, comb_trans_mat)

        # to get normalized view coordinates, divide through by the fourth element
        norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

        # transform from normalized view coordinates to display coordinates.
        view_to_disp_mat = get_view_to_display_matrix(f.scene)
        disp_coords = apply_transform_to_points(norm_view_coords, view_to_disp_mat)
        
        min_x = disp_coords[:, 0].min()
        max_x = disp_coords[:, 0].max()
        min_y = disp_coords[:, 1].min()
        max_y = disp_coords[:, 1].max()

        #range_data = int(max(max_x - min_x, max_y - min_y)*0.5)
        range_data_x = int(max_x)
        range_data_y = int(max_y)
        
        #disp_coords_rec_range.append(range_data)
        disp_coords_rec_range_x.append(range_data_x)
        disp_coords_rec_range_y.append(range_data_y)
        
        # change data type as int for image processing
        disp_coords = np.asarray(disp_coords).astype(int)
        disp_coords_rec.append(disp_coords)

        # close mayavi scene
        mlab.close()
        
    # accquire the projected 2D coordinates value range 
    range_data_x = int(np.amax(np.asarray(disp_coords_rec_range_x))*1.2)
    range_data_y = int(np.amax(np.asarray(disp_coords_rec_range_y))*1.2)
    
    #define shift value as average center value
    #offset = int(np.mean(np.asarray(disp_coords_rec_range)))

    print ("Writing scanned cross section results...\n")
    
    # save the cross section plane 
    for index, item in enumerate(disp_coords_rec, start = 0):
        
        #Generating an image of values 1 
        #im_thresh = np.ones((range_data_y,range_data_x))
        im_thresh = np.zeros((range_data_y,range_data_x))
       
        #shift coordinates based on center value
        coord = disp_coords_rec[index] 
        
        # assign image values based on ccordinates
        for i in range(N):
            
            im_thresh[[coord[:, 1][i]], [coord[:, 0][i]]] = 1
            
        #Save images as jpeg format
        filename = save_path + str('{:04}'.format(index)) + '.jpg'
        
        # save image as binary type
        plt.imsave(filename, im_thresh, cmap = cm.gray)
    


'''



'''
  #########################################################
    
    #Normalize data
    #min_max_scaler = preprocessing.MinMaxScaler()
    #point_normalized = min_max_scaler.fit_transform(Data_array)

    #point_normalized = preprocessing.scale(Data_array)
    #robust_scaler = preprocessing.RobustScaler()
    #point_normalized = robust_scaler.fit_transform(Data_array)
    #point_normalized = robust_scaler.transform(Data_array)
    
    
    #accquire data range
    min_x = Data_array[:, 0].min()
    max_x = Data_array[:, 0].max()
    min_y = Data_array[:, 1].min()
    max_y = Data_array[:, 1].max()
    
    range_data = int(max(max_x - min_x, max_y - min_y)*1.2)
    
    range_data_x = int((max_x - min_x)*1.2)
    
    range_data_y = int((max_y - min_y)*1.2)
    
    print "range_data_x, range_data_y"
    print range_data, range_data_x, range_data_y
    
    #######################################################


 
        ##########################################
        
        # Save sub models
        Converting a 2D numpy array to a structured array  
        Data_structured_array = np.core.records.array(list(tuple(item.transpose())), dtype = mydtype)
        
        #Create the ply object instances
        ply_struct_instance = PlyElement.describe(Data_structured_array, 'vertex', comments = [])
        
        #Save cleaned ply model file
        filename = save_path + str('{:05}'.format(index)) + '_sub.ply'
        PlyData([ply_struct_instance], text = False).write(filename)
            
        # Save cross section images
        item[:,Axis_sweep] = np.median(item[:,Axis_sweep])
        
        ###########################################
'''

