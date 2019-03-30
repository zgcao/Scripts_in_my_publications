# -*- coding: utf-8 -*-
"""Load the models trained by train_rrc2OAC.py"""
import os
import time

import matplotlib
import numpy as np
import pylab as plt
from netCDF4 import Dataset

matplotlib.use('agg')

# This script was adapted from ACOLITE-QV
def nc_write(ncfile, dataset, data, wavelength=None, global_dims=None,
                 new=False, attributes=None, keep=True, offset=None, 
                 replace_nan=False, metadata=None, dataset_attributes=None, double=True,
                 chunking=True, chunk_tiles=[10,10], chunksizes=None,
                 format='NETCDF4', nc_compression=True
                 ):
    from numpy import isnan, nan, float32, float64
    from math import ceil

    if os.path.exists(os.path.dirname(ncfile)) is False:
         os.makedirs(os.path.dirname(ncfile))

    dims = data.shape
    if global_dims is None: global_dims = dims

    if chunking:
        if chunksizes is not None:
            chunksizes=(ceil(dims[0]/chunk_tiles[0]), ceil(dims[1]/chunk_tiles[1]))

    if new:
        if os.path.exists(ncfile): os.remove(ncfile)
        nc = Dataset(ncfile, 'w', format=format)

        ## set global attributes
        setattr(nc, 'data source', 'SeaDAS L2 products' )
        setattr(nc, 'Algorithm', ' xgboost algorithm' )
        setattr(nc, 'generated_on',time.strftime('%Y-%m-%d %H:%M:%S'))
        setattr(nc, 'contact', 'Zhigang CAO' )
        setattr(nc, 'product_type', 'NetCDF4' )
        setattr(nc, 'Institute', 'NIGLAS, CAS' )
        setattr(nc, 'version', '0.1 beta' )

        if attributes is not None:
            for key in attributes.keys():
                if attributes[key] is not None:
                    try:
                        setattr(nc, key, attributes[key])
                    except:
                        print('Failed to write attribute: {}'.format(key))

        ## set up x and y dimensions
        nc.createDimension('x', global_dims[1])
        nc.createDimension('y', global_dims[0])
    else:
        nc = Dataset(ncfile, 'a', format=format)

    if (not double) & (data.dtype == float64):
        data = data.astype(float32)

    ## write data
    if dataset in nc.variables.keys(): 
        ## dataset already in NC file
        if offset is None:
            if data.dtype in (float32, float64): nc.variables[dataset][:] = nan
            nc.variables[dataset][:] = data
        else:
            if replace_nan:
                tmp = nc.variables[dataset][offset[1]:offset[1]+dims[0],offset[0]:offset[0]+dims[1]]
                sub_isnan=isnan(tmp)
                tmp[sub_isnan]=data[sub_isnan]
                nc.variables[dataset][offset[1]:offset[1]+dims[0],offset[0]:offset[0]+dims[1]] = tmp
                tmp = None
            else:
                nc.variables[dataset][offset[1]:offset[1]+dims[0],offset[0]:offset[0]+dims[1]] = data
    else:
        ## new dataset
        var = nc.createVariable(dataset,data.dtype,('y','x'), zlib=nc_compression,
                                chunksizes=chunksizes,fill_value=-32767)
        if wavelength is not None: setattr(var, 'wavelength', float(wavelength))
        ## set attributes
        if dataset_attributes is not None:
            for att in dataset_attributes.keys():
                setattr(var, att, dataset_attributes[att])

        if offset is None:
            if data.dtype in (float32, float64): var[:] = nan
            var[:] = data
        else:
            if data.dtype in (float32, float64): var[:] = nan
            var[offset[1]:offset[1]+dims[0],offset[0]:offset[0]+dims[1]] = data
    if keep is not True: data = None

    ## close netcdf file
    nc.close()
    nc = None

def read_mask(filename):
    import h5py
    h5file = h5py.File(filename,'r')
    mask_data = np.array(h5file['water_mask'][:], dtype=np.int8)
    h5file.close()
    print('>>> %s readed...' % filename)
    return mask_data

def mask_water(chlora,waterMask):
    """Using the generated water mask to mask land and cloud
    """
    chlora[waterMask==0] = np.nan
    return chlora

def read_img_data(filename):
    """To read a netcdf file and return a dictionary"""
    import h5py
    h5file = h5py.File(filename,'r')
    RRC_443 = np.array(h5file['geophysical_data/rhos_443'][:], dtype=np.float32)
    RRC_482 = np.array(h5file['geophysical_data/rhos_482'][:], dtype=np.float32)
    RRC_561 = np.array(h5file['geophysical_data/rhos_561'][:], dtype=np.float32)
    RRC_655 = np.array(h5file['geophysical_data/rhos_655'][:], dtype=np.float32)
    RRC_865 = np.array(h5file['geophysical_data/rhos_865'][:], dtype=np.float32)
    RRC_1609 = np.array(h5file['geophysical_data/rhos_1609'][:], dtype=np.float32)
    RRC_2201 = np.array(h5file['geophysical_data/rhos_2201'][:], dtype=np.float32)
    lat = np.array(h5file['navigation_data/latitude'][:], dtype=np.float32)
    lon = np.array(h5file['navigation_data/longitude'][:], dtype=np.float32)
    all_data = {'RRC': np.array([RRC_443,RRC_482,RRC_561,RRC_655,RRC_865,RRC_1609,RRC_2201]), 
                'LAT':lat, 'LON':lon}
    h5file.close()
    print('>>> %s readed...' % filename)
    return all_data

def apply_model_matrix(rrc,model_path):
    # verctorized code
    import xgboost as xgb
    np.seterr(divide='ignore',invalid='ignore')
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(model_path)
    
    line,sample = rrc.shape[1],rrc.shape[2]

    # transform the 3D image to the 2D array
    rrc = rrc.reshape(7,-1).T
    
    # make a simple atmospheric correction: rrc(lambda) - rrc(2201)
    rrc = rrc[:,0:6] - np.tile(rrc[:,6],(6,1)).T
    
    # generate the band ratios and FAI
    rrc[rrc<0] = np.nan
    gb = rrc[:,2]/rrc[:,0]
    rb = rrc[:,3]/rrc[:,0]
    ng = rrc[:,4]/rrc[:,2] 
    nr = rrc[:,4]/rrc[:,3]
    FAI = rrc[:,4]-(rrc[:,3]+(rrc[:,5]-rrc[:,3])*(865.0-655.0)/(1609.0-655.0))
    rrc = np.column_stack((rrc,gb,rb,ng,nr,FAI))
    
    # predict
    dtrain = xgb.DMatrix(rrc)
    chlora = bst.predict(dtrain)
    
    # remove the minus value 
    chlora[chlora<0] = np.nan
    chlora = chlora.reshape(line,sample)
    print('>>> chlora generated at :'+time.strftime('%Y-%m-%d %H:%M:%S'))
    return chlora
    
def apply_model_pixel(rrc,model_path,water_mask):
    # calculate pixel by pixel
    # using the water_mask to filter the land pixel and only calculate the water pixel
    # this is a strategy to save time
    print('>>> chlora started at :'+time.strftime('%Y-%m-%d %H:%M:%S'))
    import xgboost as xgb
    np.seterr(divide='ignore',invalid='ignore')
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(model_path)
    
    line,col = rrc.shape[1],rrc.shape[2]

    # make a coarse atmospheric correction
    rrc = rrc[0:6,:,:] - rrc[6,:,:]

    # generate the band ratios and FAI
    gb = rrc[2,:,:]/rrc[0,:,:]
    rb = rrc[3,:,:]/rrc[0,:,:]
    ng = rrc[4,:,:]/rrc[2,:,:] 
    nr = rrc[4,:,:]/rrc[3,:,:]    
    FAI = rrc[4,:,:]-(rrc[3,:,:]+(rrc[5,:,:]-rrc[3,:,:])*(865.0-655.0)/(1609.0-655.0))

    rrc = np.array([rrc[0,:,:],rrc[1,:,:],rrc[2,:,:],rrc[3,:,:],rrc[4,:,:],rrc[5,:,:],gb,rb,ng,nr,FAI])
    chlora = np.full((line,col),np.nan)
    
    # only proecss the water pixels
    idx = np.where(water_mask==1)
    line_idx = idx[0]
    col_idx = idx[1]
    for i in range(len(line_idx)):
        row,col = line_idx[i],col_idx[i]
        dtrain = xgb.DMatrix([rrc[:,row,col],])
        chlora[row,col] = bst.predict(dtrain)

    # remove the minus value 
    chlora[chlora<0] = np.nan

    print('>>> chlora ended at :'+time.strftime('%Y-%m-%d %H:%M:%S'))
    return chlora

def output_retrieval(l2r_ncfile,lat,lon,chlora):
    """Write the Chla, Lat, Lon to a nc file with a compression netcdf4 format"""
    l2r_nc_new = True
    nc_write(l2r_ncfile,'LAT', lat, new=l2r_nc_new, 
             dataset_attributes={'long_name':'latitude','units':'degree +north'})
    
    print('	+++ lat written in %s' % l2r_ncfile)
    l2r_nc_new = False

    nc_write(l2r_ncfile,'LON', lon, new=l2r_nc_new, 
             dataset_attributes={'long_name':'longitude', 'units':'degree +east'})
    print('	+++ lon written in %s' %  l2r_ncfile)

    nc_write(l2r_ncfile,'chlora', chlora, new=l2r_nc_new, 
             dataset_attributes={'long_name':'chlorophll-a','valid_min':0.1,
                                 'valid_max':100.0,'units':'mg m^-3'})
    print('	+++ Chla written in %s'  % l2r_ncfile)
    

def plot_save_chlora(chlora,outname):
    """Plot and save the image of Chla with a jet scratch color bar"""
    cmap=plt.cm.jet
    fig = plt.figure()
    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    chlora[np.isnan(chlora)] = -32767
    im=ax.imshow(chlora, cmap = cmap, vmin=0, vmax=80)
    fig.colorbar(im, ax=ax,orientation='vertical', label='Chla')
    ax.axis('off')
    canvas.print_figure(outname, dpi=300)
    print('>>> chlora png generated at :'+time.strftime('%Y-%m-%d %H:%M:%S'))
    
if __name__ == '__main__':
    import glob
    l2_dir = r'G:\L2\\'
    outputdir = r'G:\L2'    # define the filename

    model_path = r'F:\百度云同步盘\07研究任务\15Landsat8的叶绿素a反演模型\Models\bst_rrc2chora_v2.model'
    filenames = glob.glob(l2_dir+'*_L2.h5')
    print('>>> %d files will be proceed!' % len(filenames))
    
    i = 1
    for filename in filenames:
        print('%d/%d is processing...' % (i,len(filenames)))
        i = i + 1
        l2r_ncfile = outputdir+os.path.sep+'_'.join(os.path.basename(filename).split('_')[0:5])+'_chlora.nc'
        if os.path.exists(l2r_ncfile):
            print('%s existed, skip.\n',l2r_ncfile)
            continue

        # read the data
        img_data = read_img_data(filename)
        RRC = img_data['RRC']
        lat = img_data['LAT']
        lon = img_data['LON']
        
        # if find the mask file, then mask the data
        water_maskfile = os.path.splitext(filename)[0]+'_waterMask.h5'

        # apply the model to retrieve
        if not os.path.exists(water_maskfile):
            chlora = apply_model_matrix(RRC,model_path)
        else:
            water_mask = read_mask(water_maskfile)            
            #chlora = apply_model_matrix(RRC,model_path)
            chlora = apply_model_pixel(RRC,model_path,water_mask)

        # writing data to a nc file
        output_retrieval(l2r_ncfile,lat,lon,chlora)
        print('>>> '+l2r_ncfile+ ' exported!')
        
        l2r_pltfile = os.path.splitext(l2r_ncfile)[0] + '.png'
        plot_save_chlora(chlora,l2r_pltfile)
        print('>>> '+l2r_pltfile+' exported!')
        
