#!/usr/bin/env python
# coding: utf-8
"""
python scripts for process MSI data using Seadas
"""
from __future__ import print_function
import os
import glob
import datetime
import subprocess
import numpy as np


def create_par(infile, outfile, ancfile):
    '''create par file based ancfile'''
    no2file = 'no2file=$OCDATAROOT/common/no2_climatology.hdf'

    '''Write Files:'''
    f = open(ancfile, 'a')
    f.write(no2file + '\n')

    f.write('\n#IO Options:\n')
    f.write('\nifile=' + infile + '\n')
    f.write('ofile=' + outfile + '\n\n')

    f.write('\n#l2prod options:\n')

    l2_prod = 'l2prod=rhos_443 rhos_482 rhos_561 rhos_655 rhos_865 rhos_1609 rhos_2201 Rrs_443 Rrs_482 Rrs_561 Rrs_655 chlor_a latitude longitude'
    f.write(l2_prod + '\n')
    f.write('aer_opt=-1\n')
    f.write('aer_wave_short=865\n')
    f.write('aer_wave_long=1609\n')
    f.write('south=36.7\n')
    f.write('north=38.5\n')
    f.write('west=-76.91\n')
    f.write('east=-75.58\n')
    f.write('\n\n')
    f.close()
    return


def make_png(l2file):
    import h5py
    from PIL import Image, ImageDraw, ImageEnhance

    png_file = l2file.split('.')[0] + '.png'
    if os.path.exists(png_file):
        print('%s existed.' % png_file)
        return
    hfile = h5py.File(l2file, 'r')
    min_ref = 0.01
    max_ref = 0.9  # scretch thresold
    rhos_655 = hfile['geophysical_data/rhos_655']
    rhos_561 = hfile['geophysical_data/rhos_561']
    rhos_482 = hfile['geophysical_data/rhos_482']

    #
    height, width = rhos_655.shape
    img_R = rhos_655[...]
    img_G = rhos_561[...]
    img_B = rhos_482[...]

    img_R[img_R < min_ref] = min_ref
    img_R[img_R > max_ref] = max_ref
    img_G[img_G < min_ref] = min_ref
    img_G[img_G > max_ref] = max_ref
    img_B[img_B < min_ref] = min_ref
    img_B[img_B > max_ref] = max_ref

    r = 255.0 * np.log(img_R / min_ref) / np.log(max_ref / min_ref)
    g = 255.0 * np.log(img_G / min_ref) / np.log(max_ref / min_ref)
    b = 255.0 * np.log(img_B / min_ref) / np.log(max_ref / min_ref)

    # make invalid points white (saturation is the only reason for them....
    r[np.isinf(r)] = 255
    g[np.isinf(r)] = 255
    b[np.isinf(r)] = 255
    r[np.isinf(g)] = 255
    g[np.isinf(g)] = 255
    b[np.isinf(g)] = 255
    r[np.isinf(b)] = 255
    g[np.isinf(b)] = 255
    b[np.isinf(b)] = 255
    r[r < 0] = 255
    g[r < 0] = 255
    b[r < 0] = 255
    r[g < 0] = 255
    g[g < 0] = 255
    b[g < 0] = 255
    r[b < 0] = 255
    g[b < 0] = 255
    b[b < 0] = 255

    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # fill pixel
    for x in range(height):
        for y in range(width):
            if np.isnan(r[x, y]) or np.isnan(g[x, y]) or np.isnan(b[x, y]):
                color = (255, 255, 255)
            else:
                color = (int(r[x, y]), int(g[x, y]), int(b[x, y]))
            draw.point((y, x), fill=color)

    # enhance
    (ImageEnhance.Contrast(image).enhance(2.0)).save(png_file, 'png')

    print(png_file + ' is exported!')


def run_process(l1a_file, working_path, out_path):

    # parse the filename:LC08_L1TP_045030_20180507_20180517_01_T1.tar.gz
    basefilename = os.path.basename(l1a_file).split('.')
    outfile = out_path + basefilename[0] + '_ChesapeakBay_L2.h5'

    if os.path.exists(outfile):
        print('SKIP: %s' % outfile)
        return 0
    # decompress
    cmd = 'tar -xzf ' + filename + ' --verbose'
    print(cmd)
    status = subprocess.call(cmd, shell=True)
    if status != 0:
        return 1
    MTL_file = working_path + basefilename[0] + '_MTL.txt'
    # download ancfile
    cmd = 'getanc.py -cv ' + MTL_file
    status_code = subprocess.call(cmd, shell=True)
    print('outfile:' + outfile)

    #
    ancfile = working_path + os.path.basename(MTL_file) + '.anc'
    if status_code != 0:
        print('\nObtaining ancillary files faild! Exit.\n')
        return 2

    # l2gen
    create_par(MTL_file, outfile, ancfile)
    cmd = 'l2gen par=' + ancfile
    status = subprocess.call(cmd, shell=True)

    if status != 0:
        print('Process is failed: %s' % cmd)
        return 3

    print('\n\n >>> %s has been generated.' % outfile)

    return 0


if __name__ == '__main__':

    source = "/Users/zhigang/Desktop/SateliteData/lc08/"
    out_path = "/Users/zhigang/Desktop/SateliteData/L2/"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    working_path = out_path+'tmp/'
    if not os.path.exists(working_path):
        os.mkdir(working_path)

    all_files = glob.glob(source + 'LC08_L1TP*')
    f = open(out_path + 'prog_log_LC8_119038.log', 'a')
    f.write('Status(0-Sucessed,1-decompress failed,2-anc failed,3-l2gen failed),\tL1A_Filename,\tTime+\n\n')
    os.chdir(working_path)
    for filename in all_files:
        status = run_process(filename, working_path, out_path)
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(str(status) + ',' + filename + ',' + nowTime + '\n\n')
        os.system('rm -rf ' + working_path + '*.TIF --verbose')
        os.system('rm -rf ' + working_path + '*.txt --verbose')
        os.system('rm -rf ' + working_path + '*.anc --verbose')
        f.flush()
    f.close()
