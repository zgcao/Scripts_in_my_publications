#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from __future__ import print_function

import os, shutil, subprocess
import requests
from bs4 import BeautifulSoup

def downd_gcs(scene_name,outdir):
    #gsutil cp -r 199/024/LC08_L1TP_199024_20180903_20180912_01_T1/ ./
    url = 'gs://gcp-public-data-landsat/LC08/01/'
    fileparts = scene_name.split('_')
    path,row = fileparts[2][0:3],fileparts[2][3:]    
    LANDSAT_PATH = os.path.join(outdir,path+'_'+row)

    download_url = url + path + '/' + row + '/' + scene_name + '/'
    cmd = '/Users/zhigang/Scripts/google-cloud-sdk/bin/gsutil'
    cmd = cmd + ' -m cp -r ' +download_url + ' ' + outdir
    print(cmd)
    status = subprocess.call(cmd,shell = True)
    if status != 0:
        print(scene_name+' Failed!')
    return

def download_scencename(scene_name,outdir):
    url = 'https://landsat-pds.s3.amazonaws.com/c1/L8/'
    
    print('EntityId:'+scene_name.strip()+'\n')
    
    #LC08_L1TP_139045_20170304_20170316_01_T1
    fileparts = scene_name.split('_')
    path,row = fileparts[2][0:3],fileparts[2][3:]    
    LANDSAT_PATH = os.path.join(outdir,path+'_'+row)

    download_url = url + path + '/' + row + '/' + scene_name + '/index.html'
    print('url: {}'.format(download_url))
    
    # Request the html text of the download_url from the amazon server. 
    response = requests.get(download_url)
    print(response.status_code)
    # If the response status code is fine (200)
    if response.status_code == 200:
        print('Start to download...')
        # Import the html to beautiful soup
        html = BeautifulSoup(response.content, 'html.parser')

        # Create the dir where we will put this image files.
        entity_dir = os.path.join(LANDSAT_PATH, scene_name)
        os.makedirs(entity_dir)

        # Second loop: for each band of this image that we find using the html <li> tag
        for li in html.find_all('li'):

            # Get the href tag
            file = li.find_next('a').get('href')

            print('  Downloading: {}'.format(file))

            # Download the files
            # code from: https://stackoverflow.com/a/18043472/5361345

            response = requests.get(download_url.replace('index.html', file), stream=True)

            with open(os.path.join(entity_dir, file), 'wb') as output:
                shutil.copyfileobj(response.raw, output)
            del response

"""Main"""
filename = '/Volumes/TOSHIBA_EXT/Landsat8_Data/2018Check/order_976756.txt'
outdir = '/Volumes/TOSHIBA_EXT/Landsat8_Data/2018Check'
for scene_name in open(filename,'r').readlines():
    downd_gcs(scene_name.strip(),outdir)

