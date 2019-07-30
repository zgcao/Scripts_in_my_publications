#!/usr/bin/env python
# coding: utf-8

# Download the data files of Landsat T1 from google cloud storage
# Before you used the script to download, two things need to be prepared
# 1. install the gstuil toolbox as referenced:
#    https://cloud.google.com/storage/docs/gsutil_install?hl=zh-cn
# 2. Order a list of data in earthexplorer.usgs.gov
#    then, save their scenced ID from the Traking Bulkorders
#    input template:

from __future__ import print_function

import subprocess


def downd_gcs(scene_name, outdir):
    # gsutil cp -r 199/024/LC08_L1TP_199024_20180903_20180912_01_T1/ ./
    fileparts = scene_name.split('_')
    mission = fileparts[0]
    url = 'gs://gcp-public-data-landsat/' + mission + '/01/'
    path, row = fileparts[2][0:3], fileparts[2][3:]

    download_url = url + path + '/' + row + '/' + scene_name + '/'
    cmd = '/Users/zhigang/Scripts/google-cloud-sdk/bin/gsutil'
    cmd = cmd + ' -m cp -r ' + download_url + ' ' + outdir
    print(cmd)

    status = subprocess.call(cmd, shell=True)
    if status != 0:
        print(scene_name + ' Failed!')
    return


"""Main"""
filename = '/Volumes/mac_zhigang/Satellite_Data/Wuliangsuhai/LE07'
outdir = '/Volumes/mac_zhigang/Satellite_Data/Wuliangsuhai/'
for scene_name in open(filename, 'r').readlines():
    downd_gcs(scene_name.strip(), outdir)
