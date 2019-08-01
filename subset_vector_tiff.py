from __future__ import print_function
import os
from glob import glob


input_dir = '/Volumes/mac_zhigang/Satellite_Data/Wuliangsuhai'
# detmine using the QGIS - subset tool
lake_name = 'Lake_wuliangsuhai'
xmin, ymin = 290768, 4563075
xmax, ymax = 342910, 4512132

all_tiff_files = glob(os.path.join(input_dir, '*.tif'))
for tiff_file in all_tiff_files:
    base_name = os.path.splitext(tiff_file)[0]
    sub_outfile = base_name + '_' + lake_name + '.tiff'
    limit = str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
    print(limit)
    sub_cmd = 'gdal_translate -projwin ' + limit + ' -of GTiff ' + \
        tiff_file + ' ' + sub_outfile
    print(sub_cmd)
    os.system(sub_cmd)
    shp_file = base_name + '_' + lake_name + '.shp'
    shp_basename = os.path.basename(shp_file).split('.')[0]
    raster_cmd = 'gdal_polygonize.py ' + sub_outfile + ' ' + \
        shp_file + ' -b 1 -f "ESRI Shapefile"' + ' ' + shp_basename + ' DN'
    print(raster_cmd)
    os.system(raster_cmd)
