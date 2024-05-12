import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
from ntpath import join
import os
import xarray as xr
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from pyproj import Transformer, CRS

def mk_dir(fdir, force=False):
    if not os.path.isdir(fdir):
        if force == True:
            os.makedirs(fdir)
        else:
            os.mkdir(fdir)

class MULTIPROCESS:

    def __init__(self, func, params):
        self.func = func
        self.params = params
        
    def run(self, process=4, process_or_thread='p', **kwargs):
        '''
        # 并行计算加进度条
        :param func: input a kenel_function
        :param params: para1,para2,para3... = params
        :param process: number of cpu
        :param thread_or_process: multi-thread or multi-process,'p' or 't'
        :param kwargs: tqdm kwargs
        :return:
        '''

        if process > 0:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool(process)
            elif process_or_thread == 't':
                pool = TPool(process)
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results
        else:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool()
            elif process_or_thread == 't':
                pool = TPool()
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.map(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results

def calculate_area(dataset, mask):

    # Extract data and coordinates
    lons = dataset.lon.values
    lats = dataset.lat.values
    lon_interval = np.abs(lons[1] - lons[0])
    lat_interval = np.abs(lats[1] - lats[0])
    lons_new = dataset.lon.values -0.5* lon_interval
    lats_new = dataset.lat.values -0.5* lat_interval
    lons_new = np.append(lons_new, lons_new[-1]+lon_interval)
    lats_new = np.append(lats_new, lats_new[-1]+lat_interval)
    lon_grid, lat_grid = np.meshgrid(lons_new, lats_new)  # Create 2D grid of lon/lat values
    
    # Define Mollweide projection
    mollweide_projection = CRS.from_proj4("+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs(CRS.from_epsg(4326), mollweide_projection, always_xy=True)
    
    # Transform coordinates
    x, y = transformer.transform(lon_grid.ravel(), lat_grid.ravel())
    x = x.reshape(lon_grid.shape)
    y = y.reshape(lat_grid.shape)
    
    # Calculate pixel size variations
    dx = x[:-1, 1:] - x[:-1, :-1]
    dy = y[1:, :-1] - y[:-1, :-1]
    
    # Calculate area per pixel
    pixel_area = np.abs(dx) * np.abs(dy)
    
    # Create area data array
    area_array = xr.DataArray(pixel_area, dims=("lat", "lon"), coords={"lat": lats, "lon": lons})
    
    # Apply condition and calculate total area
    total_area = (area_array.where(mask)).sum().item()
    
    return total_area

def pixel_area_ratio_statistics(dataset,  all_area_mask, condition_value = 0.05 , type = '>='):

    if type == ">=":
        mask = (dataset >= condition_value)
    elif type == ">":
        mask = (dataset > condition_value)
    elif type == "<=":
        mask = (dataset <= condition_value)
    elif type == "<":
        mask = (dataset < condition_value)
    elif type == "==":
        mask = (dataset == condition_value)

    pixel_count= mask.sum().item()
    pixel_count_ratio = (mask.sum().item()) /(all_area_mask.sum().item())
    total_area = calculate_area(dataset, mask)
    total_area_ratio = calculate_area(dataset, mask)/calculate_area(dataset, all_area_mask)

    return {'pixel_count':pixel_count,'pixel_count_ratio':pixel_count_ratio,'total_area':total_area,'total_area_ratio':total_area_ratio}

def statistics_2d(dataset, quantile = 0.5):
 
    min = dataset.min().values
    max = dataset.max().values
    mean = dataset.mean().values
    sum = dataset.sum().values
    std = dataset.std().values
    var = dataset.var().values
    median = dataset.median().values
    count = dataset.count().values
    quantile = dataset.quantile(quantile).values 

    return {'min': min,'max':max,'mean':mean,'sum':sum,'std':std,'var':var,'median':median,'count':count,'quantile':quantile} 

    
 

            
