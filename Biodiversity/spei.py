from meta_info import *
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats import linregress
from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools import *
from plot import *

def calculate_trend_and_p(fpath, gs='annual', method='mean'):
    dataset = xr.open_dataset(fpath).spei
    if gs == 'annual':
        # 确保调用.mean()方法来计算年度平均值
        ds_annual = dataset.resample(time='YS').mean()
    else:
        ds_annual = dataset

    # 初始化趋势和p值数据结构，确保使用.isel(time=0)来选择数据
    trend = xr.full_like(ds_annual.isel(time=0), fill_value=np.nan)
    p_values = xr.full_like(ds_annual.isel(time=0), fill_value=np.nan)

    # 使用.lat.values和.lon.values来确保遍历数值而不是DataArray
    for lat in tqdm(ds_annual.lat.values, desc='latitude'):
        for lon in ds_annual.lon.values:
            
            y = ds_annual.sel(lat=lat, lon=lon).values
            x = np.arange(1,len(y)+1)
            #只进行非nan去趋势
            valid_indices = ~np.isnan(y)  # 获取y中非NaN值的索引
            x_clean = x[valid_indices]
            y_clean = y[valid_indices]
            # 判断y是否有足够的数据点进行线性回归
            if y_clean.size > 3:

                # 使用linregress计算趋势，确保y是numpy数组
                slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)

                # 更新趋势和p值
                trend.loc[dict(lat=lat, lon=lon)] = slope
                p_values.loc[dict(lat=lat, lon=lon)] = p_value
    # 保存结果        
    trend.to_netcdf('D:/Data/SPEI/Result/trend.nc') 
    p_values.to_netcdf('D:/Data/SPEI/Result/p_values.nc')
    return trend, p_values

def trend_sig(trend, p_values):
    
    p_values_sig=p_values.where(p_values <= 0.05)

    p_values_mask=~p_values_sig.isnull()
    
    trend_sig=trend.where(p_values_mask)
    # 保存结果
    trend_sig.to_netcdf('D:/Data/SPEI/Result/trend_sig.nc') 

    return trend_sig

def drought_statistic(dataset, threshold, ):

    threshold = threshold  # 定义干旱阈值
    mask=~(dataset.isel(time=12).isnull())  #定义掩码，去除原数据nan值
    # 步骤1: 识别干旱事件
    is_drought = dataset < threshold
    drought_times = xr.zeros_like(dataset.isel(time=12))
    drought_duration = xr.zeros_like(dataset.isel(time=12))
    drought_intensity = xr.zeros_like(dataset.isel(time=12))
    drought_frequency = xr.zeros_like(dataset.isel(time=12))

    # 遍历每个空间位置
    for lat in tqdm(dataset.lat, desc='Processing Latitude'):
        for lon in dataset.lon:
            # 提取该位置的时间序列
            time_series = is_drought.sel(lat=lat, lon=lon)
            spei_series = dataset.sel(lat=lat, lon=lon)
            # 计算干旱次数
            drought_starts = ((time_series.shift(time=1) == False) & (time_series == True))
            drought_times.loc[lat, lon] = drought_starts.sum()
            total_drought_months = time_series.sum().item()
            drought_frequency.loc[lat, lon] = total_drought_months / len(time_series)
            
            # 计算每次干旱的持续时间、强度
            duration = 0
            durations = []
            intensities = []
            intensity = 0
            
            for i in range(len(time_series)):
                if time_series[i].item():
                    duration += 1  # 累计干旱月数
                    intensity += spei_series[i].item()  # 累计干旱强度
                elif duration > 0:
                    durations.append(duration)
                    intensities.append(-intensity)  # 存储干旱强度，取相反数
                    duration = 0
                    intensity = 0
            if durations:
                # 计算平均干旱持续时间
                drought_duration.loc[lat, lon] = np.mean(durations)
            if intensities:
                # 计算平均干旱强度
                drought_intensity.loc[lat, lon] = np.mean(intensities)
        
    drought_times = drought_times.where(mask)
    drought_frequency = drought_frequency.where(mask)
    drought_duration = drought_duration.where(mask)
    drought_intensity = drought_intensity.where(mask)
    drought_times.to_netcdf('D:/Data/SPEI/Result/drought_times.nc') 
    drought_frequency.to_netcdf('D:/Data/SPEI/Result/drought_frequency.nc')
    drought_duration.to_netcdf('D:/Data/SPEI/Result/drought_duration.nc')
    drought_intensity.to_netcdf('D:/Data/SPEI/Result/drought_intensity.nc') 

    return drought_times, drought_frequency, drought_duration, drought_intensity

def drought_statistic_multiprocess(fpath, threshold, ):

    dataset=xr.open_dataset(fpath).spei
    threshold = threshold  # 定义干旱阈值
    mask=~(dataset.isel(time=12).isnull()) #定义掩码，去除原数据nan值
    # 步骤1: 识别干旱事件
    is_drought = dataset < threshold
    drought_times = xr.zeros_like(dataset.isel(time=12))
    drought_duration = xr.zeros_like(dataset.isel(time=12))
    drought_intensity = xr.zeros_like(dataset.isel(time=12))
    drought_frequency = xr.zeros_like(dataset.isel(time=12))

    def process_location(lat_lon):
        
        lat, lon = lat_lon
        time_series = is_drought.sel(lat=lat, lon=lon)
        spei_series = dataset.sel(lat=lat, lon=lon)
        
        # 干旱次数
        drought_starts = ((time_series.shift(time=1) == False) & (time_series == True))
        drought_times_val = drought_starts.sum().item()
        total_drought_months = time_series.sum().item()
        drought_frequency_val = total_drought_months / len(time_series)
        
        # 持续时间和强度
        durations = []
        intensities = []
        duration = 0
        intensity = 0
        for i in range(len(time_series)):
            if time_series[i].item():
                duration += 1  # 累计干旱月数
                intensity += spei_series[i].item()  # 累计干旱强度
            elif duration > 0:
                durations.append(duration)
                intensities.append(-intensity)  # 存储干旱强度，取相反数
                duration = 0
                intensity = 0
                
        # duration_mean = np.mean(durations) if durations else 0
        # intensity_mean = np.mean(intensities) if intensities else 0

        if durations:
        # 计算平均干旱持续时间
            duration_mean = np.mean(durations)
        else:
            duration_mean= np.nan
        if intensities:
        # 计算平均干旱强度
            intensity_mean= np.mean(intensities)
        else:
            intensity_mean= np.nan
            
        return lat, lon, drought_times_val, duration_mean, intensity_mean, drought_frequency_val
    # lat_lon_pairs = [(lat, lon) for lat in dataset.lat for lon in dataset.lon]
    lat_lon_pairs= [(lat, lon) for lat in dataset.lat.values for lon in dataset.lon.values]
    results= MULTIPROCESS(process_location, lat_lon_pairs).run(process=10)

    for lat, lon, times, duration, intensity, frequency in results:
        drought_times.loc[lat, lon] = times
        drought_duration.loc[lat, lon] = duration
        drought_intensity.loc[lat, lon] = intensity
        drought_frequency.loc[lat, lon] = frequency

    drought_times = drought_times.where(mask)
    drought_frequency = drought_frequency.where(mask)
    drought_duration = drought_duration.where(mask)
    drought_intensity = drought_intensity.where(mask)
    drought_times.to_netcdf('D:/Data/SPEI/Result/drought_times.nc') 
    drought_frequency.to_netcdf('D:/Data/SPEI/Result/drought_frequency.nc')
    drought_duration.to_netcdf('D:/Data/SPEI/Result/drought_duration.nc')
    drought_intensity.to_netcdf('D:/Data/SPEI/Result/drought_intensity.nc') 

    return drought_times, drought_frequency, drought_duration, drought_intensity