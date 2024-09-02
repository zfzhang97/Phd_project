from dask.diagnostics import ProgressBar
import dask
import multiprocessing
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from preprocessing import *
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import xarray as xr
import numpy as np
import concurrent.futures
from tqdm import tqdm

def calculate_resistance_pixel(lat, lon, ndvi_data, drought_mask_data, normal_mean_data):
    # 从数据中选择特定的像元
    pixel_ndvi = ndvi_data.sel(lat=lat, lon=lon)
    pixel_drought_mask = drought_mask_data.sel(lat=lat, lon=lon)
    pixel_ndvi_mean = normal_mean_data.sel(lat=lat, lon=lon)

    # 初始化抵抗力值
    results = []

    # 检查是否全为NaN
    if not (np.isnan(pixel_ndvi).all() or np.isnan(pixel_drought_mask).all()):
        region_indices = np.where(pixel_drought_mask == 1)[0]
        if len(region_indices) > 0:
            region_starts = [region_indices[0]]
            for i in range(1, len(region_indices)):
                if region_indices[i] != region_indices[i-1] + 1:
                    region_starts.append(region_indices[i])

            # 对每个连续区间计算均值，并替换为抵抗力
            for start_idx in region_starts:
                end_idx = start_idx + 1
                while end_idx < len(pixel_ndvi.time) and pixel_drought_mask[end_idx] == 1:
                    end_idx += 1
                region_ndvi = pixel_ndvi.isel(time=slice(start_idx, end_idx))
                region_mean = region_ndvi.mean().item()
                resistance_value = region_mean - pixel_ndvi_mean
                return results.append((lat, lon, region_ndvi.time[0], resistance_value))
    return results.append((lat, lon, None, None))
def multi_calculate_resistance(Deseason_detrend_ndvi, drought_mask, growing_season_mask, spei):
    
    # Z-score 标准化
    mean_value = Deseason_detrend_ndvi.mean(dim='time', skipna=True)
    std_value = Deseason_detrend_ndvi.std(dim='time', skipna=True)
    Deseason_detrend_ndvi = (Deseason_detrend_ndvi - mean_value) / std_value

    # 计算正常条件下的ndvi均值
    normal_growing_season_mask = growing_season_mask.where(spei >= -0.5, drop=True)
    normal_growing_season_mean = Deseason_detrend_ndvi.where(normal_growing_season_mask, drop=True).mean(dim='time', skipna=True)

    # 多进程处理
    coords = [(lat, lon) for lat in Deseason_detrend_ndvi.lat.values for lon in Deseason_detrend_ndvi.lon.values]
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_pixel = [executor.submit(calculate_resistance_pixel, lat, lon, Deseason_detrend_ndvi, drought_mask, normal_growing_season_mean) for lat, lon in coords]
        for future in tqdm(concurrent.futures.as_completed(future_to_pixel), total=len(coords)):
            results.extend(future.result())
    
    Deseason_detrend_ndvi.close()
    drought_mask.close()
    growing_season_mask.close()
    spei.close()
    # 更新抵抗力数据集
    resistance = xr.full_like(Deseason_detrend_ndvi, np.nan)

    for lat, lon, time, value in results:
        if time is not None:
            resistance.loc[{'lat': lat, 'lon': lon,  'time': time}] = value

    return resistance


def multi_calculate_resilience(Deseason_detrend_ndvi, drought_mask, growing_season_mask, spei):
    """
    计算恢复力 
    公式为: Y_post_mean-Y_normal_mean
    其中Y是去季节去趋势的ndvi
    """

    # 自动确定块大小以利用可用内存和CPU
    cpu_count = multiprocessing.cpu_count()
    chunk_size = {'lat': int(Deseason_detrend_ndvi.lat.size / cpu_count), 
                  'lon': int(Deseason_detrend_ndvi.lon.size / cpu_count)}

    # 将 xarray 数据转换为 Dask 数组以支持并行计算
    Deseason_detrend_ndvi = Deseason_detrend_ndvi.chunk(chunk_size)
    drought_mask = drought_mask.chunk(chunk_size)
    growing_season_mask = growing_season_mask.chunk(chunk_size)
    spei = spei.chunk(chunk_size)

    # 初始化恢复力数据集，所有值为NaN
    resilience = xr.full_like(Deseason_detrend_ndvi, np.nan, dtype='float32')  

    # Z-score 标准化
    mean_value = Deseason_detrend_ndvi.mean(dim='time', skipna=True)
    std_value = Deseason_detrend_ndvi.std(dim='time', skipna=True)
    Deseason_detrend_ndvi = (Deseason_detrend_ndvi - mean_value) / std_value

    # 计算正常条件下的 ndvi 均值
    normal_growing_season_mask = growing_season_mask.where(spei >= -0.5)
    normal_growing_season_mean = Deseason_detrend_ndvi.where(normal_growing_season_mask == 1).mean(dim='time', skipna=True)
    
    # 定义一个处理每个像素的函数
    def process_pixel(lat, lon):
        pixel_ndvi = Deseason_detrend_ndvi.sel(lat=lat, lon=lon)
        pixel_drought_mask = drought_mask.sel(lat=lat, lon=lon)
        pixel_growing_season_mask = growing_season_mask.sel(lat=lat, lon=lon)
        pixel_ndvi_mean = normal_growing_season_mean.sel(lat=lat, lon=lon)

        # 检查像元是否不全为 NaN
        if not (np.isnan(pixel_ndvi).all() or np.isnan(pixel_drought_mask).all() or np.isnan(pixel_growing_season_mask).all()):
            
            # 计算连续为 1 的区间的索引
            region_indices = np.where(pixel_drought_mask == 1)[0]  # 获取为 1 的位置索引
            if len(region_indices) > 0:
                region_starts = [region_indices[0]]  # 连续区间的起始索引
                for i in range(1, len(region_indices)):
                    if region_indices[i] != region_indices[i-1] + 1:  # 如果不连续，则记录新的起始索引
                        region_starts.append(region_indices[i])

                for start_idx in region_starts:
                    end_idx = start_idx + 1
                    while end_idx < len(pixel_ndvi.time) and pixel_drought_mask[end_idx] == 1:
                        end_idx += 1  # 找到连续区间的结束索引
                    region_ndvi = pixel_ndvi.isel(time=slice(int(start_idx), int(end_idx)))  # 提取生长季的ndvi
                    recovery_time = None
                    for time_idx, ndvi_value in enumerate(pixel_ndvi.sel(time=pixel_ndvi.time[(start_idx + 1):])):
                        if abs(ndvi_value - pixel_ndvi_mean) / pixel_ndvi_mean  <= 0.05:
                            recovery_time = pixel_ndvi.time[(start_idx+1) + time_idx]
                            break
                    # 如果找到恢复时间点，计算该点一年内的生长季平均NDVI
                    if recovery_time is not None:
                        # 计算两个时间点之间的差值，并转换为年份
                        time_diff_years = (pd.to_datetime(recovery_time.values) - pd.to_datetime(pixel_ndvi.time[start_idx].values)).days / 365.25  # 使用365.25来考虑闰年

                        if time_diff_years < 2:
                            one_year_later = pd.to_datetime(recovery_time.values) + pd.DateOffset(years=1)  # 恢复点之后的一年
                            two_years_later = pd.to_datetime(pixel_ndvi.time[start_idx].values) + pd.DateOffset(years=2) # 干旱开始后的两年

                            # 根据 one_year_later 是否在两年内进行选择
                            if one_year_later <= two_years_later:
                                end_time = one_year_later
                            else:
                                end_time = two_years_later
                            # 选取从恢复时间到结束时间的时间范围
                            pixel_ndvi_masked = pixel_ndvi.where(pixel_growing_season_mask == 1)
                            future_growing_season = pixel_ndvi_masked.sel(time=slice(recovery_time, end_time))
                            # 计算平均值
                            average_ndvi_over_year = future_growing_season.mean()

                            # 存储结果
                            return lat, lon, region_ndvi.time[0], average_ndvi_over_year - pixel_ndvi_mean

        return lat, lon, None, None
    
    # 使用 dask 并行计算
    tasks = [dask.delayed(process_pixel)(lat, lon) for lat in Deseason_detrend_ndvi.lat.values for lon in Deseason_detrend_ndvi.lon.values]
    results = dask.compute(*tasks)

    # 将结果存储回数据集
    for lat, lon, time, resilience_value in tqdm(results):
        if time is not None:
            resilience.loc[{'lat': lat, 'lon': lon, 'time': time}] = resilience_value

    return resilience

def run(deseason_detrend_ndvi, drought_droped, growing_season_mask, spei, out_fname):

        # client = Client()
        # adjust_thread = threading.Thread(target=adjust_dask_workers, args=(client,))
        # adjust_thread.daemon = True
        # adjust_thread.start()
       
        drought_resistance = multi_calculate_resistance(deseason_detrend_ndvi, drought_droped, growing_season_mask, spei)
        drought_resistance.to_netcdf(rf'G:\PHD_Project\2_types_results\Resistance_resilience_0.08\{out_fname}_resistance.nc')

        # drought_resilience = multi_calculate_resilience(deseason_detrend_ndvi, drought_droped, growing_season_mask, spei)
        # drought_resilience.to_netcdf(rf'G:\PHD_Project\2_types_results\Resistance_resilience_0.08\{out_fname}_resilience.nc')  
        # result_drought_resistance.append(drought_resistance)
        # result_drought_resilience.append(drought_resilience)
