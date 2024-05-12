from meta_info import *
from tools import *
from plot import *
from ntpath import join
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
import os
import math
import dask
from copy import deepcopy

def clean(datadir):

    fdir = join(datadir, 'Source')
    outdir = join(datadir,'Clean')
    mk_dir(outdir)
    for f in tqdm(os.listdir(fdir)):
        if not f.endswith('.nc4'):
            continue
        fpath = join(fdir,f)
        outpath = join(outdir,f)

        #加载数据
        dataset=xr.open_dataset(fpath)
        ndvi=dataset.ndvi

        # 步骤 1: 将 NDVI 值缩放回 [-0.03, 1.0] 范围
        ndvi_scaled = ndvi/ 10000

        # 去除等于 -5000 或者小于-0.03 或者 大于1.0
        condition = (ndvi_scaled != -0.5) & (ndvi_scaled >= -0.03) & (ndvi_scaled <= 1.0)

        ndvi_cleaned = ndvi_scaled.where(condition)
        # 保存数据
        ndvi_cleaned.to_netcdf(outpath)

def resample_merge(datadir):
    
    fdir = join(datadir, 'Clean')
    outdir = join(datadir,'Resample_merge')
    mk_dir(outdir)
    
    # 初始化空列表，用于存储重采样后的数据集
    resampled_datasets = []

    for f in tqdm(os.listdir(fdir)):
        if not f.endswith('.nc4'):
            continue
        fpath = join(fdir,f)
        outpath = join(outdir,f)
        
        # 使用xarray打开文件
        dataset=xr.open_dataset(fpath)

        # 重采样数据到0.5度分辨率，假设原始数据的经纬度命名为'lat'和'lon'
        resampled_ds = dataset.interp(lat=np.arange(-89.75, 90, 0.5), lon=np.arange(-179.75, 180, 0.5), method='linear')
        resampled_datasets.append(resampled_ds)

    # 合并所有重采样后的数据集
    combined_ds = xr.concat(resampled_datasets, dim='time')  # 时间维度名为'time'
    combined_ds=combined_ds.sortby('time')   #按照时间升序整理数据

    # 将合并后的数据集保存为一个新的NetCDF文件
    combined_ds.to_netcdf(outpath)

def monthly_compose(datadir, method='mean'):
        
    fdir = join(datadir, 'Resample_merge')
    outdir = join(datadir,'Monthly_compose')
    outpath = join(outdir,'monthly_compose.nc4')
    mk_dir(outdir)
    for f in tqdm(os.listdir(fdir)):
        if not f.endswith('.nc4'):
            continue
        fpath = join(fdir,f)
       
        # 使用xarray读取目录中的所有NetCDF文件
        dataset = xr.open_dataset(fpath)
        
        # 根据method选择聚合方法
        if method == 'mean':
            monthly_data = dataset.resample(time='1MS').mean()
        elif method == 'max':
            monthly_data = dataset.resample(time='1MS').max()
        elif method == 'sum':
            monthly_data = dataset.resample(time='1MS').sum()
        else:
            raise ValueError("Unsupported method. Choose from 'mean', 'max', 'sum'.")
        
        monthly_data.to_netcdf(outpath)

def deseason_detrend(datadir):

    fdir = join(datadir, 'Monthly_compose')
    outdir = join(datadir,'Deseason_detrend')
    outpath = join(outdir,'deseason_detrend.nc4')
    mk_dir(outdir)
    for f in tqdm(os.listdir(fdir)):
        if not f.endswith('.nc4'):
            continue
        fpath = join(fdir,f)
       
        # 使用xarray读取目录中的所有NetCDF文件
        dataset = xr.open_dataset(fpath)

        # 计算多年平均值
        multiyear_mean = dataset.mean(dim='time')

        # 筛选出多年平均值大于或等于0.2的像元，不满足条件的设置为NaN
        filtered_dataset = dataset.where(multiyear_mean >= 0.2)

        # 第1步: 去季节性
        # 计算每个月份的多年平均值
        monthly_avg = filtered_dataset.groupby('time.month').mean('time')

        # 从每个原始值中减去对应月份的多年平均值
        deseasonalized = filtered_dataset.groupby('time.month') - monthly_avg

        # 第2步: 去趋势
        deseason_detrend_data = xr.full_like(deseasonalized, fill_value=np.nan)
        
        for lat in deseasonalized.lat:
            for lon in deseasonalized.lon:
                # 获取时间编码作为自变量
                time = np.arange(1,len(deseasonalized.time.dt.month)+1)
                y = deseasonalized.sel(lat=lat, lon=lon).ndvi.values

                #只进行非nan去趋势
                valid_indices = ~np.isnan(y)  # 获取y中非NaN值的索引
                time_clean = time[valid_indices]
                y_clean = y[valid_indices]
              
                if  y_clean.size>1:
                    slope, intercept, _, _, _ = stats.linregress(time_clean, y_clean)
                    trend_line = slope * time + intercept
                    deseason_detrend_data.loc[dict(lat=lat, lon=lon)] =xr.DataArray(y - trend_line, dims=['time'])
                    
        deseason_detrend_data.to_netcdf(outpath)   

def Month_to_daily(datadir):

    fdir = join(datadir, 'Resample_merge')
    outdir = join(datadir,'Month_to_daily')
    mk_dir(outdir)
    for f in tqdm(os.listdir(fdir)):
        if not f.endswith('.nc4'):
            continue
        fpath = join(fdir,f)
        dataset = xr.open_dataset(fpath)
        for year in range(1982, 2023):
            # 为每一年生成时间范围
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            target_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            # 使用slice选取该年份的数据
            ds_year = dataset.sel(time=slice(start_date, end_date))
            ds_daily = ds_year.interp(time=target_dates, method='linear', kwargs={"fill_value": "extrapolate"})
            #定义保存路径
            outpath = join(outdir, f'{year}.nc4')
            # 将数据保存到文件
            ds_daily.to_netcdf(outpath)

class HANTS:
    
    def __init__(self):

        pass

    def left_consecutive_index(self, values_list):
        left_consecutive_non_valid_index = []
        for i in range(len(values_list)):
            if np.isnan(values_list[i]) :
                left_consecutive_non_valid_index.append(i)
            else:
                break
        return left_consecutive_non_valid_index

    def right_consecutive_index(self, values_list):
        right_consecutive_non_valid_index = []
        for i in range(len(values_list) - 1, -1, -1):
            if np.isnan(values_list[i]):
                right_consecutive_non_valid_index.append(i)
            else:
                break
        return right_consecutive_non_valid_index 
    
    def run(self, datadir):
        """
        运行HANTS算法并生成结果。
        """
        fdir = join(datadir, 'Month_to_daily')
        outdir = join(datadir,'Hants_daily')
        mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.nc4'):
                continue
            fpath = join(fdir,f)
            
            # 读取NetCDF文件
            dataset = xr.open_dataset(fpath)
            # 假设NDVI数据在名为'NDVI'的变量中
            ndvi_data = dataset['ndvi']
            
            # 计算每个像素时间序列的标准差
            std_dev = ndvi_data.std(dim='time')
            
            # 初始化HANTS处理后的NDVI数据的容器
            hants_ndvi = np.full_like(ndvi_data.values, np.nan)

            # 遍历每个像素
            lat_dim = ndvi_data.coords['lat']
            lon_dim = ndvi_data.coords['lon']
            # 对于每个像素应用HANTS
            for i, lat in enumerate(lat_dim):
                for j, lon in enumerate(lon_dim):
                    fill_val= -9999
                    original_values = ndvi_data.sel(lat=lat.item(), lon=lon.item()).values
                    input_values= original_values.copy()
                    input_values[~np.isfinite(input_values)] = fill_val
                    fet = std_dev.sel(lat=lat.item(), lon=lon.item()).item() * 2  # 两倍标准差作为fet
                    ni=len(input_values)
                    nb = 365
                    nf = 3
                    ts = range(ni)
                    low = 0
                    high = 1
                    HiLo = 'Hi'
                    delta = 0.1
                    dod = 1
                    hants_values, outliers = self.hants(ni=ni, nb=nb, nf=nf, y=input_values, ts=ts,
                                        HiLo=HiLo, low=low, high=high, fet=fet, dod=dod, delta=delta, fill_val=fill_val)
                    # 存储处理后的时间序列
                    results = np.where(hants_values == -9999, np.nan, hants_values)
                    # results = np.where(outliers.flatten(), original_values.flatten(), hants_values.flatten())

                    left_consecutive_non_valid_index = self.left_consecutive_index(list(original_values))
                    right_consecutive_non_valid_index = self.right_consecutive_index(list(original_values))
                    results_new = []
                    results= results.flatten()
                    for k in range(len(results)):
                        if k in left_consecutive_non_valid_index:
                            results_new.append(np.nan)
                        elif k in right_consecutive_non_valid_index:
                            results_new.append(np.nan)
                        else:
                            if results[k] < 0:
                                results_new.append(np.nan)
                                continue
                            elif results[k] > 1:
                                results_new.append(np.nan)
                                continue
                            else:
                                results_new.append(results[k])
                  
                    hants_ndvi[:, i, j] = np.array(results_new)
            # 创建一个新的xarray数据集来存储处理后的数据
            new_ds = xr.Dataset(
                {
                    "ndvi": (("time", "lat", "lon"), hants_ndvi),
                },
                coords={
                    "time": ndvi_data.coords["time"],
                    "lat": ndvi_data.coords["lat"],
                    "lon": ndvi_data.coords["lon"],
                },
            )
            
            #定义保存路径
            outpath = join(outdir,f)
            # 将数据保存到文件
            new_ds.to_netcdf(outpath)
              
            
    @staticmethod
    def hants(ni, nb, nf, y, ts, HiLo, low, high, fet, dod, delta, fill_val):
        
        '''
        输入：
        ni:样本个数，散点时间序列长度
        nb:拟合样本的周期长度
        nf:非0频率的频率个数
        y:散点时间序列，真实值
        ts:拟合样本的数量
        HiLo:{'Hi','Lo'},用于指示拒绝低的异常值还是高的异常值
        low:有效范围的最小值
        high:有效范围的最大值，范围外的值被拒绝
        fet:拟合曲线的误差容错,偏离大于fet的点将从曲线中删除
        dod:过度确定程度(迭代停止，如果迭代次数达到曲线拟合所需的最小值)
        delta:小正数(如0.1)抑制高振幅
        fill_val:填充值

        输出：
        yr:重构后的时间序列
        outlier:异常点
        '''


        """定义正余弦序列"""
        # [
        #  [1,     1,     ..., 1    ],
        #  [cosx1, cosx2, ..., cosxn],
        #  [sinx1, sinx2, ..., sinxn],
        #  [cosx1, cosx2, ..., cosxn],
        #  ...
        #  ]
        mat = np.zeros((min(2*nf+1, ni), ni))  # [nr,ni]
        # amp = np.zeros((nf + 1, 1))

        # phi = np.zeros((nf+1, 1))
        yr = np.zeros((ni, 1)) # 重构曲线
        outliers = np.zeros((1, len(y))) # 异常值,[1,ni]

        """Filter"""
        sHiLo = 0
        if HiLo == 'Hi': # 
            sHiLo = -1
        elif HiLo == 'Lo':
            sHiLo = 1

        nr = min(2*nf+1, ni) # number of 2*+1 frequencies, or number of input images
        noutmax = ni - nr - dod # 异常点的最大个数
        # dg = 180.0/math.pi
        mat[0, :] = 1.0 # 用于求偏最小二乘中的b


        """定义一个周期内的正余弦曲线"""
        ang = 2*math.pi*np.arange(nb)/nb 
        cs = np.cos(ang)
        sn = np.sin(ang)

        """将时序样本点转成cos、sin"""
        i = np.arange(1, nf+1)
        for j in np.arange(ni):
            index = np.mod(i*ts[j], nb) # 取余
            mat[2 * i-1, j] = cs.take(index)
            mat[2 * i, j] = sn.take(index)


        """根据数值有效范围将范围外的值设为0"""
        p = np.ones_like(y) # (ni,)
        bool_out = (y <= low) | (y > high)
        p[bool_out] = 0
        outliers[bool_out.reshape(1, y.shape[0])] = 1
        nout = np.sum(p == 0) # 统计异常值的数量


        """异常值个数判断"""
        if nout > noutmax:
            if np.isclose(y, fill_val).any():
                # 当填充值是序列中的一员时，直接填充整个序列，并将整个序列设为异常值
                ready = np.array([True])
                yr = y
                outliers = np.zeros((y.shape[0]), dtype=int)
                outliers[:] = fill_val
            else:
                # 数据点太少
                # raise Exception('Not enough data points.')
                # print('Not enough data points.')
                ready = np.array([True])
                yr = y
        else:
            ready = np.zeros((y.shape[0]), dtype=bool) # 设置迭代的初值


        """PLS,迭代优化"""
        nloop = 0
        nloopmax = ni

        while ((not ready.all()) & (nloop < nloopmax)):

            nloop += 1
            za = np.matmul(mat, p*y) # [nr,ni]*[ni,1],转换的傅里叶序列(去除异常点)

            A = np.matmul(np.matmul(mat, np.diag(p)), # 将正常的列过滤出来
                            np.transpose(mat)) # [nr, ni]*[ni,ni]*[ni,nr]
            
            #  add delta to suppress high amplitudes but not for [0,0]
            A = A + np.identity(nr)*delta  # [nr,nr]+[nr,nr]
            A[0, 0] = A[0, 0] - delta

            zr = np.linalg.solve(A, za) # 对傅里叶序列进行PLS [nr,nr],[nr,1]->[nr,1]

            # solve linear matrix equation and define reconstructed timeseries
            yr = np.matmul(np.transpose(mat), zr) # [ni,nr]*[nr,1] -> [ni,1]

            # calculate error and sort err by index
            diffVec = sHiLo*(yr-y)
            err = p*diffVec
            
            err_ls = list(err)
            err_sort = deepcopy(err)
            err_sort.sort()

            rankVec = [err_ls.index(f) for f in err_sort]

            # select maximum error and compute new ready status
            maxerr = diffVec[rankVec[-1]]
            ready = (maxerr <= fet) | (nout == noutmax)

            # if ready is still false
            if (not ready):
                i = ni - 1
                j = rankVec[i]
                
                # 迭代的本质就是p在不停的变，从而导致其他相应值的改变
                while ((p[j]*diffVec[j] > 0.5*maxerr) & (nout < noutmax)):
                    p[j] = 0
                    outliers[0, j] = 1
                    nout += 1
                    i -= 1
                    if i == 0:
                        j = 0
                    else:
                        j = 1

        return yr, outliers

def growing_season_mask_monthly(datadir):

    # 设置Dask以自动分割大块，避免创建大块
    dask.config.set({'array.slicing.split_large_chunks': True})
    # 使用通配符指定文件路径
    fdir= join(datadir,'Hants_daily\\*.nc4')
    outdir = join(datadir,'Growing_season_mask_monthly')
    mk_dir(outdir)
    # 打开多个文件并自动合并为一个数据集
    dataset = xr.open_mfdataset(fdir, concat_dim='time', combine='nested' )

    # 计算多年平均值
    multiyear_mean = dataset.mean(dim='time', skipna=True)

    # 筛选出多年平均值大于或等于0.2的像元，不满足条件的设置为NaN
    filtered_dataset = dataset.where(multiyear_mean >= 0.2)

    # 对每一年进行归一化处理
    def normalize(ds):
        yearly_min = ds.min(dim='time', skipna=True)
        yearly_max = ds.max(dim='time', skipna=True)
        return (ds - yearly_min) / (yearly_max - yearly_min)
    
    normalized = filtered_dataset.groupby('time.year').apply(normalize)
    
    # 使用每年的NDVI范围的30%作为阈值
    def filter_by_threshold(ds):
        yearly_min = ds.min(dim='time', skipna=True)
        yearly_max = ds.max(dim='time', skipna=True)
        threshold = (yearly_max - yearly_min) * 0.3
        return ds > threshold
    
    filtered = normalized.groupby('time.year').apply(filter_by_threshold)
    # 将逐日数据合并为月数据
    # 只要该月内有值为1，则该月值取为1
    growing_season_mask_monthly = filtered.resample(time='MS').any()
    # 保存文件
    outpath = join(outdir, 'growing_season_mask_monthly.nc4')
    growing_season_mask_monthly.to_netcdf(outpath)    

class Hot_mask:

    """
        提取高温掩码
        
    """
  
    def __init__(self):

        pass
     
    def process_mean_tmp(self, temp, mask):
        """
        该函数将生长季的气温处理为所处生长季的均值
        
        """
        results = xr.full_like(temp, np.nan)  # 初始化结果数据集，所有值为NaN
    
        # 遍历每个地理位置
        for lat in tqdm(temp.lat.values):
            for lon in temp.lon.values:
                # 提取单个像元的温度和掩码
                pixel_temp = temp.sel(lat=lat, lon=lon)
                pixel_mask = mask.sel(lat=lat, lon=lon)

                # 计算连续为 1 的区间的索引
                region_indices = np.where(pixel_mask == 1)[0]  # 获取连续为 1 的位置索引
                if len(region_indices) > 0:
                    region_starts = [region_indices[0]]  # 连续区间的起始索引
                    for i in range(1, len(region_indices)):
                        if region_indices[i] != region_indices[i-1] + 1:  # 如果不连续，则记录新的起始索引
                            region_starts.append(region_indices[i])

                    # 对每个连续区间计算均值，并替换为均值
                    for start_idx in region_starts:
                        end_idx = start_idx + 1
                        while end_idx < len(pixel_temp.time) and pixel_mask[end_idx] == 1:
                            end_idx += 1  # 找到连续区间的结束索引
                        region_temp = pixel_temp.isel(time=slice(int(start_idx), int(end_idx)))  # 提取连续区间的温度值

                        region_mean = region_temp.mean()
                        results.loc[{'lat': lat, 'lon': lon, 'time': region_temp.time}] = region_mean

        results.to_netcdf(r'E:\PHD_Project\Data\CRU\Hot_drought_tmp\hot_mean_tmp.nc')         

        return results

    def process_temperatures_90_percentile(self, mean_temperatures):

        """
        该函数将提取90%分位数的生长季气温
        
        """
        # 初始化结果数组
        unique_temperatures_90_percentile = xr.full_like(mean_temperatures.isel(time=0), np.nan)

        # 遍历每个像元
        for lat_idx, lat in enumerate(tqdm(mean_temperatures.lat)):
            for lon_idx, lon in enumerate(mean_temperatures.lon):
                # 获取当前像元的温度值数组
                temp_values = mean_temperatures.sel(lat=lat, lon=lon).values
                
                # 去重
                unique_temp_values = np.unique(temp_values[~np.isnan(temp_values)])
                
                # 计算 90 百分位
                if len(unique_temp_values) > 0:
                    unique_temperatures_90_percentile.loc[{'lat': lat, 'lon': lon}] = np.nanpercentile(unique_temp_values, 90)

        return unique_temperatures_90_percentile

    def run(self):
        
        """
        该函数返回hot_mask
        
        """
        spei = xr.open_dataset(r'E:\PHD_Project\Data\SPEI\Source\spei03.nc').spei
        growing_season_mask = xr.open_dataset(r'E:\PHD_Project\Data\GIMMS3g_NDVI\Growing_season_mask_monthly\Growing_season_mask_monthly_leftrightclear.nc4').ndvi
        tmp = xr.open_dataset(r'E:\PHD_Project\Data\CRU\Source\cru_ts4.07.1901.2022.tmp.dat.nc').tmp

        # 按月重采样到每月开始, 将时间对齐
        spei = spei.resample(time='MS').mean()  # 'MS'表示每月的开始，mean()是一种聚合方法，你可以根据需要选择合适的聚合方法
        growing_season_mask = growing_season_mask.resample(time='MS').mean()
        
        # 截取ndvi同等时间段
        tmp = tmp.resample(time='MS').mean().sel(time=growing_season_mask.time)
        spei = spei.sel(time=growing_season_mask.time)
        mean_temperatures = self.process_mean_tmp(tmp, growing_season_mask)
        hot_90_percentile = self.process_temperatures_90_percentile(mean_temperatures)
        hot_mask =  mean_temperatures>hot_90_percentile
        hot_mask.to_netcdf(r'E:\PHD_Project\Data\CRU\Hot_drought_tmp\hot_mask.nc') 
        return hot_mask              

class clear__edges:

    def clear_edges(self, dataset):

        # 创建一个副本以修改数据
        new_dataset = np.copy(dataset)
        # 清除开始处的连续1
        first_zero = np.argmax(new_dataset == 0)  # 找到第一个0的位置
        if first_zero == 0 and new_dataset[0] != 0:
            # 如果没有0，则整个序列都是1
            new_dataset[:] = 0
        else:
            new_dataset[:first_zero] = 0

        # 清除末尾处的连续1
        last_zero = len(new_dataset) - np.argmax(new_dataset[::-1] == 0) - 1  # 反向数组中第一个0的位置
        if last_zero == len(new_dataset) - 1 and new_dataset[-1] != 0:
            # 如果没有0，则整个序列都是1
            new_dataset[:] = 0
        else:
            new_dataset[last_zero+1:] = 0

        return new_dataset
    def run(self):
        
        growing_season_mask=xr.open_dataset(r'E:\PHD_Project\Data\GIMMS3g_NDVI\Growing_season_mask_monthly\growing_season_mask_monthly.nc4').ndvi
        # 应用函数到每个像元
        modified_data = xr.apply_ufunc(
            self.clear_edges, growing_season_mask,
            input_core_dims=[['time']],  # 时间维度作为函数处理的核心维度
            vectorize=True,  # 向量化处理以支持数组操作
            output_core_dims=[['time']],  # 输出数据保持时间维度
            dask='allowed'  # 如果使用dask分布式计算，则允许
        )

        # 保存修改后的数据
        modified_data.to_netcdf(r'E:\PHD_Project\Data\GIMMS3g_NDVI\Growing_season_mask_monthly\Growing_season_mask_monthly_leftrightclear.nc4')



if __name__ == '__main__':

  pass