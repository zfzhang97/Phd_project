from matplotlib import pyplot as plt
import cartopy.crs as ccrs

def plot_spei2d(dataset, cmap='coolwarm', vmax=None, vmin=None):
    """
    绘制指定时间点的SPEI数据。

    参数:
    - nc_file_path: str, NetCDF文件的路径。
    - time_index: int, 从NetCDF文件中选择的时间点的索引。
    - cmap: str 或 matplotlib.colors.Colormap, 用于数据可视化的颜色映射。
    """

    # 创建图和轴，使用Robinson投影
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson()})

    # 绘制数据
    dataset.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=True, vmax=vmax , vmin=vmin, cbar_kwargs={'label': ''})

    #隐藏tittle
    ax.set_title('')

    # 添加海岸线
    ax.coastlines()
    
    # 添加经纬度网格，但不显示网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='none', alpha=0)
    # gl.top_labels = False
    # gl.bottom_labels = False
    # gl.left_labels = False
    # gl.right_labels = False
    gl.xlines = False
    gl.ylines = False