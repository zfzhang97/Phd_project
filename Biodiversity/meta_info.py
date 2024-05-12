import seaborn as sns
from tools import *
root = 'E:/PHD_Project'
data_root = root + '/Data/'
results_root = root + 'Results/'
temp_root = root + 'Temp/'
global_year_range = '1982-2020'
centimeter_factor = 1 / 2.54
global_VIs_year_range_dict = {
    'NDVI': '1982-2015',
    'NDVI-origin': '1982-2015',
    'VOD-origin': '2003-2015',
    'VOD-anomaly': '2003-2015',
    'CSIF-origin': '2001-2015',
    'CSIF-anomaly': '2001-2015',
    'VOD-k-band-origin': '1988-2015',
    'VOD-k-band-anomaly': '1988-2015',
}
global_color_list = [
    '#844000',
    '#fc9831',
    '#fffbd4',
    '#86b9d2',
    '#064c6c',
]

global_cmap = sns.blend_palette(global_color_list, as_cmap=True, n_colors=6)
global_cmap_r= sns.blend_palette(global_color_list[::-1], as_cmap=True, n_colors=6)

