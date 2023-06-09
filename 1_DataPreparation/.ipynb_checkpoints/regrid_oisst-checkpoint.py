import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import glob
import os
import matplotlib.pyplot as plt
import xesmf as xe
import gc
import sys

path_temp = sys.argv[1]
name_variable = sys.argv[2]
name_variable_ds = sys.argv[3]
units = sys.argv[4]
save_regridder = sys.argv[5]
save_regridder = save_regridder == 'True'

# print(path_temp,name_variable,name_variable_ds,units,save_regridder)

def regrid(ds, variable, reuse_weights=False,filename_weights = None):
    """
    Function to regrid onto coarser ERA5 grid (0.25-degree).
    Args:
        ds (xarray dataset): file.
        variable (str): variable.
        reuse_weights (boolean): Whether to use precomputed weights to speed up calculation.
                                 Defaults to ``False``.
        filename_weights (str): if reuse_weights is True, then a string for the weights path is needed.
    Returns:
        Regridded data file for use with machine learning model.
    """
    # ds.lon = ds.longitude
    # ds.lat = ds.latitude
    # ds = ds.rename({'latitude':'lat',
    #                'longitude':'lon'})
    ds_out = xe.util.grid_2d(lon0_b=0.25-0.25, lon1_b=359.75+0.25, d_lon=0.5, 
                            lat0_b=-29.75-0.25, lat1_b=89.75+0.25, d_lat=0.5)
    
    if reuse_weights == False:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights)
        return regridder(ds[variable]),regridder
    else:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights,
            filename = filename_weights)
        return regridder(ds[variable])

def regrid_file(variable,variable_ds_name,file2load,file2write,write_regridder = False, units = ""):
    ds_temp = xr.open_dataset(file2load)
    ds_temp = ds_temp[variable_ds_name].mean('zlev').to_dataset()
    if write_regridder==True:
        ds_temp_1g, regridder = regrid(ds_temp,variable_ds_name,False)
        regridder.to_netcdf(f'/glade/work/jhayron/Data4Predictability/Regridders/regrid_{variable}_OISST.nc')
    else:
        ds_temp_1g = regrid(ds_temp,variable_ds_name,True,\
            f'/glade/work/jhayron/Data4Predictability/Regridders/regrid_{variable}_OISST.nc')
    ds_temp_1g = ds_temp_1g.to_dataset(name=variable_ds_name)
    ds_temp_1g[variable_ds_name].attrs['units'] = units
    ds_temp_1g.to_netcdf(file2write, encoding={variable_ds_name: {'zlib': True, 'complevel': 4}})
    ds_temp.close()
    ds_temp_1g.close()

folder_outputs = '/glade/work/jhayron/Data4Predictability/OISSTv2/SST/'
date_str_out = str(dt.datetime.strptime(os.path.basename(path_temp).split('.')[-2],'%Y%m%d').date())
path_out = f'{folder_outputs}{name_variable}_Daily_05Deg_{date_str_out}.nc'

if os.path.exists(os.path.dirname(path_out)) == False:
    os.mkdir(os.path.dirname(path_out))

regrid_file(name_variable,name_variable_ds,path_temp,path_out,save_regridder,units)

