import sys, os
import xarray as xr
import numpy as np

import shutil
import pandas as pd
import geopandas as gpd
import json

from dask.distributed import Client
import dask


# plotting modules
import pyvista as pv
# magic trick for white background
pv.set_plot_theme("document")
import panel
panel.extension(comms='ipywidgets')
panel.extension('vtk')
from contextlib import contextmanager
import matplotlib.pyplot as plt


@contextmanager
def mpl_settings(settings):
    original_settings = {k: plt.rcParams[k] for k in settings}
    plt.rcParams.update(settings)
    yield
    plt.rcParams.update(original_settings)


plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# define Pandas display settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

from pygmtsar import S1, Stack, tqdm_dask, ASF, Tiles, XYZTiles


SUBSWATH = 2
POLARIZATION = 'VV'
FILENAME = 'Aug 2018 - Sep 2018'


def get_sar_1_collections_from(file_path):
    with open (file_path, 'r') as f:
        scenes = f.readlines()
    return [f.strip() for f in scenes]


SAR_1_COLLECTIONS_FILE = f'data/bursts/{FILENAME}.txt'
scenes = get_sar_1_collections_from(SAR_1_COLLECTIONS_FILE)
SCENES = scenes
SCENES.reverse()

WORKDIR = 'raw_golden_desc'
DATADIR = 'data_golden_desc'
DEM = f'{DATADIR}/dem.nc'


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


# clear_directory(WORKDIR)
# clear_directory(DATADIR)


geojson = '''
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [121.0017, 14.5361]
  },
  "properties": {}
}
'''
AOI = gpd.GeoDataFrame.from_features([json.loads(geojson)])
AOI = AOI.buffer(0.02)


# Set these variables to None and you will be prompted to enter your username and password below.
asf_username = 'mirasnickanthony'
asf_password = 'e44 4E6 E447E S56E!'
asf = ASF(asf_username, asf_password)
print(asf.download(DATADIR, SCENES))
S1.download_orbits(DATADIR, S1.scan_slc(DATADIR))
Tiles().download_dem(AOI, filename=DEM, skip_exist=False).plot.imshow(cmap='cividis')


# simple Dask initialization
if 'client' in globals():
    client.close()

client = Client()
#import psutil
#client = Client(n_workers=max(1, psutil.cpu_count() // 4), threads_per_worker=min(4, psutil.cpu_count()))


scenes = S1.scan_slc(DATADIR, subswath=SUBSWATH, polarization=POLARIZATION)
sbas = Stack(WORKDIR, drop_if_exists=True).set_scenes(scenes)
sbas.compute_reframe(AOI)
sbas.load_dem(DEM, AOI)
sbas.compute_align()
sbas.compute_geocode(1)


baseline_pairs = sbas.sbas_pairs(days=24)


sbas.set_landmask(None)
sbas.load_landmask('recurrence_120E_20Nv1_4_2021.tif')
landmask = (sbas.get_landmask()*-1)>-0.02
sbas.set_landmask(None)
landmask_ra = sbas.ll2ra(landmask)
landmask_ra


# load radar topography
topo = sbas.get_topo()
# load Sentinel-1 data
data = sbas.open_data()


WAVELENGTH = 20
COARSEN_GRID = (1, 4)
sbas.compute_interferogram_multilook(
    baseline_pairs,
    'intf_mlook',
    wavelength=WAVELENGTH,
    phase=sbas.phasediff(baseline_pairs, data, topo),
    coarsen=COARSEN_GRID
)


ds_sbas = sbas.open_stack('intf_mlook')
ds_sbas = ds_sbas.where(landmask_ra.interp_like(ds_sbas, method='nearest'))
intf_sbas = ds_sbas.phase
corr_sbas = ds_sbas.correlation
sbas_phase_goldstein = sbas.goldstein(intf_sbas, corr_sbas, 8)
intf15m = sbas.interferogram(sbas_phase_goldstein)


tqdm_dask(result := dask.persist(corr_sbas, intf15m), desc='Compute Phase and Correlation')
corr, intf = result
corr_ll = sbas.ra2ll(corr_sbas)


PRE_FLOOD_DOI = '2018-08-23 2018-09-04'
CO_FLOOD_DOI = '2018-09-04 2018-09-16'
corr_sbas_df = corr_ll.to_dataframe()
pre_flood_df = corr_sbas_df[corr_sbas_df.index.get_level_values(0) == PRE_FLOOD_DOI]
co_flood_df = corr_sbas_df[corr_sbas_df.index.get_level_values(0) == CO_FLOOD_DOI]
pre_flood_df.to_csv(f'csv/{PRE_FLOOD_DOI} {POLARIZATION}.csv')
co_flood_df.to_csv(f'csv/{CO_FLOOD_DOI} {POLARIZATION}.csv')
