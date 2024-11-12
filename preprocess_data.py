import sys, os
import xarray as xr
import numpy as np

import shutil
import pandas as pd
import geopandas as gpd
import json

from dask.distributed import Client
import dask
import seaborn as sns


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


# simple Dask initialization
if 'client' in globals():
    client.close()


client = Client(processes=False)


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


def get_sar_1_collections_from(file_path):
    with open (file_path, 'r') as f:
        scenes = f.readlines()
    return [f.strip() for f in scenes]


def readlines_in_file(file_path) -> list:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def concatenate_text_files(directory_path):
    # Initialize an empty string to store the concatenated content
    concatenated_content = []
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file has a '.txt' extension
        if filename.endswith('.txt') and filename.startswith('2020') is False:
            # Get the full file path
            file_path = os.path.join(directory_path, filename)
            # Open the text file and read its content
            for line in readlines_in_file(file_path):
                concatenated_content.append(line.strip())

    return concatenated_content


def doi(directory_path):
    # Initialize an empty string to store the concatenated content
    pre_flood_dates = []
    co_flood_dates = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith('.txt'):
            for line in readlines_in_file(file_path):
                if filename.startswith('co_flood'):
                    co_flood_dates.append(line)
                elif filename.startswith('pre_flood'):
                    pre_flood_dates.append(line)

    return pre_flood_dates, co_flood_dates


SUBSWATH = 2
POLARIZATION = 'VV'

SCENES = concatenate_text_files('data/bursts/manila/')
SCENES.reverse()

WORKDIR = 'raw_golden'
DATADIR = 'data_golden'
DEM = f'{DATADIR}/dem.nc'

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


def download_orbits():
    asf_username = 'mirasnickanthony'
    asf_password = 'e44 4E6 E447E S56E!'
    asf = ASF(asf_username, asf_password)
    print(asf.download(DATADIR, SCENES))
    S1.download_orbits(DATADIR, S1.scan_slc(DATADIR))
    Tiles().download_dem(AOI, filename=DEM, skip_exist=False)


def set_scenes():
    scenes = S1.scan_slc(DATADIR, subswath=SUBSWATH, polarization=POLARIZATION)
    sbas = Stack(WORKDIR, drop_if_exists=True).set_scenes(scenes)
    sbas.compute_reframe(AOI)
    sbas.load_dem(DEM, AOI)
    sbas.compute_align()
    sbas.compute_geocode(1)
    return sbas


def get_landmask(sbas):
    sbas.set_landmask(None)
    sbas.load_landmask('recurrence_120E_20Nv1_4_2021.tif')
    landmask = (sbas.get_landmask()*-1)>-0.02
    sbas.set_landmask(None)
    return sbas.ll2ra(landmask)


download_orbits()


sbas = set_scenes()
landmask_ra = get_landmask(sbas)


# load radar topography
topo = sbas.get_topo()
# load Sentinel-1 data
data = sbas.open_data()

baseline_pairs = sbas.sbas_pairs(days=24)

WAVELENGTH = 20
COARSEN_GRID = (1, 4)


intensity = sbas.multilooking(np.square(np.abs(data)), wavelength=WAVELENGTH, coarsen=COARSEN_GRID)
phase = sbas.phasediff(baseline_pairs, data, topo)
phase = sbas.multilooking(phase, wavelength=WAVELENGTH, coarsen=COARSEN_GRID)  # honestly I'm not sure what this does
corr = sbas.correlation(phase, intensity)
phase_goldstein = sbas.goldstein(phase, corr, 8)
interferogram = sbas.interferogram(phase_goldstein)
corr = sbas.correlation(phase_goldstein, intensity, 8)

tqdm_dask(result := dask.persist(corr, interferogram), desc='Compute Phase and Correlation')
corr, interferogram = result


corr = corr.where(landmask_ra.interp_like(corr, method='nearest'))
intensity = intensity.where(landmask_ra.interp_like(intensity, method='nearest'))
corr_ll = sbas.ra2ll(corr)
intensity_ll = sbas.ra2ll(intensity)


PRE_FLOOD_DOI = '2018-08-23 2018-09-04'
CO_FLOOD_DOI = '2018-09-04 2018-09-16'
corr_sbas_df = corr_ll.to_dataframe()
pre_flood_df = corr_sbas_df[corr_sbas_df.index.get_level_values(0) == PRE_FLOOD_DOI]
co_flood_df = corr_sbas_df[corr_sbas_df.index.get_level_values(0) == CO_FLOOD_DOI]
pre_flood_df.to_csv(f'csv/coherence/{PRE_FLOOD_DOI} {POLARIZATION}.csv')
co_flood_df.to_csv(f'csv/coherence/{CO_FLOOD_DOI} {POLARIZATION}.csv')
co_flood_df.to_csv(f'csv//coherence/{CO_FLOOD_DOI} {POLARIZATION}.csv')
print('Saved Dataframes')
