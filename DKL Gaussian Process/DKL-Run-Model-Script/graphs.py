from matplotlib import colors
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from helper_funcs import global_mean

dict = {
    'tas': {
        'title': "TAS",
        'cbar_kwargs': {"label":"Temperature change / K"},
    },
    'pr': {
    'title': "PR",
        'cbar_kwargs': {"label":"Precipitation change / mm/day"},
    },
    'pr90': {
        'title': "PR90",
        'cbar_kwargs': {"label":"Precipitation change / mm/day"},
    },
    'dtr': {
        'title': "Diurnal Temperature Range",
        'cbar_kwargs': {"label":"Temperature change / K"},
    }
}

def plot_maps(mean_best, truth, args):
    # plotting predictions
    divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)
    diffnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2)

    ## Temperature
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 3))
    fig.suptitle(dict[args['variable']]['title'])

    # Test
    plt.subplot(131, projection=proj)
    truth.sel(time=slice(2080,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                                cbar_kwargs=dict[args['variable']]['cbar_kwargs'])
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='True')

    # Emulator
    plt.subplot(132, projection=proj)
    mean_best.sel(time=slice(2080,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                        cbar_kwargs=dict[args['variable']]['cbar_kwargs'])
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='GP posterior mean')

    # Difference
    difference = truth - mean_best
    plt.subplot(133, projection=proj)
    difference.sel(time=slice(2080,None)).mean('time').plot(cmap="bwr", norm=diffnorm,
                    cbar_kwargs=dict[args['variable']]['cbar_kwargs'])
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='Difference')

    # Save
    plt.savefig(args['model_output_dir'] + dict[args['variable']]['title'] + '.png', bbox_inches='tight')


def plot_timeseries(mean_best, truth, args):
    plt.figure(figsize=(10, 6))
    global_mean(truth).plot(label="Truth")
    global_mean(mean_best).plot(label='GP')
    plt.title(dict[args['variable']]['title'] + " Global Mean Timeseries")
    plt.ylabel('Global Mean Value')
    plt.xlabel('Year')
    plt.legend()
    plt.grid(True)
    plt.savefig(args['model_output_dir'] + 'global_mean_timeseries.png', bbox_inches='tight')