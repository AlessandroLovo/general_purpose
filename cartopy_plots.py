# '''
# Created in November 2021

# @author: Alessandro Lovo
# '''
import numpy as np
import logging
import warnings
from collections import deque
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.animation import FuncAnimation, PillowWriter

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point as acp

logger = logging.getLogger(__name__)
logger.level = logging.INFO

path_to_here = str(Path(__file__).resolve().parent)
if not path_to_here in sys.path:
    sys.path.insert(1,path_to_here)

from utilities import significative_data

data_proj = ccrs.PlateCarree()

def Greenwich(*args):
    '''
    Adds a new copy of the Greenwich meridian at the end of a series of arrays. Useful for plotting data around the pole.
    
    USAGE:
        extended_array = Greenwich(array)
    or
        extended_lon, *list_of_extended_arrays = Greenwich(lon, *list_of_arrays)
    
    If a single argument is provided, the first 'column' is copied to the end
    
    If more arguments are provided the first one is assumed to be longitude data, for which the added column will be filled with the value 360
    '''
    if len(args) == 1:
        return acp(args[0])
    args = [acp(a) for a in args]
    args[0][...,-1] += 360 # fix the longitude
    return args

def draw_map(m, background='stock_img', **kwargs):
    '''
    Plots a background map using cartopy.
    Additional arguments are passed to the cartopy function gridlines
    
    Parameters:
    -----------
        m: cartopy axis
        resolution: either 'low' or 'high'
        **kwargs: arguments passed to cartopy.gridlines
    '''
    if 'draw_labels' not in kwargs:
        kwargs['draw_labels'] = True

    if background == 'stock_img':
        m.stock_img()
    elif background == 'land-sea':
        m.add_feature(cfeat.LAND)
        m.add_feature(cfeat.OCEAN)
        m.add_feature(cfeat.LAKES)
    else:
        if background != 'coastlines':
            warnings.warn(f"Unrecognized option {background = }, using 'coastlines' instead")
        m.coastlines()
    m.gridlines(**kwargs)

def geo_plotter(m, lon, lat, values, mode='contourf',
                 levels=None, cmap='RdBu_r', title=None,
                 put_colorbar=True, colorbar_label=None,
                 draw_coastlines=True, draw_gridlines=True, draw_labels=True,
                 greenwich=False, **kwargs):
    '''
    Contourf plot together with coastlines and meridians
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        mode : 'contour', 'contourf', 'scatter', 'pcolormesh'. By default 'contourf'
        levels: contour levels for the field values
        cmap: colormap
        title: plot title
        put_colorbar: whether to show a colorbar
        draw_coastlines: whether to draw the coastlines
        draw_gridlines: whether to draw the gridlines
        
        greenwich: if True automatically adds the Greenwich meridian to avoid gaps in the plot
    '''
    if greenwich and mode in ['scatter', 'pcolormesh']:
        logger.warning('Ignoring greenwich kwarg')
        greenwich = False
    if greenwich:
        _lon, _lat, _values = Greenwich(lon, lat, values)
    else:
        _lon, _lat, _values = lon, lat, values

    if mode == 'contourf':        
        im = m.contourf(_lon, _lat, _values, transform=data_proj,
                        levels=levels, cmap=cmap, extend='both', **kwargs)
    elif mode == 'contour':
        im = m.contour(_lon, _lat, _values, transform=data_proj,
                       levels=levels, cmap=cmap, extend='both', **kwargs)
    elif mode == 'pcolormesh':
        if levels is not None:
            logger.warning('Ignoring levels kwarg')
        im = m.pcolormesh(_lon, _lat, _values, transform=data_proj,
                          cmap=cmap, **kwargs)
    elif mode == 'scatter':
        if levels is not None:
            logger.warning('Ignoring levels kwarg')
        im = m.scatter(_lon.flatten(), _lat.flatten(), c=_values.flatten(), transform=data_proj,
                       cmap=cmap, **kwargs)
    else:
        raise ValueError(f'Unknown {mode = }')

    if draw_coastlines:
        m.coastlines()
    if draw_gridlines:
        m.gridlines(draw_labels=draw_labels)
    if put_colorbar:
        plt.colorbar(im, label=colorbar_label, extend='both')
    if title is not None:
        m.set_title(title, fontsize=20)
        
    return im
    
    
def geo_contourf(m, lon, lat, values,
                 levels=None, cmap='RdBu_r', title=None,
                 put_colorbar=True, colorbar_label=None,
                 draw_coastlines=True, draw_gridlines=True, draw_labels=True,
                 greenwich=False, **kwargs):
    '''
    Contourf plot together with coastlines and meridians
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        levels: contour levels for the field values
        cmap: colormap
        title: plot title
        put_colorbar: whether to show a colorbar
        draw_coastlines: whether to draw the coastlines
        draw_gridlines: whether to draw the gridlines
        
        greenwich: if True automatically adds the Greenwich meridian to avoid gaps in the plot
    '''
    return geo_plotter(m, lon, lat, values, mode='contourf',
                       levels=levels, cmap=cmap, title=title,
                       put_colorbar=put_colorbar, colorbar_label=colorbar_label,
                       draw_coastlines=draw_coastlines, draw_gridlines=draw_gridlines, draw_labels=draw_labels,
                       greenwich=greenwich, **kwargs)
        
def geo_contour(m, lon, lat, values, levels=None, cmap1='PuRd', cmap2=None, greenwich=False):
    '''
    Plots a contour plot with the possbility of having two different colormaps for positive and negative data
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        levels: contour levels for the field values
        cmap1: principal colormap
        cmap2: if provided negative values will be plotted with `cmap1` and positive ones with `cmap2`
        
        greenwich: if True automatically adds the Greenwich meridian to avoid gaps in the plot
    '''
    if greenwich:
        _lon, _lat, _values = Greenwich(lon, lat, values)
    else:
        _lon, _lat, _values = lon, lat, values
    
    if cmap2 is None: # plot with just one colormap
        im = m.contour(_lon, _lat, _values, transform=data_proj,
                       levels=levels, cmap=cmap1)
        return im
    else: # separate positive and negative data
        v_neg = _values.copy()
        v_neg[v_neg > 0] = 0
        imn = m.contour(_lon, _lat, v_neg, transform=data_proj,
                        levels=levels, cmap=cmap1, vmin=levels[0], vmax=0)
        v_pos = _values.copy()
        v_pos[v_pos < 0] = 0
        imp = m.contour(_lon, _lat, v_pos, transform=data_proj,
                        levels=levels, cmap=cmap2, vmin=0, vmax=levels[-1])
        return imn, imp
        
def geo_contour_color(m, lon, lat, values, t_values=None, t_threshold=None, levels=None,
                      colors=["sienna","chocolate","green","lime"], linestyles=["solid","dashed","dashed","solid"],
                      linewidths=[1,1,1,1], draw_contour_labels=True, fmt='%1.0f', fontsize=12, greenwich=False):
    '''
    Plots contour lines divided in four categories: in order
        significative negative data
        non-significative negative data
        non-significative positive data
        significative positive data
        
    Significance is determined by comparing the `t_values`, which is an array of the same shape of `lon`, `lat' and `values`, with `t_threshold`
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        t_values: 2D array of the t_field (significance). If None all data are considered significant
        t_threshold: float, t values above the threshold are considered significant. If None all data are considered significant
        levels: contour levels for the field values
        
        fmt: fmt of the inline contour labels
        fontsize: fontsize of the inline contour labels
        
        greenwich: if True automatically adds the Greenwich meridian to avoid gaps in the plot
        
    For the following see above for the order of the items in the lists
        colors
        linestyles
        linewidths
    '''
    if greenwich:
        _lon, _lat, _values = Greenwich(lon, lat, values)
        if t_values is not None and t_threshold is not None:
            _t_values = Greenwich(t_values)
        else:
            _t_values = None
    else:
        _lon, _lat, _values, _t_values = lon, lat, values, t_values
    
    # divide data in significative and non significative:
    data_sig, _ = significative_data(_values, _t_values, t_threshold, both=False, default_value=np.NaN)
    
    cn, cnl, cp, cpl = None, None, None, None
    plot_insignificant = t_values is not None and t_threshold is not None
    if plot_insignificant:
        # negative insignificant anomalies
        i = 1
        v_neg = _values.copy()
        v_neg[v_neg > 0] = 0
        cn = m.contour(_lon, _lat, v_neg, transform=data_proj,
                       levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
        if draw_contour_labels:
            cnl = m.clabel(cn, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
        # positive insignificant anomalies
        i = 2
        v_pos = _values.copy()
        v_pos[v_pos < 0] = 0
        cp = m.contour(_lon, _lat, v_pos, transform=data_proj,
                       levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
        if draw_contour_labels:
            cpl = m.clabel(cp, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
    
    # negative significant anomalies
    i = 0
    v_neg = data_sig.copy()
    v_neg[v_neg > 0] = 0
    cns = m.contour(_lon, _lat, v_neg, transform=data_proj,
                    levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
    if draw_contour_labels and not plot_insignificant:
        cnl = m.clabel(cns, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
    # positive significant anomalies
    i = -1
    v_pos = data_sig.copy()
    v_pos[v_pos < 0] = 0
    cps = m.contour(_lon, _lat, v_pos, transform=data_proj,
                    levels=levels, colors=colors[i], linestyles=linestyles[i], linewidths=linewidths[i])
    if draw_contour_labels and not plot_insignificant:
        cpl = m.clabel(cps, colors=[colors[i]], manual=False, inline=True, fmt=fmt, fontsize=fontsize)
    
    return cn, cnl, cp, cpl, cns, cps
        
def PltMaxMinValue(m, lon, lat, values, colors=['red','blue']):
    '''
    Writes on the plot the maximum and minimum values of a field.
    
    Parameters:
    -----------
        m: cartopy axis
        lon: 2D longidute array
        lat: 2D latitude array
        values: 2D field array
        colors: the two colors of the text, respectively for the min and max values
    '''
    # plot min value
    coordsmax = tuple(np.unravel_index(np.argmin(values, axis=None), values.shape))
    x, y = lon[coordsmax], lat[coordsmax]
    txtn = m.text(x, y, f"{np.min(values) :.0f}", transform=data_proj, color=colors[0])
    txtn.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    # plot max value
    coordsmax = tuple(np.unravel_index(np.argmax(values, axis=None), values.shape))
    x, y = lon[coordsmax], lat[coordsmax]
    txtp = m.text(x, y, f"{np.max(values) :.0f}", transform=data_proj, color=colors[1])
    txtp.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        
    return txtn, txtp

def ShowArea(lon_mask, lat_mask, field_mask, coords=[-7,15,40,60], **kwargs):
    '''
    Shows the grid points, colored with respect to a given field, for instance the area of the cell
    
    Parameters:
    ----------- 
        lon_mask: 2D array of longitude grid points
        lat_mask: 2D array of same shape as lon_mask with the latitudes
        field_mask: 2D field (e.g. area of the grid cells) array of same shape as lon_mask
        coords: limits of the plot in the format [min_lon, max_lon, min_lat, max_lat]

        **kwargs:
            projection: default ccrs.PlateCarree()
            background: 'coastlines' (default), 'stock_img' or 'land-sea'
            figsize: default (15,15)
            draw_labels: whether to show lat and lon labels, default True
            show_grid: whether to display the grid connecting data points, default True
            title: default 'Area of a grid cell'
            
    Returns:
    --------
        fig: Figure
        m: cartopy axis
    '''
    # extract additional arguments
    projection = kwargs.pop('projection', ccrs.PlateCarree())
    background = kwargs.pop('background', 'coastlines')
    figsize = kwargs.pop('figsize', (15,15))
    draw_labels = kwargs.pop('draw_labels', True)
    show_grid = kwargs.pop('show_grid', True)
    title = kwargs.pop('title', 'Area of a grid cell')
    
    fig = plt.figure(figsize=figsize)
    m = plt.axes(projection=projection)
    m.set_extent(coords, crs=ccrs.PlateCarree())
    
    draw_map(m, background, draw_labels=draw_labels)
    
    if show_grid:
        # make longitude monotonically increasing
        _lon_mask = lon_mask.copy()
        modify = False
        for i in range(lon_mask.shape[1] - 1):
            if lon_mask[0,i] > lon_mask[0,i+1]:
                modify = True
                break
        if modify:
            _lon_mask[:,:i+1] -= 360
        # print(_lon_mask)
        
        m.pcolormesh(_lon_mask, lat_mask, np.ones_like(lon_mask), transform=data_proj,
                     alpha=0.35, cmap='Greys', edgecolors='grey')
    
    im = m.scatter(lon_mask, lat_mask, c=field_mask, transform=data_proj,
                   s=500, alpha = .35, cmap='RdBu_r')
    plt.title(title)
    plt.colorbar(im)
    
    return fig, m


def multiple_field_plot(lon, lat, f, projections=ccrs.Orthographic(central_latitude=90), extents=None, figsize=(9,6), fig_num=None,
                        colorbar='individual', titles=None, **kwargs):
    '''
    Plots several fields

    Parameters
    ----------
    lon : np.ndarray
        longitude: either 1D or meshgridded
    lat : np.ndarray
        latitude: either 1D or meshgridded
    f : np.ndarray
        fields to plot, with shape (lat, lon, nfields)
    projections : ccrs.Projection or list[ccrs.Projection], optional
        projection to use for each field, by default ccrs.Orthographic(central_latitude=90)
    extents : tuple or list[tuple], optional
        extents to apply to each field, by default None
    figsize : tuple, optional
        figure size, by default (9,6)
    fig_num : int, optional
        figure number of the first field, by default None
    colorbar : 'individual', 'shared', 'disabled', optional
        How to plot the colorbar:
            'disabled': every field has its own colorbar, not centerd around 0
            'individual': every field has its own colorbar, centered around 0
            'shared': every field has the same colorbar, centered around 0
        by default 'individual'
    titles : str or list[str], optional
        titles for each field, by default None

    **kwargs:
        passed to geo_plotter
    '''
    
    if len(lon.shape) != len(lat.shape):
        raise ValueError('lon and lat must have the same number of dimensions')
    if len(lon.shape) == 1:
        lon, lat = np.meshgrid(lon, lat)

    if lon.shape != f.shape[:2]:
        raise ValueError('f must have the first 2 dimensions with the same shape of lon and lat')

    if len(f.shape) == 3:
        n_fields = f.shape[2]
    else:
        n_fields = 1

    # broadcast
    if not isinstance(projections, list):
        projections = [projections]*n_fields
    if not isinstance(extents, list):
        extents = [extents]*n_fields
    if not isinstance(titles, list):
        titles = [titles]*n_fields

    norm = None
    if colorbar == 'shared':
        mx = max(-np.min(f), np.max(f)) or 1
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=0., vmin=-mx, vmax=mx)

    for i in range(n_fields):
        _f = f[...,i]
        if colorbar == 'individual':
            mx = max(-np.min(_f), np.max(_f)) or 1
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=0., vmin=-mx, vmax=mx)

        if fig_num is not None:
            plt.close(fig_num + i)
            fig = plt.figure(figsize=figsize, num=fig_num + i)
        else:
            fig = plt.figure(figsize=figsize)

        m = fig.add_subplot(projection = projections[i])
        if extents[i]:
            m.set_extent(extents[i])

        geo_plotter(m, lon, lat, _f, title=titles[i], norm=norm, **kwargs)

        fig.tight_layout()

def mfp(lon, lat, f,
         projections=[
            ccrs.Orthographic(central_latitude=90),
            ccrs.Orthographic(central_latitude=90),
            ccrs.PlateCarree()
        ],
        fig_num=8,
        extents=[None, None, (-5, 10, 39, 60)],
        titles=['Temperature [K]', 'Geopotential [m]', 'Soil Moisture [m]'],
        mode='pcolormesh',
        **kwargs):
    '''Simply multiple field plot with useful default arguments'''
    return multiple_field_plot(lon, lat, f, projections=projections, fig_num=fig_num, extents=extents, titles=titles, mode=mode, **kwargs)


    


###### animations #######
def save_animation(ani, name, fps=1, progress_callback=lambda i, n: print(f'\b\b\b\b{i}', end=''), **kwargs):
    if not name.endswith('.gif'):
        name += '.gif'
    writer = PillowWriter(fps=fps)
    ani.save(name, writer=writer, progress_callback=progress_callback, **kwargs)
    
def animate(tau, lon, lat, temp=None, zg=None, temp_t_values=None, zg_t_values=None, t_threshold=None,
            temp_levels=None, zg_levels=None, frame_title='', greenwich=False, masker=None, weight_mask=None, **kwargs):
    '''
    Returns an animation of temperature and geopotential profiles. It is also possible to have a side plot with the evolution of the temperature in a given region.
    
    Parameters:
    -----------
        tau: 1D array with the days
        lon: 2D longitude array
        lat: 2D latitude array
        temp: 3D temperature array (with shape (len(tau), *lon.shape)) that will be plotted as a contourf. If not provided it isn't plotted
        zg: 3D geopotential array with the same shape as `temp` wich will be plotted as a contour. If not provided it isn't plotted
        
        temp_t_values: 3D array of the significance of the temperature. Optional, if not provided all temperature data are considered significant
        zg_t_values: 3D array of the significance of the geopotential. Optional, if not provided all geopotential data are considered significant
        t_threshold: float. t_values above `t_threshold` are considered significant
        
        temp_levels: contour levels for the temperature
        zg_levels: contour levels for the geopotential
        frame_title: pre-title to put on top of each frame. The total title will also say which day it is
        greenwich: whether to copy the Greenwich meridian to avoid gaps in the plot.
        
        masker: None or function that takes as input a single argument (array), and returns a slice of said array over the region of interest.
            For example it can be a partial of ERA_Fields.create_mask or another example could be 
                masker = lambda data : return data[..., 6:12, 2:15]
            If None only the geo_plot is produced. Otherwise a side plot with the evolution of the temperature is also produced
        weigth_mask: array with the same shape of the output of `masker` and having the sum of its elements equals to 1.
            It used to weight the temperature values over the region of interest to get a meaningful mean
        
        **kwargs:
            figsize
            projection: default ccrs.Orthographic(central_latitude=90)
            extent: [min_lon, max_lon, min_lat, max_lat]. Default [-180, 180, 40, 90]
            draw_grid_labels: default False
            temp_cmap: colormap for the temperature contourf, default 'RdBu_r'
            zg_colors: colors for the geopotential contour lines, default ["sienna","chocolate","green","lime"]
            zg_linestyles: linestyles for the geopotential contour lines, default ["solid","dashed","dashed","solid"]
            zg_linewidths: linewidths for the geopotential contour lines, default [1,1,1,1]
            draw_zg_labels: default True
            zg_label_fmt: default '%1.0f'
            zg_label_fontsize: default 12
            
            temp_threshold: a red threshold to put on the plot of the evolution of the temperature when a `masker`is provided
    '''
    
    default_figsize = (15,12)
    if masker is not None:
        default_figsize = (25,12)
    figsize = kwargs.pop('figsize', default_figsize)
    projection = kwargs.pop('projection', ccrs.Orthographic(central_latitude=90))
    extent = kwargs.pop('extent', [-180, 180, 40, 90])
    draw_grid_labels = kwargs.pop('draw_grid_labels', False)
    temp_cmap = kwargs.pop('temp_cmap', 'RdBu_r')
    zg_colors = kwargs.pop('zg_colors', ["sienna","chocolate","green","lime"])
    zg_linestyles = kwargs.pop('zg_linestyles', ["solid","dashed","dashed","solid"])
    zg_linewidths = kwargs.pop('zg_linewidths', [1,1,1,1])
    draw_zg_labels = kwargs.pop('draw_zg_labels', True)
    zg_label_fmt = kwargs.pop('zg_label_fmt', '%1.0f')
    zg_label_fontsize = kwargs.pop('zg_label_fontsize', 12)
    temp_threshold = kwargs.pop('temp_threshold', None)
    
    temp_sign, _ = significative_data(temp, temp_t_values, t_threshold, both=False)
    
    if zg_t_values is None:
        zg_t_values = [None]*temp.shape[0]
        
    nrows = 1
    if masker is not None:
        nrows = 2
        
    fig = plt.figure(figsize=figsize)
    m = fig.add_subplot(1,nrows,1,projection=projection)
    m.set_extent(extent, crs=data_proj)
    if temp is not None:
        geo_contourf(m, lon, lat, temp[0], levels=temp_levels, cmap=temp_cmap, put_colorbar=True, draw_coastlines=False, draw_gridlines=False, greenwich=greenwich)
    
    if masker is not None:
        # create side plot
        ax = fig.add_subplot(1,2,2)
        ax.set_xlabel('day')
        ax.set_ylabel('temperature')
        ax.set_xlim(tau[0], tau[-1])
        
        # draw thresholds
        ax.hlines([0], *ax.get_xlim(), linestyle='dashed', color='grey')
        if temp_threshold is not None:
            ax.hlines([temp_threshold], *ax.get_xlim(), linestyle='solid', color='red')
        
        # compute mask
        lon_mask = masker(lon)
        lat_mask = masker(lat)
        # make the lengitude monotonically increasing
        _lon_mask = lon_mask.copy()
        modify = False
        for i in range(lon_mask.shape[1] - 1):
            if lon_mask[0,i] > lon_mask[0,i+1]:
                modify = True
                break
        if modify:
            _lon_mask[:,:i+1] -= 360
        
        # This way we plot always just the last two points
        temp_ints = deque(maxlen=2)
        days = deque(maxlen=2)
    
    def _plot_frame(i):
        m.cla()
        m.coastlines()
        m.gridlines(draw_labels=draw_grid_labels)
        # plot significant temperature
        if temp_sign is not None:
            geo_contourf(m, lon, lat, temp_sign[i], levels=temp_levels, put_colorbar=False,
                         draw_coastlines=False, draw_gridlines=False, greenwich=greenwich)
        # plot geopotential
        if zg is not None:
            geo_contour_color(m, lon, lat, zg[i], zg_t_values[i], t_threshold, levels=zg_levels,
                              colors=zg_colors, linestyles=zg_linestyles, linewidths=zg_linewidths,
                              draw_contour_labels=draw_zg_labels, fmt=zg_label_fmt, fontsize=zg_label_fontsize,
                              greenwich=greenwich)
            # plot max and min values of the geopotential
            PltMaxMinValue(m, lon, lat, zg[i])
            m.set_title(f'{frame_title} day {tau[i]}')
        
        # plot mask over the region of interest
        if masker is not None:
            m.pcolormesh(_lon_mask, lat_mask, np.ones_like(lon_mask), transform=data_proj, alpha=0.35, cmap='Greys', edgecolors='grey')
            # plot temperature on the side plot
            temp_ints.append(np.sum(masker(temp[i])*weight_mask))
            days.append(tau[i])
            ax.plot(days, temp_ints, color='black')
        
    ani = FuncAnimation(fig, _plot_frame, frames=len(tau))
    return ani