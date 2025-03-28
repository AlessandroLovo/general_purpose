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
from functools import wraps

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

def is_monotonic(lon:np.ndarray) -> bool:
    return np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0)

def monotonize_longitude(lon:np.ndarray) -> np.ndarray:
    """
    Generate a new array of longitudes that is monotonically increasing.

    Parameters:
        lon (np.ndarray): The input array of longitudes.

    Returns:
        np.ndarray: The monotonized array of longitudes.

    Raises:
        AssertionError: If the input array is not 1-dimensional.
        ValueError: If more than one sign change is detected in the array.
    """
    assert len(lon.shape) == 1, 'Only 1D arrays are allowed'
    is_increasing = np.diff(lon) > 0
    nsc = len(set(list(is_increasing))) - 1 # number of sign changes
    if not nsc:
        return lon
    elif nsc > 1:
        raise ValueError('Only one sign change is allowed')

    is_overall_increasing = np.mean(is_increasing) > 0
    if not is_overall_increasing: # flip the array so it is overall increasing
        lon = lon[::-1]
        is_increasing = np.diff(lon) > 0

    lon[:np.argmin(is_increasing) + 1] -= 360

    if not is_overall_increasing: # flip the array back to decreasing
        lon = lon[::-1]

    return lon

def draw_map(m, background='stock_img', **kwargs):
    '''
    Plots a background map using cartopy.
    Additional arguments are passed to the cartopy function gridlines

    Parameters
    ----------
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
    Multi-mode geographical plot

    Parameters
    ----------
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
        draw_labels: whether to draw the tick labels with lon and lat
        greenwich: if True automatically adds the Greenwich meridian to avoid gaps in the plot

    Returns
    -------
    im : the plotted object
    '''
    orientation = kwargs.pop('orientation','vertical') # colorbar orientation
    extend = kwargs.pop('extend', 'both') # extend colorbar
    assert lon.shape == lat.shape == values.shape, 'lon, lat and values must have the same shape'

    if mode in ['scatter', 'pcolormesh']:
        if greenwich:
            logger.warning('Ignoring greenwich kwarg')
            greenwich = False
    else:
        # check that longitude is monotonically increasing
        for i in range(lon.shape[0]):
            if not is_monotonic(lon[i,:]):
                logger.warning('Longitude is not monotonic! Monotonizing it.')
                lon[i,:] = monotonize_longitude(lon[i,:])
    if greenwich:
        _lon, _lat, _values = Greenwich(lon, lat, values)
    else:
        _lon, _lat, _values = lon, lat, values

    if mode == 'contourf':
        im = m.contourf(_lon, _lat, _values, transform=data_proj,
                        levels=levels, cmap=cmap, extend=extend, **kwargs)
    elif mode == 'contour':
        im = m.contour(_lon, _lat, _values, transform=data_proj,
                       levels=levels, cmap=cmap, extend=extend, **kwargs)
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
        plt.colorbar(im, label=colorbar_label, extend=extend, orientation=orientation)
    if title is not None:
        m.set_title(title, fontsize=20)

    return im


def geo_contourf(m, lon, lat, values,
                 levels=None, cmap='RdBu_r', title=None,
                 put_colorbar=True, colorbar_label=None,
                 draw_coastlines=True, draw_gridlines=True, draw_labels=True,
                 greenwich=False, **kwargs):
    '''
    Contourf plot together with coastlines and meridians. Here just for backward compatibility. See geo_plotter

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

def significance_hatching(m, lon, lat, significance, hatches=('//', None), greenwich=False, **kwargs):
    """
    Generate a contour plot with significance hatching. This is meant to be used in conjunction with `geo_plotter`

    Parameters:
    - m: A map object.
    - lon: An array-like object representing the longitudes.
    - lat: An array-like object representing the latitudes.
    - significance: An array-like of bool object representing the whether the data is significant.
    - hatches: 2-ple of strings representing the hatching pattern, respectively for non-significant and significant values.
    - **kwargs: Additional keyword arguments to be passed to the contourf function.

    Returns:
    - A contour plot with significance hatching.
    """
    if greenwich:
        _lon, _lat, _sign = Greenwich(lon, lat, significance)
    else:
        _lon, _lat, _sign = lon, lat, significance
    return m.contourf(_lon, _lat, _sign, transform=data_proj, levels=[-0.5,0.5,1.5], colors='none', cmap=None, hatches=hatches, **kwargs)

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



def multiple_field_plot(lon, lat, f, significance=None, projections=ccrs.Orthographic(central_latitude=90), extents=None, cmaps='RdBu_r', fig=None, figsize=(9,6), fig_num=None, one_fig_layout=False,
                        colorbar='individual', mx=None, titles=None, apply_tight_layout=True, significance_hatches=('//', None), **kwargs):
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
    significance : np.ndarray[bool], optional
        Array of the same shape as f, with the significance of each pixel
    projections : ccrs.Projection or list[ccrs.Projection], optional
        projection to use for each field, by default ccrs.Orthographic(central_latitude=90)
    extents : tuple or list[tuple], optional
        extents to apply to each field, by default None
    figsize : tuple, optional
        figure size, by default (9,6)
    fig_num : int, optional
        figure number of the first field, by default None
    one_fig_layout : int or tuple, optional
        Layout to put all the fields in the same figure.
        If int:
            a 3 digit number: <n_rows><n_cols><start>. As an example 130 means 1 row 3 columns
            if <start> is non 0, the first <start> subplots will be empty
        If tuple:
            (<n_rows>, <n_cols>)
            Use this if you want to have more than 9 plots in the same figure
    colorbar : 'individual', 'shared', 'disabled', optional
        How to plot the colorbar:
            'disabled': every field has its own colorbar, not centerd around 0
            'individual': every field has its own colorbar, centered around 0
            'shared': every field has the same colorbar, centered around 0
        by default 'individual'
    mx : float or list[float], optional
        maximum color value, by default None, which means it is computed automatically.
    titles : str or list[str], optional
        titles for each field, by default None
    apply_tight_layout : bool, optional
        Whether to apply tight layout to the figure. Default True
    significance_hatches : tuple, optional
        Hatches to use for respectively for non-significant and significant pixels, by default ('//', None)

    **kwargs:
        passed to geo_plotter

    Returns
    -------
    ims : list
        list of the plotted objects, useful for accessing colorbars for example.
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
    if not isinstance(cmaps, list):
        cmaps = [cmaps]*n_fields

    if colorbar == 'shared':
        if isinstance(mx, list):
            raise ValueError('Cannot provide different mx values if colrbar is shared')
        if isinstance(cmaps, list):
            assert len(set(cmaps)) == 1, 'Cannot provide different cmaps if colorbar is shared'
    else:
        if not isinstance(mx, list):
            mx = [mx]*n_fields

    ims = []
    levels = kwargs.pop('levels', 7)
    if isinstance(levels, list):
        assert len(levels) == n_fields
    else:
        levels = [levels]*n_fields
    for i in range(n_fields):
        if levels[i] is None:
            levels[i] = 7

    norm = kwargs.pop('norm', None)
    if colorbar == 'shared':
        if mx is None:
            mx = np.nanmax(np.abs(f)) or 1
        if norm is None:
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=0., vmin=-mx, vmax=mx)
        else:
            logger.warning('Using provided norm')

        for i in range(n_fields):
            if isinstance(levels[i], int):
                levels[i] = np.linspace(-mx,mx, levels[i])
            else:
                logger.warning('Using provided levels, this may not guarantee a shared colorbar')

    put_colorbar = kwargs.pop('put_colorbar', True)
    common_colorbar = False
    if one_fig_layout:
        if colorbar == 'shared' and put_colorbar:
            put_colorbar = False
            common_colorbar = True
        if isinstance(one_fig_layout, int):
            if n_fields > 9:
                raise ValueError('Cannot put more than 9 subplots in a figure using an integer one_fig_layout. Switch to one_fig_layout=(n_rows, n_cols)')
            if one_fig_layout < 110 or one_fig_layout > 919:
                raise ValueError(f'Invalid {one_fig_layout = }')
            if np.prod([int(j) for j in str(one_fig_layout)[:2]]) - int(str(one_fig_layout)[-1]) < n_fields:
                logger.warning(f'The provided layout ({one_fig_layout}) cannot accommodate all the {n_fields} plots, switching to one that can (single row)')
                one_fig_layout = n_fields*10 + 100
        else:
            try:
                one_fig_layout = tuple(one_fig_layout)
            except:
                raise TypeError('one_fig_layout must be int or tuple')
            if len(one_fig_layout) != 2:
                raise ValueError('one_fig_layout needs to have exactly two elements: number of rows and number of columns')
            if np.prod(one_fig_layout) < n_fields:
                raise ValueError(f'Cannot accomodate {n_fields} subplots in a {one_fig_layout[0]} by {one_fig_layout[1]} grid!')

        if fig is None:
            plt.close(fig_num)
            fig = plt.figure(num=fig_num, figsize=figsize)

    if not isinstance(mx, list):
        mx = [mx]*n_fields

    for i in range(n_fields):
        _f = f[...,i]
        _norm = norm
        _mx = mx[i]
        if _mx is None:
            _mx = np.nanmax(np.abs(_f)) or 1
        if _norm is None and colorbar == 'individual':
            _norm = matplotlib.colors.TwoSlopeNorm(vcenter=0., vmin=-_mx, vmax=_mx)
        if isinstance(levels[i], int):
            levels[i] = np.linspace(-_mx,_mx, levels[i])

        if one_fig_layout:
            if isinstance(one_fig_layout, list):
                assert len(one_fig_layout) == n_fields
                ofl = one_fig_layout[i]
                if isinstance(ofl, int):
                    m = fig.add_subplot(ofl, projection=projections[i])
                else:
                    assert len(ofl) == 3
                    m = plt.subplot2grid(ofl[:2], ofl[-1], projection=projections[i])
            elif isinstance(one_fig_layout, int):
                m = fig.add_subplot(one_fig_layout + i + 1, projection=projections[i])
            else:
                m = plt.subplot2grid(one_fig_layout, np.unravel_index(i, one_fig_layout), projection=projections[i])
        else:
            if fig is not None:
                raise ValueError('Cannot provide fig if not using one_fig_layout')
            if fig_num is not None:
                plt.close(fig_num + i)
                fig = plt.figure(figsize=figsize, num=fig_num + i)
            else:
                fig = plt.figure(figsize=figsize)
            m = fig.add_subplot(111, projection=projections[i])

        if extents[i]:
            m.set_extent(extents[i])

        ims.append(geo_plotter(m, lon, lat, _f, title=titles[i], norm=_norm, levels=levels[i], cmap=cmaps[i], put_colorbar=put_colorbar, **kwargs))

        if significance is not None:
            significance_hatching(m, lon, lat, significance[...,i], hatches=significance_hatches, greenwich=kwargs.get('greenwich', False))

        if not one_fig_layout and apply_tight_layout:
            fig.tight_layout()


    if one_fig_layout:
        if common_colorbar:
            plt.colorbar(ims[-1], label=kwargs.pop('colorbar_label', None), extend=kwargs.get('extend','both'))
        if apply_tight_layout:
            fig.tight_layout()

    return ims

@wraps(multiple_field_plot)
def mfp(lon, lat, f,  # This functions maps to multiple_field_plot() and not the merge conflict multiple_field_plot2()
         projections=[
            ccrs.Orthographic(central_latitude=90),
            ccrs.Orthographic(central_latitude=90),
            ccrs.PlateCarree()
        ],
        fig_num=8,
        extents=[None, None, (-5, 10, 39, 60)],
        titles=['Temperature [K]', 'Geopotential [m]', 'Soil Moisture [m]'],
        mode='pcolormesh',
        greenwich=True,
        draw_gridlines=False, draw_labels=False,
        **kwargs):
    '''Simply multiple field plot with useful default arguments'''
    return multiple_field_plot(lon, lat, f,
                               projections=projections, fig_num=fig_num, extents=extents, titles=titles, mode=mode,
                               draw_gridlines=draw_gridlines, draw_labels=draw_labels, greenwich=greenwich,
                               **kwargs)

def multiple_field_plot2(lon, lat, f, projections=ccrs.Orthographic(central_latitude=90), extents=None,
                        figsize=(9,6), fig_num=None, figure=None, axes=None, levs=None, use_norm=True,
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
    figure : figure, optional
        The figure handle is provided
    axes : axes, optional,
        If the axes are provided then they would be reused, otherwise make new ones
    levs : list, optional,
        If provided it will mark the maximal and minimal value
    use_norm : bool, optional,
        If use_norm=False then levels=levs will be used instead
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
        if levs is None:
            mx = np.nanmax(np.abs(f)) or 1
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=0., vmin=-mx, vmax=mx)
        else:
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=np.mean(levs), vmin=levs[0], vmax=levs[-1])

    for i in range(n_fields):
        _f = f[...,i]
        if colorbar == 'individual':
            if levs is None:
                mx = np.nanmax(np.abs(_f)) or 1
                norm = matplotlib.colors.TwoSlopeNorm(vcenter=0., vmin=-mx, vmax=mx)
            else:
                norm = matplotlib.colors.TwoSlopeNorm(vcenter=np.mean(levs), vmin=levs[0], vmax=levs[-1])
        print(f'{norm = }')
        if figure is None:
            if fig_num is not None:
                plt.close(fig_num + i)
                fig = plt.figure(figsize=figsize, num=fig_num + i)
            else:
                fig = plt.figure(figsize=figsize)
        else:
            fig = figure
        if axes is None:
            m = fig.add_subplot(projection = projections[i])
            if extents[i]:
                m.set_extent(extents[i])
        else:
            m = axes
        if use_norm:
            geo_plotter(m, lon, lat, _f, title=titles[i], norm=norm, **kwargs)
        else:
            geo_plotter(m, lon, lat, _f, title=titles[i], levels=levs, **kwargs)

        fig.tight_layout()
        return m




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