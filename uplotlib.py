import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm.notebook import tqdm
import uncertainties as unc
from scipy import stats

import logging
import sys

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO


def plot(*args, ax=None, **kwargs):
    '''
    Quick interface to the plt.errorbar function:
    
    Parameters:
        *args:
            x: optional, if given array-like. If it is an array of ufloats, xerrorbars are plotted
            y: array-like. If it is an array of ufloats yerrorbars are plotted
        
        ax: optional, axis on which to execute the plot
        
        **kwargs: passed to plt.errorbar, e.g. 'fmt'
        
    Returns: ErrorbarContainer
        
    '''
    if len(args) == 2:
        xs = args[0]
        ys = args[1]
    else:
        ys = args[0]
        xs = np.arange(len(ys))

    x_val = nominal_value(xs)
    x_err = std_dev(xs)
    if np.nansum(x_err) == 0:
        x_err = None
    y_val = nominal_value(ys)
    y_err = std_dev(ys)
    if np.nansum(y_err) == 0:
        y_err = None

    if ax is None:
        ax = plt
    
    return ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, **kwargs)

def errorband(x, y, ax=None, band_alpha=0.5, **kwargs):
    '''
    Plots a curve with an errorband

    Parameters
    ----------
    x : array-like of floats or ufloats
        x array. If of ufloats a vertical errorband will be plotted
    y : array-like of floats or ufloats
        y array. If of ufloats a horizontal errorband will be plotted
    ax : plt.Axis object, optional
        axis over which to perform the plot, by default None
    band_alpha : float, optional
        transparency of the errorband, by default 0.5

    Returns
    -------
    line :
        output of plt.plot
    shadey :
        output of plt.fill_between
    shadex :
        output of plt.fill_betweenx
    '''
    color = kwargs.pop('color', None)
    if ax is None:
        ax = plt
    x_val = nominal_value(x)
    y_val = nominal_value(y)
    line = ax.plot(x_val, y_val, color=color, **kwargs)
    if color is None:
        color = plt.gca().line[-1].get_color()

    shadey = None
    y_std = std_dev(y)
    if (y_std > 0).any():
        shadey = ax.fill_between(x_val, y_val - y_std, y_val + y_std, color=color, alpha=band_alpha)

    shadex = None
    x_std = std_dev(x)
    if (x_std > 0).any():
        shadex = ax.fill_betweenx(y_val, x_val - x_std, x_val + x_std, color=color, alpha=band_alpha)

    return line, shadey, shadex


class DataMisaligner():
    def __init__(self, x_misalignment=0):
        self.xm = x_misalignment
        self.data = []
        self.data_kwargs = []
        
    def add(self, *args, **kwargs):
        kwargs.pop('ax')
        if len(args) == 2:
            xs = args[0]
            ys = args[1]
        else:
            ys = args[0]
            xs = np.arange(len(ys))
        
        if isinstance(xs[0], unc.core.AffineScalarFunc):
            x_d = np.array([[x.n, x.s] for x in xs])
            x_val = x_d[:,0]
            x_err = x_d[:,1]
        else:
            x_val = xs
            x_err = None
        if isinstance(ys[0], unc.core.AffineScalarFunc):
            y_d = np.array([[y.n, y.s] for y in ys])
            y_val = y_d[:,0]
            y_err = y_d[:,1]
        else:
            y_val = ys
            y_err = None
            
        self.data.append([x_val, y_val, x_err, y_err])
        self.data_kwargs.append(kwargs)
        
    def plot(self, ax):
        # find multiplicities of the x coordinate
        x_values = []
        multiplicities = []
        for dataset in self.data:
            for x_val in dataset[0]:
                if x_val not in x_values:
                    x_values.append(x_val)
                    multiplicities.append(1)
                else:
                    multiplicities[x_values.index(x_val)] += 1
                    
        # misalign the data
        misaligned_xs = []
        current_multiplicities = []
        current_x_values = []
        for dataset in self.data:
            misaligned_xs.append([])
            for i,x_val in enumerate(dataset[0]):
                if x_val not in current_x_values:
                    current_x_values.append(x_val)
                    current_multiplicities.append(1)
                else:
                    current_multiplicities[current_x_values.index(x_val)] += 1
                    
                cm = current_multiplicities[current_x_values.index(x_val)]
                m = multiplicities[current_x_values.index(x_val)]
                    
                misaligned_xs[-1].append(x_val + self.xm*(cm - 1 - 0.5*(m - 1)))
                
        
        # plot
        for i, dataset in enumerate(self.data):
            _, y_val, x_err, y_err = dataset
            x_val = misaligned_xs[i]
            kwargs = self.data_kwargs[i]
            ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, **kwargs)
            
        # add vertical lines where data was misaligned
        ax.vlines([x_values[i] for i,m in enumerate(multiplicities) if m > 1], *ax.get_ylim(), color='black', linestyle='dotted', alpha=0.2)
        
        
                



class ExtendedKDE():
    def __init__(self, xs, ignore_Dirac_deltas=True, normalize=True):
        x_d = np.array([[x.n, x.s] for x in xs])
        x_val = x_d[:,0]
        x_err = x_d[:,1]
        
        min_idx = np.argmin(x_val)
        max_idx = np.argmax(x_val)
        self.min_x = x_val[min_idx] - 2*x_err[min_idx]
        self.max_x = x_val[max_idx] + 2*x_err[max_idx]
        self.gaussians = None
        self.normalize = normalize
        
        if ignore_Dirac_deltas:
            self.gaussians = [stats.norm(*p) for p in x_d if p[1] > 0]
        
        else:
            self.gaussians = [stats.norm(*p) for p in x_d]
        
    def __call__(self, point):
        if self.normalize:
            return np.mean([g.pdf(point) for g in self.gaussians])
        return np.sum([g.pdf(point) for g in self.gaussians])
    
    def evaluate(self, points):
        '''
        Performs a kde histogram of ufloat data

        Parameters:
            points: int or array-like:
                if int: number of points that are computed
                if array-like: array of points where to sample
        '''
        return points, np.array([self(point) for point in tqdm(points)])
    

    def plot(self, points=30, ax=None, xrange=None, switch_xy=False, **kwargs):
        '''
        Performs a kde histogram of ufloat data

        Parameters:
            points: int or array-like:
                if int: number of points that are computed
                if array-like: array of points where to sample
            ax: optional, axis on which to execute the plot
        '''
        if isinstance(points, int):
            if xrange is not None:
                points = np.linspace(*xrange, points)
            else:
                points = np.linspace(self.min_x, self.max_x, points)
            
        points, values = self.evaluate(points)
        
        if ax is not None:
            if switch_xy==False:
                ax.plot(points, values, **kwargs)
            else: 
                ax.plot(values, points, **kwargs)
        else:
            plt.plot(points,values, **kwargs)

        return points, values
    

    
def side_hist_plot(xdata, ydata, bins=30, external_axes=None, fit=True, **kwargs):
    '''
    Makes a plot of 'ydata' vs 'xdata' with a kde plot of 'ydata' on its right.
    
    Params:
        'xdata': array-like
        'ydata': array-like of (possibly) ufloats. If ufloats, the side histogram will be made using ExtendedKDE, otherwise with hist
        'bins': number of points for the kde plot
        'external_axes': tuple of size 2, external axes on which to do the plot
        'fit': whether to fit the histogram with a gaussian. Default True.
        
        **kwargs:
            figsize
            xlabel
            ylabel
            title
            
            label: label to create a legend in the plot
            fit_color: color of the fit line
            fit_linestyle
            
            adjust_ylims: bool. If True plot and histogram will have the same ylims
            
    Returns:
        fig: plt.figure
        axes: (ax_plot, ax_hist)
        fit_params: array of ufloats: [center, sigma] of the gaussian used for the fit.
    '''
    
    figsize = kwargs.pop('figsize', (10,7))
    num = kwargs.pop('num', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    title = kwargs.pop('title', None)
    
    label = kwargs.pop('label', None)
    fit_color = kwargs.pop('fit_color', 'red')
    fit_linestyle = kwargs.pop('fit_linestyle', None)
    
    adjust_ylims = kwargs.pop('adjust_ylims', False)
    
    
    
    fig = None
    ax_plot = None
    ax_hist = None
    
    if external_axes is None:
        if num is not None:
            plt.close(num)
        fig = plt.figure(figsize=figsize, num=num)
        gs = GridSpec(4,5)

        ax_plot = fig.add_subplot(gs[:,0:3])
        ax_hist = fig.add_subplot(gs[:,3:])
    
    else:
        ax_plot, ax_hist = external_axes
        fig = ax_plot.figure
    
    # plot
    plot(xdata, ydata, ax=ax_plot, label=label, **kwargs)
    if label is not None:
        ax_plot.legend()
    if xlabel is not None:
        ax_plot.set_xlabel(xlabel)
    if ylabel is not None:
        ax_plot.set_ylabel(ylabel)
    if title is not None:
        ax_plot.set_title(title)
    if adjust_ylims:
        ax_hist.set_ylim(*ax_plot.get_ylim())
        
    # hist
    if isinstance(ydata[0], unc.core.AffineScalarFunc):
        kernel = ExtendedKDE(ydata)
        ax_hist.set_xlabel('Probability')
        
        x1,f1 = kernel.plot(points=bins, xrange=ax_plot.get_ylim(), ax=ax_hist, switch_xy=True, **kwargs)
    else:
        xrange = ax_plot.get_ylim()

        f1, x1 = np.histogram(ydata, bins=np.linspace(*xrange, bins+1), density=True)
        x1 = 0.5*(x1[1:] + x1[:-1])
        ax_hist.plot(f1, x1, **kwargs)
    
    plt.setp(ax_hist.get_yticklabels(), visible=False) #Turn off tick labels
    
    fit_params = None
    if fit:
        try:
            from lmfit.models import GaussianModel
        except ImportError as e:
            raise ImportError('You need lmfit package to do this gaussian fit') from e
        # gaussian fit
        mod1 = GaussianModel(prefix='g1_')
        pars1 = mod1.guess(f1, x=x1)
        out1 = mod1.fit(f1, pars1, x=x1)
        ax_hist.plot(out1.best_fit, x1, color=fit_color, linestyle=fit_linestyle)
        
        # get fit parameters
        dist1 = unc.ufloat(out1.params['g1_center'].value, out1.params['g1_center'].stderr)
        sigma_dist1 = unc.ufloat(out1.params['g1_sigma'].value, out1.params['g1_sigma'].stderr)
        
        fit_params = np.array([dist1, sigma_dist1])
    
    return fig, (ax_plot, ax_hist), fit_params


def frmt(v, float_digits=2, ufloat_digits=1):   
    if type(v) == str:
        return v
    if type(v) in [unc.core.Variable, unc.core.AffineScalarFunc]:
        return f'${v:.{ufloat_digits}uL}$'
    if type(v) in [int, np.int64]:
        return f'${v}$'
    return f'${v:.{float_digits}f}$'

def vectorized_frmt(x:np.ndarray, float_digits=2, ufloat_digits=1):
    _x = x.flatten()
    _x = np.array([frmt(v, float_digits, ufloat_digits) for v in _x])
    return _x.reshape(x.shape)

def df2latex(df, float_digits=2, ufloat_digits=1):
    '''
    Creates a latex table from a pandas.DataFrame object that can contain ufloat values

    Params:
        df : pandas.DataFrame
        float_digits: int, number of decimal digits for float values
        ufloat_digits: int, number of significant digits of the error in ufloat values
    '''
    
    for i in range(len(df)):
        s = ''
        for key in df.keys():
            v = df.iloc[i][key]
            s += frmt(v, float_digits, ufloat_digits)
            s += ' & '
            
        s = s[:-2] +  '\\\\'
        print(s)

@np.vectorize
def ufloatify(n,s):
    return unc.ufloat(n,s)
        
@np.vectorize
def nominal_value(x):
    '''
    Returns the nominal value of an object if it has one (e.g. it is an uncertainties.core.AffineScalarFunc),
    or if it is convertible to an uncertainties.core.Variable (e.g. '12+/-3').
    Otherwise returns the object itself.
    
    If provided an array, performs the operation elementwise.
    '''
    if isinstance(x, str):
        if 'j' in x:
            return complex(x)
        try:
            return unc.ufloat_fromstr(x).n
        except:
            logger.warning(f'Could not parse string {x}, returning it')
            return x
    if isinstance(x, unc.core.AffineScalarFunc):
        return x.n
    return float(x)

@np.vectorize
def std_dev(x):
    '''
    Returns the standard deviation of an object if it has one (e.g. it is an uncertainties.core.AffineScalarFunc),
    or if it is convertible to an uncertainties.core.Variable (e.g. '12+/-3').
    Otherwise returns 0.
    
    If provided an array, performs the operation elementwise.
    '''
    if isinstance(x, str):
        if 'j' in x: # complex number
            return 0.
        try:
            return unc.ufloat_fromstr(x).s
        except:
            logger.warning(f'Could not parse string {x}, assuming std is 0')
            return 0.
    if isinstance(x, unc.core.AffineScalarFunc):
        return x.s
    return 0.

def avg(x:np.ndarray, weights=None, axis=None, mean_std=False):
    '''
    Returns the average value of an array-like with error accounting fro the dispersion.

    Parameters
    ----------
    x : array-like
        array of values
    weights : array-like, optional
        array of the same shape as x to be used as weights for the mean and standard deviation
    axis : int
        axis over which to perform the average, if None the array is first flattened
    mean_std : bool, optional
        Whether the error should be the dispersion of the sample (np.std(x)) or the error of the mean (np.std(x)/np.sqrt(len(x))), by default False

    Returns
    -------
    unc.ufloat
        mean +/- std
    '''
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if weights is None:
        if axis is None:
            x = x.reshape(-1)
            axis = 0
        m = np.mean(x,axis=axis)
        s = np.std(x,axis=axis)
        if mean_std:
            s /= np.sqrt(x.shape[axis])
    else:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        if weights.shape != x.shape:
            weights = np.ones_like(x)*weights # broadcast
        if axis is None:
            x = x.reshape(-1)
            weights = weights.reshape(-1)
            axis = 0

        weights /= np.sum(weights, axis=axis)

        m = np.sum(weights*x, axis=axis)
        s = np.sqrt(np.sum(weights*(x - m)**2, axis=axis))
        if mean_std:
            s /= np.sqrt(np.sum(weights != 0, axis=axis)) # count only non-zero weights
    
    return ufloatify(m,s)

def xr_avg(da, dim):
    '''
    Apply function `avg` to an xarray dataarray `da` along dimension(s) `dim`

    Parameters
    ----------
    da : xr.DataArray
        dataarray
    dim : str or list[str]
        dimension(s) along which to apply `avg`

    Returns
    -------
    xr.DataArray[ufloat]
        averaged dataarray
    '''
    import xarray as xr
    if not isinstance(dim, list):
        dim = [dim]
    return xr.apply_ufunc(lambda x: avg(x).item(), da, input_core_dims=[dim], exclude_dims=set(dim), vectorize=True)

def mad(x:np.ndarray, axis=None):
    '''
    Returns the median of the absolute deviations from the median. Useful for robust estimates.
    '''
    m = np.median(x, axis=axis)
    return np.median(np.abs(x - m), axis=axis)

def robust_avg(x:np.ndarray, axis=None):
    m = np.median(x, axis=axis)
    s = 1.4826*mad(x, axis=axis)
    return ufloatify(m,s)


def _safe_ufloat_fromstr(x):
    '''
    Safe and vectorized version of uncertainties.ufloat_fromstr
    '''
    if not isinstance(x, str):
        return x
    try:
        return unc.ufloat_fromstr(x)
    except ValueError:
        return x
    
safe_ufloat_fromstr = np.vectorize(_safe_ufloat_fromstr, otypes=[object])

def convert_df(df, discard_errors=False):
    '''
    Converts ufloats reported as strings in a pandas.DataFrame into proper ufloats.
    
    Returns a new DataFrame, the original one isn't modified.
    '''
    new_df = pd.DataFrame([])
    for column in df.columns:
        c = np.array(df[column])
        c = safe_ufloat_fromstr(c)
        if discard_errors:
            c = nominal_value(c)
        new_df[column] = c
        
    return new_df

def split_ufloats_in_df(df):
    '''
    Splits every column of a dataframe that contains ufloats into two columns:
    one with the nominal values and one with the stderr.
    Columns with non ufloat types are let unaffected.
    
    Returns a new DataFrame, the original one isn't modified.
    '''
    new_df = pd.DataFrame([])
    for column in df.columns:
        c = df[column]
        new_df[column] = nominal_value(c)
        if any(isinstance(val, unc.core.Variable) for val in c):
            sc = std_dev(c)
            if any(sc != 0):
                new_df[f'{column}_err'] = sc
    return new_df