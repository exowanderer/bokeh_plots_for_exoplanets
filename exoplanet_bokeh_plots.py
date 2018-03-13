import kplr
import numpy as np
# import matplotlib.pyplot as plt

from functools import partial
from lmfit     import Parameters, Minimizer, report_errors, minimize

from matplotlib import rcParams
from time      import time

from sklearn.externals import joblib
from scipy import special
from pandas import DataFrame
from exoparams import PlanetParams
import batman

from bokeh.io       import output_notebook, show
from bokeh.plotting import figure
from bokeh.models   import Span
from bokeh.layouts  import gridplot

# output_notebook()

def batman_wrapper_lmfit(period, tCenter, inc, aprs, rprs, edepth, ecc, omega, u1, u2, 
                         intcpt, slope, crvtur, times, ldtype='quadratic', transitType='primary'):
    
    if intcpt == 1.0 and slope == 0.0 and crvtur == 0.0:
        OoT_crvtur = 1.0 # OoT == Out of Transit
    else:
        OoT_crvtur = intcpt + slope*(times-times.mean()) + crvtur*(times-times.mean())**2
    
    bm_params           = batman.TransitParams() # object to store transit parameters
    
    bm_params.per       = period   # orbital period
    bm_params.t0        = tCenter  # time of inferior conjunction
    bm_params.inc       = inc      # inclunaition in degrees
    bm_params.a         = aprs     # semi-major axis (in units of stellar radii)
    bm_params.rp        = rprs     # planet radius (in units of stellar radii)
    bm_params.fp        = edepth   # planet radius (in units of stellar radii)
    bm_params.ecc       = ecc      # eccentricity
    bm_params.w         = omega    # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype   # limb darkening model # NEED TO FIX THIS
    bm_params.u         = [u1, u2] # limb darkening coefficients # NEED TO FIX THIS
    
    m_eclipse = batman.TransitModel(bm_params, times, transittype=transitType)# initializes model
    
    return m_eclipse.light_curve(bm_params)*OoT_crvtur

def transit_line_model(model_params, times):
    intcpt  = model_params['intcpt'].value if 'intcpt' in model_params.keys() else 1.0
    slope   = model_params['slope'].value  if 'slope'  in model_params.keys() else 0.0
    crvtur  = model_params['crvtur'].value if 'crvtur' in model_params.keys() else 0.0
    
    # Transit Parameters
    period  = model_params['period'].value
    tCenter = model_params['tCenter'].value
    inc     = model_params['inc'].value
    aprs    = model_params['aprs'].value
    edepth  = model_params['edepth'].value
    tdepth  = model_params['tdepth'].value
    ecc     = model_params['ecc'].value
    omega   = model_params['omega'].value
    u1      = model_params['u1'].value
    u2      = model_params['u2'].value
    
    # delta_phase = deltaphase_eclipse(ecc, omega) if ecc is not 0.0 else 0.5
    # t_secondary = tCenter + period*delta_phase
    
    rprs  = np.sqrt(tdepth)
    
    transit_model = batman_wrapper_lmfit(period, tCenter, inc, aprs, rprs, edepth, ecc, omega, u1, u2, 
                         intcpt, slope, crvtur, times, ldtype='quadratic', transitType='primary')
    
    line_model    = intcpt + slope*(times-times.mean()) + crvtur*(times-times.mean())**2.
    
    return transit_model * line_model

def bokeh_errorbars(xs, ys, yerrs, xerrs=None, color='#1f77b4', size=5, alpha=1, fig=None, show_now = False):
    
    if xerrs is None:
        xerrs = np.zeros(xs.size)
    
    if fig is None:
        fig = figure()
    
    fig.circle(xs, ys, color=color, size=size, alpha=alpha)
    
    # create the coordinates for the errorbars
    err_xs = []
    err_ys = []
    
    for x, y, yerr, xerr in zip(xs, ys, yerrs, xerrs):
        err_xs.append((x - xerr, x + xerr))
        err_ys.append((y - yerr, y + yerr))
    
    # plot them
    fig.multi_line(err_xs, err_ys, color=color, alpha=alpha)
    
    if show_now: show(fig)
    
    return fig

def bokeh_hist(data, bins=100, range=None, color='#1f77b4', density=True, alpha=1.0, fig = None, show_now = False):
    if fig is None:
        fig = figure()
    
    data_sorted = np.sort(data)
    hist, edges = np.histogram(data_sorted, density=density, bins=bins, range=range)
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], 
             fill_color=color, line_color=color, alpha=alpha)
    
    if show_now: show(fig)
    
    return fig

def bokeh_corner_plot(dataset, TOOLS=None, hist_color='orange', kde_color="violet"):
    if isinstance(dataset, np.ndarray):
        dataset = DataFrame(dataset)
    
    if TOOLS is None:
        TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help"
    
    scatter_plots = []
    y_max = len(dataset.columns) - 1
    for i, y_col in enumerate(dataset):
        for j, x_col in enumerate(dataset):
            df = DataFrame({x_col: dataset[x_col].tolist(), y_col: dataset[y_col].tolist()})
            fig = figure(tools=TOOLS, toolbar_location="below", toolbar_sticky=False)
            if i >= j:
                if i != j:
                    fig.scatter(x=x_col, y=y_col, source=df)
                else:
                    x_now       = np.sort(dataset[x_col].values)
                    mu  , sigma = np.mean(x_now), np.std(x_now)
                    hist, edges = np.histogram(x_now, density=True, bins=len(x_now)//100)
                    pdf         = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x_now-mu)**2 / sigma**2)
                    cdf         = 0.5*(1+special.erf((x_now-mu)/np.sqrt(2*sigma**2)))
                    
                    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],fill_color=hist_color, line_color=hist_color, alpha=1.0)
                    fig.line(x_now, pdf, line_color=kde_color, line_width=8, alpha=0.7)#, legend="PDF")
                    #fig.line(x_now, cdf, line_color="black"  , line_width=2, alpha=0.5, legend="CDF")
                if j > 0:
                    fig.yaxis.axis_label = ""
                    fig.yaxis.visible = False
                if i < y_max:
                    fig.xaxis.axis_label = ""
                    fig.xaxis.visible = False
            else:
                fig.outline_line_color = None
            
            scatter_plots.append(fig)
    
    grid = gridplot(scatter_plots, ncols = len(dataset.columns))
    show(grid)