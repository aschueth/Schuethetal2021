import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from datetime import datetime
from matplotlib.colors import Normalize
import numpy.ma as ma
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
import cmocean
import operator
import gc
import xarray as xr
from xarray.backends import NetCDF4DataStore
from scipy.ndimage import map_coordinates
import matplotlib.patheffects as PathEffects
from CM1calc import *
import dask
import dask_image.ndfilters
import dask_image.ndmeasure
import dask.array as da
from metpy import constants as mpconsts
from skimage.measure import label
from skimage.morphology import erosion, dilation, opening, closing, disk
import scipy
import pandas as pd
import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.2f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format
                
SMALL_SIZE = 25
MEDIUM_SIZE = 40
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('legend', handlelength=5)


x0,y0,z0 = 7., 0., 0. #kilometers from the origin, aka where the radar is located
z0=0.
RHIlength=13. #length of the RHI
bins=1000 #This is the number of points along the line, "bins" if you will, bin length 10-15 m in the Ka
zenithres=0.3 #angle resolution in degrees, this is the correct resolution of the Ka
zenithmax=20.1 #max angle of the RHI

azimuth=158.19858
y0 = 0
x0 = 9

RHIlength=20

x1=x0+(RHIlength*np.cos(np.deg2rad(azimuth)))
y1=y0+(RHIlength*np.sin(np.deg2rad(azimuth)))

xbeg=min([x0,x1])
xend=max([x0,x1])
ybeg=min([y0,y1])
yend=max([y0,y1])

filr = 625

print('Beginning to read data in')
filename = '/lustre/research/weiss/schueth/resdep125m/cm1out_000'+str(filr)+'.nc'
ds = xr.open_dataset(filename,chunks={'ni':288,'nj':288})
time = ds['time'].values
x = ds['xh']
y = ds['yh']
z = ds['z']
w = ds['winterp']
zvort = ds['zvort']


X,Y=np.meshgrid(x.values,y.values)
dxx=dx(x)

xsmall = x.isel(ni=slice(find_nearest(x,xbeg)-2,find_nearest(x,xend)+2))
ysmall = y.isel(nj=slice(find_nearest(y,ybeg)-2,find_nearest(y,yend)+2))

th_RHI = ds['th'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
qr_RHI = ds['qr'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
qi_RHI = ds['qi'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
qc_RHI = ds['qc'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
qs_RHI = ds['qs'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
qg_RHI = ds['qg'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
qv_RHI = ds['qv'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]

th_corner = ds['th'].isel(time=0,ni=-1,nj=0).values
qr_corner = ds['qr'].isel(time=0,ni=-1,nj=0).values
qi_corner = ds['qi'].isel(time=0,ni=-1,nj=0).values
qc_corner = ds['qc'].isel(time=0,ni=-1,nj=0).values
qs_corner = ds['qs'].isel(time=0,ni=-1,nj=0).values
qg_corner = ds['qg'].isel(time=0,ni=-1,nj=0).values
qv_corner = ds['qv'].isel(time=0,ni=-1,nj=0).values

trhop_RHI=perturbation(trho(th_RHI,qv_RHI,qi_RHI,qc_RHI,qs_RHI,qr_RHI,qg_RHI),trho(th_corner,qv_corner,qi_corner,qc_corner,qs_corner,qr_corner,qg_corner))


crosstrp=np.zeros((len(np.arange(0.0125,0.5126,0.0125)),len(np.linspace(x0,x1,320))))
xplot_x=np.zeros((len(np.arange(0.0125,0.5126,0.0125)),len(np.linspace(x0,x1,320))))
zplot_x=np.zeros((len(np.arange(0.0125,0.5126,0.0125)),len(np.linspace(x0,x1,320))))

for iteration, z_val in enumerate(np.arange(0.0125,0.5126,0.0125)):

    #similar to the distance value arrays above, but these are to plot the data
    xplot_x[iteration,:]=np.arange(0.,np.sqrt(np.abs(x1-x0)**2+np.abs(y1-y0)**2),0.0625)
    zplot_x[iteration,:]=z_val

    #interpolates the distances to the indices
    xinterp=interp1d(xsmall,np.arange(len(xsmall)))
    yinterp=interp1d(ysmall,np.arange(len(ysmall)))
    zinterp=interp1d(z,np.arange(len(z)),bounds_error=False,fill_value=0.0)#takes care of the situation where the lowest level is below the lowest grid point so it can't interpolate

    #find the decimal indices where the rhix values exist, these indices is where the image mapping is going to take place
    rhixi = xinterp(np.linspace(x0,x1,320))
    rhiyi = yinterp(np.linspace(y0,y1,320))
    rhizi = zinterp(np.zeros_like(rhiyi)+z_val)
    
    #first input is the full three dimensional data array
    #second input is the indices of the data array where the observations should be
    crosstrp[iteration,:]=scipy.ndimage.map_coordinates(trhop_RHI[:,:,:],[rhizi,rhiyi,rhixi])
    
rawfile = '/home/aschueth/pythonscripts/Schuethetal2021/RHIplane_parcels.txt'
col_names = ['pi', 't', 'h', 'x','y','z','baro','stretch']
df = pd.read_csv(rawfile, usecols=[0,1,2,3,4,5,6,7],names=col_names)

fils='/lustre/research/weiss/schueth/resdep125m/cm1out_pdata.nc'
dsp = xr.open_dataset(fils)
    
wrap_mask=df.x<0

x = df['h'][wrap_mask]
y = df['z'][wrap_mask]
z = df.baro[wrap_mask]
# define grid.
xi = np.arange(11,16.01,0.125)
yi = np.arange(0,0.501,0.025)
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
fig,ax = plt.subplots(figsize=(20, 6),facecolor='white')
CS = plt.contourf(xi-11,yi,zi,np.linspace(-0.0001,0.0001,100),cmap=cmocean.cm.balance,extend='both')
plt.colorbar(ticks=np.linspace(-0.0001,0.0001,6),pad=0.01, format=OOMFormatter(-3, mathText=True)) # draw colorbar
t2=ax.text(0.5, 0.97, r"Baroclinic Tendency [s$^{-2}$]", verticalalignment='top', horizontalalignment='center',transform=ax.transAxes, fontsize=45)
t2.set_path_effects([PathEffects.withStroke(linewidth=6,foreground='white')])
# plot data points.
# plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(5,0)
plt.ylim(0,0.5)
plt.scatter(df['h'][wrap_mask]-11,df['z'][wrap_mask],c='k',s=0.5,alpha=0.1)
subset=np.logical_and(np.logical_and(df.h>11,df.h<16),df.z<0.5)
baro_mask=df.baro>np.percentile(df.baro[subset],99)
print('99th percentile baroclinic:',np.percentile(df.baro[subset],99))
# plt.scatter(df['h'][wrap_mask][baro_mask],df['z'][wrap_mask][baro_mask],c='k',s=3)
plt.scatter(df['h'][wrap_mask][baro_mask]-11,df['z'][wrap_mask][baro_mask],c='k',s=2)
CS=plt.contour(xplot_x[:,176:258]-11,zplot_x[:,176:258],crosstrp[:,176:258],[-3,-2.04,-1,-0.5],colors='0.25')
manual_locations = [(1.5, 0.1), (2.5, 0.18), (4.8, 0.12), (4.4, 0.08)]
plt.clabel(CS, inline=1, fontsize=14.5,fmt='%1.1f',manual=manual_locations)
plt.xlabel('Range [km]')
plt.ylabel('Height [km]')
######################################
patch = matplotlib.patches.Ellipse((1-0.055,0.825),0.0375,0.1,linewidth=1.5,color='0.25',linestyle='dashed',fill=False,transform=ax.transAxes,zorder=1000)
ax.add_patch(patch)
ax.text(0.99,0.98, r"$\theta_\rho$' [K]",c='0.25', verticalalignment='top', horizontalalignment='right',transform=ax.transAxes, fontsize=35)
plt.savefig(f'/home/aschueth/pythonscripts/Schuethetal2021/baroclinic_tendency.png',bbox_inches='tight')


x = df['h'][wrap_mask]
y = df['z'][wrap_mask]
z = df.stretch[wrap_mask]
# define grid.
xi = np.arange(11,16.01,0.125)
yi = np.arange(0,0.501,0.025)
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
fig,ax = plt.subplots(figsize=(20, 6),facecolor='white')
CS = plt.contourf(xi-11,yi,zi,np.linspace(-0.0001,0.0001,100),cmap=cmocean.cm.balance,extend='both')
plt.colorbar(ticks=np.linspace(-0.0001,0.0001,6),pad=0.01, format=OOMFormatter(-3, mathText=True)) # draw colorbar
t2=ax.text(0.5, 0.97, r"3D Stretching Tendency [s$^{-2}$]", verticalalignment='top', horizontalalignment='center',transform=ax.transAxes, fontsize=45)
t2.set_path_effects([PathEffects.withStroke(linewidth=6,foreground='white')])
plt.xlim(5,0)
plt.ylim(0,0.5)
plt.scatter(df['h'][wrap_mask]-11,df['z'][wrap_mask],c='k',s=0.5,alpha=0.1)
subset=np.logical_and(np.logical_and(df.h>11,df.h<16),df.z<0.5)
stretch_mask=df.stretch>np.percentile(df.stretch[subset],99)
print('99th percentile stretching:',np.percentile(df.stretch[subset],99))
plt.scatter(df['h'][wrap_mask][stretch_mask]-11,df['z'][wrap_mask][stretch_mask],c='k',s=2)
CS=plt.contour(xplot_x[:,176:258]-11,zplot_x[:,176:258],crosstrp[:,176:258],[-3,-2.04,-1,-0.5],colors='0.25')
manual_locations = [(1.5, 0.1), (2.5, 0.18), (4.8, 0.12), (4.4, 0.08)]
plt.clabel(CS, inline=1, fontsize=14.5,fmt='%1.1f',manual=manual_locations)
plt.xlabel('Range [km]')
plt.ylabel('Height [km]')
######################################
patch = matplotlib.patches.Ellipse((1-0.055,0.825),0.0375,0.1,linewidth=1.5,color='0.25',linestyle='dashed',fill=False,transform=ax.transAxes,zorder=1000)
ax.add_patch(patch)
ax.text(0.99,0.98, r"$\theta_\rho$' [K]",c='0.25', verticalalignment='top', horizontalalignment='right',transform=ax.transAxes, fontsize=35)
plt.savefig(f'/home/aschueth/pythonscripts/Schuethetal2021/stretching_tendency.png',bbox_inches='tight')



somega = np.zeros_like(np.array(df.t)).astype(float)
for p,t in enumerate(df.t):
    somega[p] = (dsp.u[t,p].values*dsp.xi[t,p].values+dsp.v[t,p].values*dsp.eta[t,p].values+dsp.w[t,p].values*dsp.zeta[t,p].values)/np.sqrt(dsp.u[t,p].values**2+dsp.v[t,p].values**2+dsp.w[t,p].values**2)
x = df['h'][wrap_mask]
y = df['z'][wrap_mask]
z = somega[wrap_mask]
# define grid.
xi = np.arange(11,16.01,0.125)
yi = np.arange(0,0.501,0.025)
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
fig,ax = plt.subplots(figsize=(20, 6),facecolor='white')
CS = plt.contourf(xi-11,yi,zi,np.linspace(-0.1,0.1,100),cmap="PuOr_r",extend='both')
plt.colorbar(ticks=np.linspace(-0.1,0.1,6),pad=0.01) # draw colorbar
t2=ax.text(0.5, 0.97, r"3D Streamwise Vorticity [s$^{-1}$]", verticalalignment='top', horizontalalignment='center',transform=ax.transAxes, fontsize=45)
t2.set_path_effects([PathEffects.withStroke(linewidth=6,foreground='white')])
plt.xlim(5,0)
plt.ylim(0,0.5)
plt.scatter(df['h'][wrap_mask]-11,df['z'][wrap_mask],c='k',s=0.5,alpha=0.1)
subset=np.logical_and(np.logical_and(df.h>11,df.h<16),df.z<0.5)
somega_mask=somega>np.percentile(somega[subset],99)
print('99th percentile svort:',np.percentile(somega[subset],99))
plt.scatter(df['h'][somega_mask]-11,df['z'][somega_mask],c='k',s=2)
CS=plt.contour(xplot_x[:,176:258]-11,zplot_x[:,176:258],crosstrp[:,176:258],[-3,-2.04,-1,-0.5],colors='0.25')
manual_locations = [(1.5, 0.1), (2.5, 0.18), (4.8, 0.12), (4.4, 0.08)]
plt.clabel(CS, inline=1, fontsize=14.5,fmt='%1.1f',manual=manual_locations)
plt.xlabel('Range [km]')
plt.ylabel('Height [km]')
######################################
patch = matplotlib.patches.Ellipse((1-0.055,0.825),0.0375,0.1,linewidth=1.5,color='0.25',linestyle='dashed',fill=False,transform=ax.transAxes,zorder=1000)
ax.add_patch(patch)
ax.text(0.99,0.98, r"$\theta_\rho$' [K]",c='0.25', verticalalignment='top', horizontalalignment='right',transform=ax.transAxes, fontsize=35)
plt.savefig(f'/home/aschueth/pythonscripts/Schuethetal2021/svort_parcels.png',bbox_inches='tight')
