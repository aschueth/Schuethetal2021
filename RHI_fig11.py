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
%matplotlib inline

def RHIvort(vel,phi):
    '''
    Function calculates the vorticity on an RHI assuming solid body rotation. Vorticity vector is also assumed normal to plane.
    Calculation: (1/r)*(du_r/dphi), taken from wolfram mathworld on spherical vorticity calculations. 
    Originally was negative, but due to polar coordiantes being reversed in radar world, it is actually positive.
    
    Parameter: radar (pyart object), swp_id (int)
    Returns: vorticity array, same shape as velocity array
    
    For loop iterates through all radii and all slices, the expanded version would look like this:
        
    #radius = 4
    every40 = slice(0, None, 4)
    every41 = slice(1, None, 4)
    every42 = slice(2, None, 4)
    every43 = slice(3, None, 4)
    
    vort4 = np.zeros_like(vel)
    vort4[every40,:] = (1/rangearray[every40,:])*np.gradient(vel[every40,:],phi[every40],axis=0)
    vort4[every41,:] = (1/rangearray[every41,:])*np.gradient(vel[every41,:],phi[every41],axis=0)
    vort4[every42,:] = (1/rangearray[every42,:])*np.gradient(vel[every42,:],phi[every42],axis=0)
    vort4[every43,:] = (1/rangearray[every43,:])*np.gradient(vel[every43,:],phi[every43],axis=0)

    '''
    rangearray=np.linspace(0,RHIlength*1000,bins)
    vort = np.zeros_like(vel)
    vort=(1/rangearray)*np.gradient(vel,np.deg2rad(phi),axis=0)*2 
        
    return vort

@jit
def variant(array,delta):
    '''
    Function finds local variance at a point. The mean in a floating window of size i-delta to i+delta
    is subtracted from the value at the point i, to calculate the local variance. A larger window will blur the variance more.

    Parameters: array (array), delta (int)
    Returns: variance (array, not the same size I don't think)

    This function is optimized by jit
    '''

    var=np.zeros_like(array)
    d=len(np.shape(array))

    for i in range(np.shape(array)[-1]):
        for j in range(np.shape(array)[-2]):
            if d == 3:
                for k in range(np.shape(array)[-3]):
                    var[k,j,i]=abs(array[k,j,i]-np.nanmean(array[k,j-delta:j+delta,i-delta:i+delta]))
                return(var)
            elif d == 2:
                var[i,j]=abs(array[i,j]-np.nanmean(array[i-delta:i+delta,j-delta:j+delta]))
                return(var)
            
def UHcenter(x,z,w,zvort):
    '''
    Function finds the centroid of the gaussian smoothed 2-5 km UH, if fields are zero, return middle of domain

    Paramters: x (array), z (array), w (array), zvort (array)
    Returns: indices of centroid (y,x)

    Example: x0, y0 = CM1calc.UHcenter(x,z,w,zvort)
    '''

    gausfactor=np.around((1./dx(x))/0.16)
    UH = (w[0,find_nearest(z,2):find_nearest(z,5),:,:]*zvort[0,find_nearest(z,2):find_nearest(z,5),:,:])
    UH=UH.data
    UH = UH.sum(axis=0)
    gauss = dask_image.ndfilters.gaussian_filter(UH,sigma=gausfactor)
    UHfil=gauss>(0.75*np.amax(gauss))

    labelsmaxUH,numlabelsmaxUH=dask_image.ndmeasure.label(UHfil)
    biggest=label_count(labelsmaxUH,numlabelsmaxUH,UHfil)
    mask=np.zeros_like(labelsmaxUH)
    try:
        mask[labelsmaxUH==biggest[0][0]]=1
        centroidsUH=dask_image.ndmeasure.center_of_mass(mask)
    except:
        print('UH not found')
        centroidsUH=(np.shape(UH)[0]/2,np.shape(UH)[0]/2)
    return centroidsUH
    
radar_levels = [x/10. for x in range(-50,651)]
trlevels = [x / 100. for x in range(-900,1)]
ppertlevels = [x / 10. for x in range(-100,101)]

startTime = datetime.now()

x0,y0,z0 = 7., 0., 0. #kilometers from the origin, aka where the radar is located
z0=0.
RHIlength=13. #length of the RHI
bins=1000 #This is the number of points along the line, "bins" if you will, bin length 10-15 m in the Ka
zenithres=0.3 #angle resolution in degrees, this is the correct resolution of the Ka
zenithmax=20.1 #max angle of the RHI
        
azimuth=175

x1=x0+(RHIlength*np.cos(np.deg2rad(azimuth)))
y1=y0+(RHIlength*np.sin(np.deg2rad(azimuth)))

xbeg=min([x0,x1])
xend=max([x0,x1])
ybeg=min([y0,y1])
yend=max([y0,y1])

filr=682
filename = '/lustre/research/weiss/schueth/resdep125m/cm1out_000'+str(filr)+'.nc'
ds = xr.open_dataset(filename,chunks={'ni':288,'nj':288})
time = ds['time'].values
x = ds['xh']
y = ds['yh']
z = ds['z']
w = ds['winterp']
zvort = ds['zvort']

centroidsUH = UHcenter(x,z,w,zvort).compute()

x=x-x[int(centroidsUH[1])]
y=y-y[int(centroidsUH[0])]
X,Y=np.meshgrid(x.values,y.values)
dxx=dx(x)

xsmall = x.isel(ni=slice(find_nearest(x,xbeg)-2,find_nearest(x,xend)+2))
ysmall = y.isel(nj=slice(find_nearest(y,ybeg)-2,find_nearest(y,yend)+2))

xvort =ds['xvort'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
yvort =ds['yvort'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]

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

xvortproj=-np.sin(np.deg2rad(azimuth))*xvort
yvortproj=np.cos(np.deg2rad(azimuth))*yvort

zenith = np.arange(zenithmax/zenithres+1)*zenithres
xplot = np.zeros((len(zenith),int(bins)))
zplot = np.zeros((len(zenith),int(bins)))
rhitrp=np.zeros((len(zenith),int(bins)))
rhivort=np.zeros((len(zenith),int(bins)))

for iteration, zen in enumerate(zenith):

    #the distance value of the end of the RHI
    x1=x0+(RHIlength*np.cos(np.deg2rad(azimuth))*np.cos(np.deg2rad(zen)))
    y1=y0+(RHIlength*np.sin(np.deg2rad(azimuth))*np.cos(np.deg2rad(zen)))
    z1=z0+(RHIlength*np.sin(np.deg2rad(zen)))

    #similar to the distance value arrays above, but these are to plot the data
    xplot[iteration,:]=np.linspace(0.,np.sqrt(np.abs(x1-x0)**2+np.abs(y1-y0)**2),bins)
    zplot[iteration,:]=np.linspace(z0,z1,bins)

    #interpolates the distances to the indices
    xinterp=interp1d(xsmall,np.arange(len(xsmall)))
    yinterp=interp1d(ysmall,np.arange(len(ysmall)))
    zinterp=interp1d(z,np.arange(len(z)),bounds_error=False,fill_value=0.0)#takes care of the situation where the lowest level is below the lowest grid point so it can't interpolate

    #find the decimal indices where the rhix values exist, these indices is where the image mapping is going to take place
    rhixi = xinterp(np.linspace(x0,x1,bins))
    rhiyi = yinterp(np.linspace(y0,y1,bins))
    rhizi = zinterp(np.linspace(z0,z1,bins))

    #first input is the full three dimensional data array
    #second input is the indices of the data array where the observations should be
    rhitrp[iteration,:]= map_coordinates(trhop_RHI[:,:,:], [rhizi,rhiyi,rhixi])
    rhivort[iteration,:]= map_coordinates(xvortproj+yvortproj, [rhizi,rhiyi,rhixi])
    
    
cc=0
fig = plt.figure(figsize=(15, 7),facecolor='white')
ax2 = plt.subplot2grid((1,1),(0,0))
ptr = ax2.pcolormesh(xplot,zplot,rhitrp,cmap=cmocean.cm.gray,vmin=-9,vmax=0)
ax2.contour(xplot,zplot,rhivort,[0.02,0.03,0.04,0.05,0.06],colors='k',linewidths=0.5)
ax2.contour(xplot,zplot,rhivort,[-0.02],colors='k',linewidths=0.5,linestyle=(0,(5,20)))

cbar = plt.colorbar(ptr)
cbar.ax.tick_params(labelsize=25)
t2=ax2.text(0.5, 0.97, r"$\theta'_\rho$ [K]", verticalalignment='top', horizontalalignment='center',transform=ax2.transAxes, fontsize=40)
t2.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
t3=ax2.text(0.99, 0.97, f"Time: {int(time[0]/1e9)} s", verticalalignment='top', horizontalalignment='right',transform=ax2.transAxes, fontsize=25)
t3.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])

plt.ylim(0, 4.5)
plt.xlim(0,13)
plt.ylabel('Kilometers', fontsize=30)
plt.xticks(np.arange(0., 14., 1.0))
plt.yticks(np.arange(0., 5., 1.0))
plt.grid('on')
plt.tick_params(axis='both', which='major', labelsize=25)

patch = matplotlib.patches.Ellipse((0.14,0.77), 0.1,0.2,color='w',fill=True,transform=fig.transFigure)
patch0 = matplotlib.patches.Ellipse((0.14,0.77), 0.1,0.2,linestyle='solid',linewidth=0.5,color='k',fill=False,transform=fig.transFigure)
patch1 = matplotlib.patches.Ellipse((0.1475,0.77), 0.08,0.16,linestyle='solid',linewidth=0.5,color='k',fill=False,transform=fig.transFigure)
patch2 = matplotlib.patches.Ellipse((0.155,0.77), 0.06,0.12,linestyle='solid',linewidth=0.5,color='k',fill=False,transform=fig.transFigure)
patch3 = matplotlib.patches.Ellipse((0.1625,0.77), 0.04,0.08,linestyle='solid',linewidth=0.5,color='k',fill=False,transform=fig.transFigure)
patch4 = matplotlib.patches.Ellipse((0.17,0.77), 0.02,0.04,linestyle='solid',linewidth=0.5,color='k',fill=False,transform=fig.transFigure)

ax = fig.add_axes([0,0,1,1],facecolor=None)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_zorder(1000)

ax.add_patch(patch)
ax.add_patch(patch0)
ax.add_patch(patch1)
ax.add_patch(patch2)
ax.add_patch(patch3)
ax.add_patch(patch4)

patchm1 = matplotlib.patches.Ellipse((0.14,0.64), 0.02,0.04,linestyle='--',linewidth=0.5,color='k',fill=False,transform=fig.transFigure)

ax.add_patch(patchm1)

ax.patch.set_alpha(0)
t4=ax.text(0.15, 0.62, "-0.02", verticalalignment='top', horizontalalignment='right',transform=fig.transFigure, fontsize=10,zorder=1001)
t4.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='white')])
t4=ax.text(0.1, 0.76, "0.02", verticalalignment='top', horizontalalignment='right',transform=fig.transFigure, fontsize=10,zorder=1001)
t4.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='white')])
t4=ax.text(0.12, 0.79, "0.03", verticalalignment='top', horizontalalignment='right',transform=fig.transFigure, fontsize=10,zorder=1001)
t4.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='white')])
t4=ax.text(0.14, 0.76, "0.04", verticalalignment='top', horizontalalignment='right',transform=fig.transFigure, fontsize=10,zorder=1001)
t4.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='white')])
t4=ax.text(0.155, 0.79, "0.05", verticalalignment='top', horizontalalignment='right',transform=fig.transFigure, fontsize=10,zorder=1001)
t4.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='white')])
t4=ax.text(0.175, 0.76, "0.06", verticalalignment='top', horizontalalignment='right',transform=fig.transFigure, fontsize=10,zorder=1001)
t4.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='white')])
t4=ax.text(0.07, 0.93, r"Vorticity [s$^{-1}$]", verticalalignment='top', horizontalalignment='left',transform=fig.transFigure, fontsize=20,zorder=1001)
t4.set_path_effects([PathEffects.withStroke(linewidth=5,foreground='white')])
ax.axis("off")

plt.subplots_adjust(wspace=0.01,hspace=0.01)
plt.tight_layout()

plt.savefig(f'/home/aschueth/pythonscripts/Schuethetal2021/RHI_fig11.png',bbox_inches='tight')
