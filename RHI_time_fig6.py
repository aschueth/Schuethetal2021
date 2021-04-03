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
    vort=(1/rangearray)*np.gradient(vel,np.deg2rad(phi),axis=0) 
        
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

filename = '/lustre/research/weiss/schueth/resdep125m/cm1out_rst_000001.nc'
print(filename)
ds = xr.open_dataset(filename,chunks={'ni':288,'nj':288})
p_base = (ds["prs"][0,:,:,:]/100)
        
azimuth=170

x1=x0+(RHIlength*np.cos(np.deg2rad(azimuth)))
y1=y0+(RHIlength*np.sin(np.deg2rad(azimuth)))

xbeg=min([x0,x1])
xend=max([x0,x1])
ybeg=min([y0,y1])
yend=max([y0,y1])


# a script runs this script in a batch, regex replaces filr with a list of the different file numbers [521,561,601,682]
if filr == 521:
    letters=['a)','b)','c)','d)','e)']
elif filr == 561:
    letters=['f)','g)','h)','i)','j)']
elif filr == 601:
    letters=['k)','l)','m)','n)','o)']
elif filr == 682:
    letters=['p)','q)','r)','s)','t)']
else:
    letters=['','','','','']

print('Beginning to read data in')
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

ref =ds['dbz'].isel(time=0,nk=0)
u = ds['uinterp'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]+10.5
v = ds['vinterp'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]+8.7
w = ds['winterp'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
xvort =ds['xvort'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
yvort =ds['yvort'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
zvort =ds['zvort'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
ppert = ds['prspert'].isel(time=0)[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]/100.
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

th_plan = ds['th'].isel(time=0,nk=0)
qr_plan = ds['qr'].isel(time=0,nk=0)
qi_plan = ds['qi'].isel(time=0,nk=0)
qc_plan = ds['qc'].isel(time=0,nk=0)
qs_plan = ds['qs'].isel(time=0,nk=0)
qg_plan = ds['qg'].isel(time=0,nk=0)
qv_plan = ds['qv'].isel(time=0,nk=0)

trhop_plan=perturbation(trho(th_plan,qv_plan,qi_plan,qc_plan,qs_plan,qr_plan,qg_plan),trho(th_corner[0],qv_corner[0],qi_corner[0],qc_corner[0],qs_corner[0],qr_corner[0],qg_corner[0]))

p0=p_base[:,find_nearest(y,ybeg)-2:find_nearest(y,yend)+2,find_nearest(x,xbeg)-2:find_nearest(x,xend)+2]
p=p0+ppert
dz = np.repeat(np.repeat(np.gradient(z*1000)[:, np.newaxis], np.shape(ppert)[1], axis=1)[:,:, np.newaxis], np.shape(ppert)[2], axis=2)
thetav=th_RHI*(1.+0.61*qv_RHI)
dthetavdz = np.gradient(thetav,axis=0)/dz

dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz = vector_gradient_components(u,v,w,dxx*1000,np.gradient(z)*1000)
divergence = dudx+dvdy

t=th_RHI*((p/1000)**mpconsts.kappa.m)

tv=(t)*(1.+0.61*qv_RHI)
rich = ((9.81/tv)*dthetavdz)/(dudz**2+dvdz**2)

wind = np.sqrt(u**2+v**2+w**2)
svort=(u*xvort+v*yvort+w*zvort)/wind

xvortproj=-np.sin(np.deg2rad(azimuth))*xvort
yvortproj=np.cos(np.deg2rad(azimuth))*yvort

#takes care of the special cases of geometry
if azimuth >=0. and azimuth < 90.:
    uproj=u*(1.-np.sin(np.deg2rad(azimuth)))
    vproj=v*(1.-np.cos(np.deg2rad(azimuth)))

if azimuth >= 90. and azimuth < 180.:
    uproj=-u*(1.-np.sin(np.deg2rad(azimuth)))
    vproj=v*(1.-np.cos(np.deg2rad(180-azimuth)))

if azimuth >= 180. and azimuth < 270.:
    uproj=-u*(1.-np.sin(np.deg2rad(360-azimuth)))
    vproj=-v*(1.-np.cos(np.deg2rad(180-azimuth)))

if azimuth >= 270. and azimuth <360.:
    uproj=u*(1.-np.sin(np.deg2rad(360-azimuth)))
    vproj=-v*(1.-np.cos(np.deg2rad(azimuth)))

zenith = np.arange(zenithmax/zenithres+1)*zenithres
xplot = np.zeros((len(zenith),int(bins)))
zplot = np.zeros((len(zenith),int(bins)))
rhiwind=np.zeros((len(zenith),int(bins)))
rhiu=np.zeros((len(zenith),int(bins)))
rhiv=np.zeros((len(zenith),int(bins)))
rhiw=np.zeros((len(zenith),int(bins)))
rhitrp=np.zeros((len(zenith),int(bins)))
#     rhivortnorm=np.zeros((len(zenith),int(bins)))
rhippert=np.zeros((len(zenith),int(bins)))
rhirich=np.zeros((len(zenith),int(bins)))
rhidiv=np.zeros((len(zenith),int(bins)))

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
    rhiu[iteration,:]= map_coordinates(uproj[:,:,:], [rhizi,rhiyi,rhixi])
    rhiv[iteration,:]= map_coordinates(vproj[:,:,:], [rhizi,rhiyi,rhixi])
    rhiw[iteration,:]= map_coordinates(w[:,:,:], [rhizi,rhiyi,rhixi])
    rhiwind[iteration,:]= map_coordinates(uproj*np.cos(np.deg2rad(zen))+vproj*np.cos(np.deg2rad(zen))+w*np.sin(np.deg2rad(zen)), [rhizi,rhiyi,rhixi]) #adds the projection of the u component on the desired plane and the projection of the v component on the desired plane together
    rhitrp[iteration,:]= map_coordinates(trhop_RHI[:,:,:], [rhizi,rhiyi,rhixi])
#         rhivortnorm[iteration,:]= map_coordinates(xvortproj+yvortproj, [rhizi,rhiyi,rhixi])
    rhippert[iteration,:]= map_coordinates(ppert[:,:,:], [rhizi, rhiyi, rhixi])
    rhirich[iteration,:]= map_coordinates(rich[:,:,:], [rhizi, rhiyi, rhixi])
    rhidiv[iteration,:]= map_coordinates(divergence[:,:,:], [rhizi, rhiyi, rhixi])


#the distance value of the end of the RHI
x1=x0+(RHIlength*np.cos(np.deg2rad(azimuth)))
y1=y0+(RHIlength*np.sin(np.deg2rad(azimuth)))

crossuv=np.zeros((len(np.arange(0.0125,5.0126,0.025)),len(np.linspace(x0,x1,104))))
crossw=np.zeros((len(np.arange(0.0125,5.0126,0.025)),len(np.linspace(x0,x1,104))))

xplot_x=np.zeros((len(np.arange(0.0125,5.0126,0.025)),len(np.linspace(x0,x1,104))))
zplot_x=np.zeros((len(np.arange(0.0125,5.0126,0.025)),len(np.linspace(x0,x1,104))))

for iteration, z_val in enumerate(np.arange(0.0125,5.0126,0.025)):

    #similar to the distance value arrays above, but these are to plot the data
    xplot_x[iteration,:]=np.arange(0.,np.sqrt(np.abs(x1-x0)**2+np.abs(y1-y0)**2),0.125)
    zplot_x[iteration,:]=z_val

    #interpolates the distances to the indices
    xinterp=interp1d(xsmall,np.arange(len(xsmall)))
    yinterp=interp1d(ysmall,np.arange(len(ysmall)))
    zinterp=interp1d(z,np.arange(len(z)),bounds_error=False,fill_value=0.0)#takes care of the situation where the lowest level is below the lowest grid point so it can't interpolate

    #find the decimal indices where the rhix values exist, these indices is where the image mapping is going to take place
    rhixi = xinterp(np.linspace(x0,x1,104))
    rhiyi = yinterp(np.linspace(y0,y1,104))
    rhizi = zinterp(np.zeros_like(rhiyi)+z_val)
    
    #first input is the full three dimensional data array
    #second input is the indices of the data array where the observations should be
    crossuv[iteration,:]=scipy.ndimage.map_coordinates(uproj[:,:,:]+vproj[:,:,:],[rhizi,rhiyi,rhixi])
    crossw[iteration,:]=scipy.ndimage.map_coordinates(w[:,:,:],[rhizi,rhiyi,rhixi])
    
cc=0
fig = plt.figure(figsize=(45, 13),facecolor='white')

#row,column
ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2)
plan=plt.contourf(X,Y,trhop_plan,trlevels,cmap=cmocean.cm.rain_r,extend='both')
plt.contour(X, Y, ref, [40], colors='black',linewidths=3,linestyles='dashed')
plt.plot([x0,x0+(RHIlength*np.cos(np.deg2rad(azimuth))*np.cos(np.deg2rad(zen)))],[y0,y0+(RHIlength*np.sin(np.deg2rad(azimuth))*np.cos(np.deg2rad(zen)))],color="black",linewidth=4)
plt.plot(x0,y0,'ko', markersize=18)
plt.plot(x0,y0,'wo', markersize=12)
cbar = plt.colorbar(plan, ticks=[-9,-8,-7,-6,-5,-4,-3,-2,-1,0])
cbar.ax.tick_params(labelsize=35)
plt.xlabel('X [km]', fontsize=35)
plt.ylabel('Y [km]', fontsize=35)
plt.xticks(np.arange(-15,26,5))
plt.yticks(np.arange(-15,26,5))
plt.tick_params(axis='both', which='major', labelsize=30)
t1=ax1.text(0.5, 0.99, r"$\theta'_\rho$ [K]", verticalalignment='top', horizontalalignment='center',transform=ax1.transAxes, fontsize=70)
t1.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
t2=ax1.text(0.025, 0.975, letters[cc], verticalalignment='top', horizontalalignment='left',transform=ax1.transAxes, fontsize=75)
t2.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
cc+=1
plt.xlim(-15,25)
plt.ylim(-15,25)
plt.title('Time= '+str(int(time[0]/np.timedelta64(1, 's')))+' s   '+'Az= '+str(int(360-azimuth+90))+r'$^\circ$',fontsize=50)

ax2 = plt.subplot2grid((2,3),(0,1))
ptr = ax2.pcolormesh(xplot,zplot,rhitrp,cmap=cmocean.cm.rain_r,vmin=-9,vmax=0)#cmap=sftemp(),
cbar = plt.colorbar(ptr,extend='both')
cbar.ax.tick_params(labelsize=35)
t2=ax2.text(0.5, 0.97, r"$\theta'_\rho$ [K]", verticalalignment='top', horizontalalignment='center',transform=ax2.transAxes, fontsize=55)
t2.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
plt.ylim(0, 2.5)
plt.xlim(13, 0)
plt.ylabel('Height [km]', fontsize=35)
plt.xticks(np.arange(0., 14., 1.0),[])
plt.yticks(np.arange(0., 2.5, 0.5))
plt.grid('on')
plt.tick_params(axis='both', which='major', labelsize=30)
t2=ax2.text(0.025, 0.975, letters[cc], verticalalignment='top', horizontalalignment='left',transform=ax2.transAxes, fontsize=75)
t2.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
skip= (slice(None, None, 10), slice(None, None, 5))
Q=ax2.quiver(xplot_x[skip],zplot_x[skip],-crossuv[skip],crossw[skip],color="k")
qk = ax1.quiverkey(Q, 0.56, 0.955, 20, r'20 m s$^{-1}$', labelpos='E', coordinates='figure',fontproperties={'size':35})
cc+=1

ax3 = plt.subplot2grid((2,3),(0,2))
pppert=ax3.pcolormesh(xplot,zplot,rhippert,cmap=cmocean.cm.thermal,vmin=-3.,vmax=0.)
cbar3 = plt.colorbar(pppert, ticks=[-3.,-2.5,-2.,-1.5,-1.,-0.5,0],extend='both')
cbar3.ax.tick_params(labelsize=35)
plt.ylim(0, 2.5)
plt.xlim(13, 0)
plt.xticks(np.arange(0., 14., 1.0),[])
plt.yticks(np.arange(0., 2.5, 0.5),[])
plt.grid('on')
plt.tick_params(axis='both', which='major', labelsize=30)
t3=ax3.text(0.5, 0.97, r"$P'$ [hPa]", verticalalignment='top', horizontalalignment='center',transform=ax3.transAxes, fontsize=55)
t3.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
t2=ax3.text(0.025, 0.975, letters[cc], verticalalignment='top', horizontalalignment='left',transform=ax3.transAxes, fontsize=75)
t2.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
cc+=1

ax4 = plt.subplot2grid((2,3),(1,1))
pvel=ax4.pcolormesh(xplot,zplot,rhiwind,cmap=cmocean.cm.balance, vmin=-30, vmax=30)
cbar = plt.colorbar(pvel, ticks=[-30,-20,-10,0,10,20,30],extend='both')
cbar.ax.tick_params(labelsize=35)
plt.contour(xplot,zplot,rhidiv,[-0.01],colors='black',linewidths=1,linestyles='solid')
t4=ax4.text(0.5, 0.97, r'V$_r$ [m s$^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax4.transAxes, fontsize=55)
t4.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
plt.ylim(0, 2.5)
plt.xlim(13, 0)
plt.ylabel('Height [km]', fontsize=35)
plt.xlabel('Range [km]', fontsize=30)
plt.xticks(np.arange(0., 14., 1.0))
plt.yticks(np.arange(0., 2.5, 0.5))
plt.grid('on')
plt.tick_params(axis='both', which='major', labelsize=30)
t2=ax4.text(0.025, 0.975, letters[cc], verticalalignment='top', horizontalalignment='left',transform=ax4.transAxes, fontsize=75)
t2.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
##################################
patch = matplotlib.patches.Ellipse((0.875,0.775),0.075,0.15,linewidth=1,color='k',fill=False,transform=ax4.transAxes,zorder=1000)
ax4.add_patch(patch)
t=ax4.text(0.875, 0.775, "-0.01", verticalalignment='center', horizontalalignment='left',transform=ax4.transAxes, fontsize=20,zorder=1001)
t.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])
ax4.text(0.98,0.98, r'Div [s$^{-1}$]', verticalalignment='top', horizontalalignment='right',transform=ax4.transAxes, fontsize=40)
######################################
cc+=1

rhivortnorm = RHIvort(rhiwind,zenith)
ax5 = plt.subplot2grid((2,3),(1,2))
psvort=ax5.pcolormesh(xplot,zplot,rhivortnorm,cmap="PuOr_r",vmin=-0.075,vmax=0.075)
cbar5 = plt.colorbar(psvort, ticks=[-0.075,-0.05,-0.025,0,0.025,0.05,0.075],extend='both')
cbar5.ax.tick_params(labelsize=35)
#################### Some rudimentary morphological processing to remove noise
labels,numlabels=label((rhirich<0.25),return_num=True)
mask=np.zeros_like(rhirich)
mask[labels==labels[0,0]]=1
mask=erosion(mask,disk(1))#4 is too high
mask=dilation(mask,disk(1))
labels,numlabels=label(mask,return_num=True)
mask[labels!=labels[0,0]]=0
##################################
plt.contour(xplot,zplot,mask,[1],colors='black',linewidths=2)
t5=ax5.text(0.5, 0.97, r'$\nabla \times \mathbf{V}$ [s$^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax5.transAxes, fontsize=55)
t5.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
plt.ylim(0, 2.5)
plt.xlim(13, 0)
plt.xlabel('Range [km]', fontsize=35)
plt.xticks(np.arange(0., 14., 1.0))
plt.yticks(np.arange(0., 2.5, 0.5),[])
plt.grid('on')
plt.tick_params(axis='both', which='major', labelsize=30)
t2=ax5.text(0.025, 0.975, letters[cc], verticalalignment='top', horizontalalignment='left',transform=ax5.transAxes, fontsize=75)
t2.set_path_effects([PathEffects.withStroke(linewidth=12,foreground='white')])
##################################
patch = matplotlib.patches.Ellipse((0.94,0.8),0.075,0.15,linewidth=2,color='k',fill=False,transform=ax5.transAxes,zorder=1000)
ax5.add_patch(patch)
t=ax5.text(0.94, 0.8, "0.25", verticalalignment='center', horizontalalignment='left',transform=ax5.transAxes, fontsize=20,zorder=1001)
t.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])
ax5.text(0.965,0.98, r'Ri', verticalalignment='top', horizontalalignment='right',transform=ax5.transAxes, fontsize=40)
######################################

plt.tight_layout()#fixes the aspect ratio some

new_cbar_bounds = (cbar3.ax.get_position(original=True).bounds[0]-0.03,cbar3.ax.get_position(original=True).bounds[1],cbar3.ax.get_position(original=True).bounds[2],cbar3.ax.get_position(original=True).bounds[3])
cbar3.ax.set_position(new_cbar_bounds)
new_bounds = (ax3.get_position(original=True).bounds[0]-0.03,ax3.get_position(original=True).bounds[1],ax3.get_position(original=True).bounds[2],ax3.get_position(original=True).bounds[3])
ax3.set_position(new_bounds)

new_cbar_bounds = (cbar5.ax.get_position(original=True).bounds[0]-0.03,cbar5.ax.get_position(original=True).bounds[1],cbar5.ax.get_position(original=True).bounds[2],cbar5.ax.get_position(original=True).bounds[3])
cbar5.ax.set_position(new_cbar_bounds)
new_bounds = (ax5.get_position(original=True).bounds[0]-0.03,ax5.get_position(original=True).bounds[1],ax5.get_position(original=True).bounds[2],ax5.get_position(original=True).bounds[3])
ax5.set_position(new_bounds)

plt.savefig(f'/home/aschueth/pythonscripts/Schuethetal2021/{filr}_{azimuth}.pdf',dpi=300)
plt.close()
fig.clf()
gc.collect()
