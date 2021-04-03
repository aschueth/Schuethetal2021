import matplotlib
matplotlib.use('agg')
import glob
import numpy as np
import pyart
import matplotlib.pyplot as plt
import netCDF4
import os
import cartopy.io.shapereader as shpreader
from matplotlib.colors import Normalize
from metpy.plots import ctables
import matplotlib.colors as colors
import metpy
from scipy import ndimage
import cmocean
import pandas as pd
from datetime import datetime
from datetime import timedelta
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature,NaturalEarthFeature
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from cartopy.io.shapereader import Reader
import matplotlib.patheffects as PathEffects
from astropy.convolution import convolve
from boto.s3.connection import S3Connection
import tempfile
import osmnx as ox
import shutil

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3, fontsize=12, text_pad=5, inner_spacing=2):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    text pad controls the spacing between the text (km, distance1, distance2) and the scale bar.
        squished numbers are bad!
 
    edited by Jessie McDonald. Found on GitHub - https://github.com/SciTools/cartopy/issues/490
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]
 
    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 
 
    #Generate the x coordinate for the ends of the scalebar
    # the edge is the black lines in between and extending on the edges of the white and gray parts
    # sbx is the center. length * 500 == half the length
 
    #edge = 500
    edge = inner_spacing*100
    bar_xs = [sbx - length * 500 - edge, sbx + length * 500 +edge]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth+4,solid_capstyle="butt")  # background
 
    #add colors - 4 colors
    x1, x2, x3, x4, x5 = sbx-length*500, sbx -length*250, sbx, sbx+length*250, sbx+length*500
    ax.plot([x1,x2-edge/2], [sby, sby], transform=tmc, color='w', linewidth=linewidth,solid_capstyle="butt")
    ax.plot([x2+edge/2,x3-edge/2] ,[sby, sby], transform=tmc, color='gray', linewidth=linewidth,solid_capstyle="butt")
    ax.plot([x3+edge/2,x4-edge/2] ,[sby, sby], transform=tmc, color='w', linewidth=linewidth,solid_capstyle="butt")
    ax.plot([x4+edge/2,x5] ,[sby, sby], transform=tmc, color='gray', linewidth=linewidth,solid_capstyle="butt")
 
    # add "km", center point, and end point
    y_lift = linewidth*text_pad
    ax.text(x3, sby+length*y_lift , str(round(2 * length / 4)), transform=tmc,
                    horizontalalignment='center', verticalalignment='bottom',
                    color='k', fontsize=fontsize)
    ax.text(x5-length*10, sby+length*y_lift , str(round(length)), transform=tmc,
            horizontalalignment='center', verticalalignment='bottom',
            color='k', fontsize=fontsize)
    ax.text(x1+length*150, sby+length*y_lift , 'km', transform=tmc,
            horizontalalignment='right', verticalalignment='bottom',
            color='k', fontsize=fontsize)
    
def aliasfix(array,delta,nyq):
    '''
    Half-assed attempt to fix de-aliasing errors. Calculates variance
    (difference between point and mean) over a floating window in the domain.
    If the variance is greater than 35 (arbitrary threshold), then it is a de-aliasing error.
    The if statements fix this according to the nyquist velocity.
    '''

    mean = convolve(array,np.ones((delta,delta))/delta**2.)
    maskpos = np.logical_and(np.abs(array-mean)>nyq*1.5,array>0)
    maskneg = np.logical_and(np.abs(array-mean)>nyq*1.5,array<0)

    array[maskpos]= -2.*nyq+array[maskpos]
    array[maskneg]= 2.*nyq+array[maskneg]

    return array

def createCircleAroundWithRadius(lat, lon, radiuskm,sectorstart,sectorfinish,heading):
    #    ring = ogr.Geometry(ogr.wkbLinearRing)
    latArray = []
    lonArray = []
    for brng in range(int(heading+sectorstart),int(heading+sectorfinish)): #degrees of sector

        lat2, lon2 = getLocation(lat,lon,brng,radiuskm)
        latArray.append(lat2)
        lonArray.append(lon2)
    return lonArray,latArray

def getLocation(lat1, lon1, brng, distancekm):
    lat1 = lat1 * np.pi / 180.0
    lon1 = lon1 * np.pi / 180.0
    #earth radius
    R = 6378.1
    #R = ~ 3959 MilesR = 3959
    bearing = (brng / 90.)* np.pi / 2.

    lat2 = np.arcsin(np.sin(lat1) * np.cos(distancekm/R) + np.cos(lat1) * np.sin(distancekm/R) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(np.sin(bearing)*np.sin(distancekm/R)*np.cos(lat1),np.cos(distancekm/R)-np.sin(lat1)*np.sin(lat2))
    lon2 = 180.0 * lon2 / np.pi
    lat2 = 180.0 * lat2 / np.pi
    return lat2, lon2

def getDistance(lat1,lon1,lat2,lon2):
    R = 6378.1

    dlon = np.deg2rad(lon2) - np.deg2rad(lon1)
    dlat = np.deg2rad(lat2) - np.deg2rad(lat1)

    a = np.sin(dlat / 2)**2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def RHIvort(radar,swp_id):
    phi=np.deg2rad(radar.elevation['data'])
    rangearray = radar.range['data']
    vel=radar.fields['vel_fix']['data']
    vort = np.zeros_like(vel)
    vort=(1/rangearray)*np.gradient(vel,phi,axis=0)
    
    gatex, gatey, gatez = radar.get_gate_x_y_z(0)
    dist = np.sqrt(gatex**2+gatey**2)
    height = gatez
    
    return vort, dist, height

Z = [[0,0],[0,0]]
dbzlevels = [x for x in np.linspace(-30, 30.01, 100)]
CS3 = plt.contourf(Z, dbzlevels, cmap='pyart_HomeyerRainbow',extend="both")
plt.clf()

Z = [[0,0],[0,0]]
vellevels = [x for x in np.linspace(-40, 40.01, 100)]
CS4 = plt.contourf(Z, vellevels, cmap=cmocean.cm.balance,extend="both")
plt.clf()

day='20180608'

ka1_file_PPI = '/home/aschueth/pythonscripts/Schuethetal2021/Ka1180608231108.RAWBYY9'
ka2_file_RHIs = ['/home/aschueth/pythonscripts/Schuethetal2021/Ka2180608230633.nc','/home/aschueth/pythonscripts/Schuethetal2021/Ka2180608230812.nc','/home/aschueth/pythonscripts/Schuethetal2021/Ka2180608231129.nc','/home/aschueth/pythonscripts/Schuethetal2021/Ka2180608231526.nc','/home/aschueth/pythonscripts/Schuethetal2021/Ka2180608231955.nc']

###################################### PPI PLOT
radar_PPI = pyart.io.read(ka1_file_PPI)
scale=0.75
n_swps = radar_PPI.nsweeps

print('dealiasing')
#create new variable for dealiasing velocities

dealias_data = pyart.correct.region_dealias.dealias_region_based(radar_PPI)
nyq=radar_PPI.get_nyquist_vel(0)

radar_PPI.add_field('corrected_velocity', dealias_data, replace_existing=True)
velfix=np.array([[]])
for swp_id in range(n_swps):
    if swp_id == 0:
        velfix=aliasfix(radar_PPI.get_field(swp_id,'corrected_velocity'),13,nyq)
    else:
        velfix=np.append(velfix,aliasfix(radar_PPI.get_field(swp_id,'corrected_velocity'),13,nyq),axis=0)

velfixdict=np.copy(dealias_data).tolist()
velfixdict['data']=velfix
radar_PPI.add_field('corrected_velocity', velfixdict, replace_existing=True)

radar_name = radar_PPI.metadata['instrument_name']

swp_id=1
plotter = pyart.graph.RadarDisplay(radar_PPI)
azimuth = radar_PPI.fixed_angle['data'][swp_id]

#creating the mask for attenuation
reflectivity = radar_PPI.fields['reflectivity']['data']
spectrum_width = radar_PPI.fields['spectrum_width']['data']
velocity = radar_PPI.fields['corrected_velocity']['data']
total_power = radar_PPI.fields['total_power']['data']
normal = radar_PPI.fields['normalized_coherent_power']['data']
normal_mask = (normal.flatten() < 0.4)
range_mask=np.zeros(np.shape(reflectivity))
for i in range(0,len(range_mask[:,0])):
    range_mask[i,:]=radar_PPI.range['data']>(13400)

range_mask=range_mask.astype(bool)
total_mask = [any(t) for t in zip(range_mask.flatten(), normal_mask.flatten())]

refl_mask = np.ma.MaskedArray(reflectivity, mask=total_mask)
sw_mask = np.ma.MaskedArray(spectrum_width, mask=total_mask)
vel_mask = np.ma.MaskedArray(velocity, mask=total_mask)

#create the dictionary for the masks
refl_dict = {'data':refl_mask}
sw_dict = {'data':sw_mask}
vel_dict = {'data':vel_mask}
radar_PPI.add_field('refl_fix',refl_dict)
radar_PPI.add_field('sw_fix',sw_dict)
radar_PPI.add_field('vel_fix',vel_dict)

ka1dep=pd.read_csv('/home/aschueth/pythonscripts/Schuethetal2021/20180608_deployments_ka1.csv')
ka2dep=pd.read_csv('/home/aschueth/pythonscripts/Schuethetal2021/20180612_deployments_ka2.csv')
currentscantime=datetime.strptime(radar_PPI.time['units'][14:-1], "%Y-%m-%dT%H:%M:%S")
for t in range(ka1dep.time_begin.count()):
    try:
        beginscan1=datetime.strptime(ka1dep.time_begin[t], "%m/%d/%Y %H:%M")
        endscan1=datetime.strptime(ka1dep.time_end[t], "%m/%d/%Y %H:%M")
    except: pass

    if currentscantime >= beginscan1 and currentscantime <= endscan1:
        klat1=ka1dep.lat[t]
        klon1=ka1dep.lon[t]
        klat1=ka1dep.lat[t]
        klon1=ka1dep.lon[t]
        head1=ka1dep.heading[t]
        rhib1=ka1dep.rhib[t]
        rhie1=ka1dep.rhie[t]
        klat = klat1
        klon = klon1
        head = head1
for t in range(ka2dep.time_begin.count()):
    try:
        beginscan2=datetime.strptime(ka2dep.time_begin[t], "%m/%d/%Y %H:%M")
        endscan2=datetime.strptime(ka2dep.time_end[t], "%m/%d/%Y %H:%M")
    except: pass

    if currentscantime >= beginscan2 and currentscantime <= endscan2:
        klat2=ka2dep.lat[t]
        klon2=ka2dep.lon[t]

        
radar_PPI.azimuth['data']=radar_PPI.azimuth['data']+head
radar_PPI.latitude['data'] = np.array([klat])
radar_PPI.longitude['data'] = np.array([klon])
       
        
SMALL_SIZE = 35
MEDIUM_SIZE = 50
BIGGER_SIZE = 75

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure(figsize=(55, 55),facecolor='white')
display = pyart.graph.RadarMapDisplay(radar_PPI)

minlat = getLocation(klat,klon,180,14)[0]
maxlat = getLocation(klat,klon,0,14)[0]
minlon = getLocation(klat,klon,270,14)[1]
maxlon = getLocation(klat,klon,90,14)[1]

ax1 = plt.subplot2grid((8,4),(0,0), colspan=2, rowspan=3, projection=display.grid_projection)
ax1.plot(klon,klat,color='r',markersize=200,transform=display.grid_projection)
display.plot_ppi_map('refl_fix', swp_id, cmap='pyart_HomeyerRainbow', ax=ax1, vmin=-30., vmax=30.,
                     colorbar_flag=False, title='Reflectivity',
                     min_lon=minlon,max_lon=maxlon,min_lat=minlat,max_lat=maxlat,embelish=False)
t1=ax1.text(0.025, 0.985, 'a)', verticalalignment='top', horizontalalignment='left',transform=ax1.transAxes, fontsize=90)
t1.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])

#make a colorbar
norm = plt.Normalize(vmin=-30, vmax=30)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='pyart_HomeyerRainbow'), pad=0.02, ax=ax1, extend='both')
cbar.set_label('Radar Reflectivity [dBZ]')

rhib=293
rhie=293
#plots rhi ring and spokes
for j in range(int(rhib),int(rhie)+1,10):
    ang=j
    if ang > 360.:
        ang=int(ang-360.)
    A,B = createCircleAroundWithRadius(klat2,klon2,(radar_PPI.range['data'][-1]-500.)/1000.,rhib,rhie+1,head1)
    C,D = getLocation(klat2,klon2,ang,(radar_PPI.range['data'][-1]-500.)/1000.)
    display.plot_line_geo(A,B,marker=None,color='k',linewidth=1)
    display.plot_line_geo([klon2,D],[klat2,C],marker=None,color='k',linewidth=2)
    if np.logical_and(C>minlat,np.logical_and(C<maxlat,np.logical_and(D>minlon,D<maxlat))):
        d1=plt.text(D, C, str(ang)+r'$^{\circ}$', horizontalalignment='center', transform=ccrs.PlateCarree(), fontsize=MEDIUM_SIZE, zorder=9)
        d1.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='xkcd:pale yellow')])
        
######
ax2 = plt.subplot2grid((8,4),(0,2), colspan=2, rowspan=3, projection=display.grid_projection)
ax2.plot(klon,klat,color='r',markersize=200,transform=display.grid_projection)
display.plot_ppi_map('vel_fix', swp_id, cmap=cmocean.cm.balance, ax=ax2, vmin=-40., vmax=40.,
                     colorbar_flag=False, title='Velocity',
                     min_lon=minlon,max_lon=maxlon,min_lat=minlat,max_lat=maxlat,embelish=False)
t2=ax2.text(0.025, 0.985, 'b)', verticalalignment='top', horizontalalignment='left',transform=ax2.transAxes, fontsize=90)
t2.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])

#make a colorbar
norm = plt.Normalize(vmin=-40, vmax=40)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmocean.cm.balance), pad=0.02, ax=ax2,extend='both')
cbar.set_label(r'Radial Velocity [m s$^{-1}$]')
# cbar.ax.tick_params(labelsize = 14)

scale_bar(ax1, length=8, location=(0.85, 0.02), linewidth=15, fontsize=32, text_pad=4, inner_spacing=0.5)
scale_bar(ax2, length=8, location=(0.85, 0.02), linewidth=15, fontsize=32, text_pad=4, inner_spacing=0.5)

#plots rhi ring and spokes
for j in range(int(rhib),int(rhie)+1,10):
    ang=j
    if ang > 360.:
        ang=int(ang-360.)
    A,B = createCircleAroundWithRadius(klat2,klon2,(radar_PPI.range['data'][-1]-500.)/1000.,rhib,rhie+1,head1)
    C,D = getLocation(klat2,klon2,ang,(radar_PPI.range['data'][-1]-500.)/1000.)
    display.plot_line_geo(A,B,marker=None,color='k',linewidth=1)
    display.plot_line_geo([klon2,D],[klat2,C],marker=None,color='k',linewidth=2)
    if np.logical_and(C>minlat,np.logical_and(C<maxlat,np.logical_and(D>minlon,D<maxlat))):
        d1=plt.text(D, C, str(ang)+r'$^{\circ}$', horizontalalignment='center', transform=ccrs.PlateCarree(), fontsize=MEDIUM_SIZE, zorder=9)
        d1.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='xkcd:pale yellow')])

#plots radar markers
# if np.logical_and(klat1>minlat,np.logical_and(klat1<maxlat,np.logical_and(klon1>minlon,klon1<maxlon))):
ax1.plot(klon1,klat1,marker='+', transform=ccrs.PlateCarree(),color='k',markersize=32,markeredgewidth=8,path_effects=[PathEffects.withStroke(linewidth=18,foreground='xkcd:pale blue')], zorder=10)
ax2.plot(klon1,klat1,marker='+', transform=ccrs.PlateCarree(),color='k',markersize=32,markeredgewidth=8,path_effects=[PathEffects.withStroke(linewidth=18,foreground='xkcd:pale blue')], zorder=10)
d1a=ax1.text(klon1-0.009, klat1-0.011, 'Ka1',transform=ccrs.PlateCarree(), zorder=10)
d1a.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='xkcd:pale blue')])
d1b=ax2.text(klon1-0.009, klat1-0.011, 'Ka1',transform=ccrs.PlateCarree(), zorder=10)
d1b.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='xkcd:pale blue')])
    
# if np.logical_and(klat1>minlat,np.logical_and(klat1<maxlat,np.logical_and(klon1>minlon,klon1<maxlon))):
ax1.plot(klon2,klat2,marker='+', transform=ccrs.PlateCarree(),color='k',markersize=32,markeredgewidth=8,path_effects=[PathEffects.withStroke(linewidth=18,foreground='xkcd:pale yellow')], zorder=10)
ax2.plot(klon2,klat2,marker='+', transform=ccrs.PlateCarree(),color='k',markersize=32,markeredgewidth=8,path_effects=[PathEffects.withStroke(linewidth=18,foreground='xkcd:pale yellow')], zorder=10)
d2a=ax1.text(klon2+0.006, klat2, 'Ka2',transform=ccrs.PlateCarree(), zorder=10)
d2a.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='xkcd:pale yellow')])
d2b=ax2.text(klon2+0.006, klat2, 'Ka2',transform=ccrs.PlateCarree(), zorder=10)
d2b.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='xkcd:pale yellow')])

fname = filsys+'roads/cb_2017_us_county_5m.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='gray')
ax1.add_feature(shape_feature, facecolor='none', linewidth=1.5, linestyle="--")
ax2.add_feature(shape_feature, facecolor='none', linewidth=1.5, linestyle="--")

xmin=getLocation(klat,klon,270,(radar_PPI.range['data'][-1]+8000)/1000)[1]
xmax=getLocation(klat,klon,90,(radar_PPI.range['data'][-1]+8000)/1000)[1]
ymin=getLocation(klat,klon,180,(radar_PPI.range['data'][-1]+8000)/1000)[0]
ymax=getLocation(klat,klon,0,(radar_PPI.range['data'][-1]+8000)/1000)[0]

ox.config(log_file=True, log_console=True, use_cache=True)
G = ox.graph_from_bbox(ymax,ymin,xmax,xmin)
ox.io.save_graph_shapefile(G, filepath=filsys+'roads/tmp'+str(0),encoding='utf-8')
fname = filsys+'roads/tmp'+str(0)+'/edges.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='gray', linewidth=0.5)
ax1.add_feature(shape_feature, facecolor='none')
ax2.add_feature(shape_feature, facecolor='none')
shutil.rmtree(filsys+'roads/tmp'+str(0)+'/')

fname = filsys+'roads/GPhighways.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black')
ax1.add_feature(shape_feature, facecolor='none')
ax2.add_feature(shape_feature, facecolor='none')

fname = filsys+'roads/builtupp_usa.shp'
x = [i.x for i in list(Reader(fname).geometries())]
y = [i.y for i in list(Reader(fname).geometries())]
xmin=getLocation(klat,klon,270,(radar_PPI.range['data'][-1]+8000)/1000)[1]
xmax=getLocation(klat,klon,90,(radar_PPI.range['data'][-1]+8000)/1000)[1]
ymin=getLocation(klat,klon,180,(radar_PPI.range['data'][-1]+8000)/1000)[0]
ymax=getLocation(klat,klon,0,(radar_PPI.range['data'][-1]+8000)/1000)[0]
ind=list(set(list(np.where(x>xmin)[0]))&set(list(np.where(x<xmax)[0]))&set(list(np.where(y>ymin)[0]))&set(list(np.where(y<ymax)[0])))
xsub = []
ysub=[]
namsub=[]
for i in ind:
    xsub.append(x[i])
    ysub.append(y[i])
    namsub.append(list(Reader(fname).records())[i].attributes['NAM'])
ax1.scatter(xsub,ysub,transform=ccrs.PlateCarree(),marker='*',c='blue',s=600)
ax2.scatter(xsub,ysub,transform=ccrs.PlateCarree(),marker='*',c='blue',s=600)
for i in range(len(ind)):
    if np.logical_and(ysub[i]>minlat,np.logical_and(ysub[i]<maxlat,np.logical_and(xsub[i]>minlon,xsub[i]<maxlon))):
        t1=ax1.text(xsub[i], ysub[i]+0.0075, namsub[i],horizontalalignment='center',transform=ccrs.PlateCarree(), fontsize=35)
        t1.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='white')])
        t2=ax2.text(xsub[i], ysub[i]+0.0075, namsub[i],horizontalalignment='center',transform=ccrs.PlateCarree(), fontsize=35)
        t2.set_path_effects([PathEffects.withStroke(linewidth=4,foreground='white')])

################################################ RHI PLOT
letters = ['c)','d)','e)','f)','g)','h)','i)','j)','k)','l)','m)','n)','o)','p)','q)','r)','s)','t)','u)','v)','w)','x)','y)','z)']
swps=[6,6,6,9,10]
jitter=[4,5,3,5,8]
c=0
cax1 = plt.axes([0.125, 0.095, 0.18, 0.01])
cax2 = plt.axes([0.325, 0.095, 0.18, 0.01])
cax3 = plt.axes([0.52, 0.095, 0.18, 0.01])
cax4 = plt.axes([0.72, 0.095, 0.18, 0.01])
for fi in range(len(ka2_file_RHIs)):
    radar_RHI = pyart.io.read(ka2_file_RHIs[fi])
    radar_RHI.elevation['data'] = radar_RHI.elevation['data']-radar_RHI.elevation['data'][jitter[fi]] #fix jitter
    scale=0.75
    n_swps = radar_RHI.nsweeps

    print('dealiasing')
    #create new variable for dealiasing velocities

    dealias_data = pyart.correct.region_dealias.dealias_region_based(radar_RHI)
    nyq=radar_RHI.get_nyquist_vel(0)

    radar_RHI.add_field('corrected_velocity', dealias_data, replace_existing=True)
    velfix=np.array([[]])
    for swp_id in range(n_swps):
        if swp_id == 0:
            velfix=aliasfix(radar_RHI.get_field(swp_id,'corrected_velocity'),13,nyq)
        else:
            velfix=np.append(velfix,aliasfix(radar_RHI.get_field(swp_id,'corrected_velocity'),13,nyq),axis=0)

    velfixdict=np.copy(dealias_data).tolist()
    velfixdict['data']=velfix
    radar_RHI.add_field('corrected_velocity', velfixdict, replace_existing=True)

    radar_name = radar_RHI.metadata['instrument_name']

    kadep=pd.read_csv('/home/aschueth/pythonscripts/Schuethetal2021/20180612_deployments_ka2.csv')

    swp_id=swps[fi]
    plotter = pyart.graph.RadarDisplay(radar_RHI)
    azimuth = radar_RHI.fixed_angle['data'][swp_id]
    print(azimuth)
    print(radar_RHI.scan_type)

    #creating the mask for attenuation
    reflectivity = radar_RHI.fields['reflectivity']['data']
    spectrum_width = radar_RHI.fields['spectrum_width']['data']
    velocity = radar_RHI.fields['corrected_velocity']['data'] #can I automatically calculate storm motion??
    total_power = radar_RHI.fields['total_power']['data']
    normal = radar_RHI.fields['normalized_coherent_power']['data']
    normal_mask = (normal < 0.4)
    range_mask=np.zeros(np.shape(reflectivity))
    for i in range(0,len(range_mask[:,0])):
        range_mask[i,:]=radar_RHI.range['data']>(13400)

    range_mask=range_mask.astype(bool)

    total_mask = [any(t) for t in zip(range_mask.flatten(), normal_mask.flatten())]

    refl_mask = np.ma.MaskedArray(reflectivity, mask=total_mask)
    sw_mask = np.ma.MaskedArray(spectrum_width, mask=total_mask)
    vel_mask = np.ma.MaskedArray(velocity, mask=total_mask)

    refl_dict = {'data':refl_mask}
    sw_dict = {'data':sw_mask}
    vel_dict = {'data':vel_mask}
    radar_RHI.add_field('refl_fix',refl_dict)
    radar_RHI.add_field('sw_fix',sw_dict)
    radar_RHI.add_field('vel_fix',vel_dict)

    currentscantime=datetime.strptime('20'+ka2_file_RHIs[fi].split("/")[-1][-15:-3], "%Y%m%d%H%M%S")

    for t in range(kadep.time_begin.count()):
        try:
            beginscan1=datetime.strptime(ka1dep.time_begin[t], "%m/%d/%Y %H:%M")
            endscan1=datetime.strptime(ka1dep.time_end[t], "%m/%d/%Y %H:%M")
        except:
            pass
        try:
            beginscan2=datetime.strptime(ka2dep.time_begin[t], "%m/%d/%Y %H:%M")
            endscan2=datetime.strptime(ka2dep.time_end[t], "%m/%d/%Y %H:%M")
        except:
            pass
        try:
            if currentscantime >= beginscan1 and currentscantime <= endscan1:
                try:
                    klat1=ka1dep.lat[t]
                    klon1=ka1dep.lon[t]
                    head1=ka1dep.heading[t]
                except:
                    klat1 = np.nan
                    klon1 = np.nan
                    head1 = np.nan

                try:
                    rhib1=ka1dep.rhib[t]
                    rhie1=ka1dep.rhie[t]
                except:
                    rhib1 = np.nan
                    rhie1 = np.nan

                if thefile.split("/")[-1][:3] == 'Ka1':
                    klat = klat1
                    klon = klon1
                    head = head1
        except: pass
        try:
            if currentscantime >= beginscan2 and currentscantime <= endscan2:
                try:
                    klat2=ka2dep.lat[t]
                    klon2=ka2dep.lon[t]
                    head2=ka2dep.heading[t]
                except:
                    klat2 = np.nan
                    klon2 = np.nan
                    head2 = np.nan

                try:
                    rhib2=ka2dep.rhib[t]
                    rhie2=ka2dep.rhie[t]
                except:
                    rhib2 = np.nan
                    rhie2 = np.nan

                if thefile.split("/")[-1][:3] == 'Ka2':
                    klat = klat2
                    klon = klon2
                    head = head2
        except: pass

    ang=head+azimuth
    if ang > 360.:
        ang=int(ang-360.)

    radar_RHI=radar_RHI.extract_sweeps([swp_id])
    vort, dist, height = RHIvort(radar_RHI,swp_id)
    
    ax3 = plt.subplot2grid((8,4),(fi+3,0))
    vplot=ax3.pcolormesh(dist/1000, height/1000, radar_RHI.fields['refl_fix']['data'][:-1,:], vmin=-30,vmax=30, cmap='pyart_HomeyerRainbow')
    plt.xlim(0, ((radar_RHI.range['data'][-1]-1500.)/1000.))
    t3=ax3.text(0.025, 0.985, letters[c], verticalalignment='top', horizontalalignment='left',transform=ax3.transAxes, fontsize=90)
    t3.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])

    plt.grid('on')
#     if fi == 0:
#         t3 = ax3.text(0.5, 1.18, 'Radar Reflectivity [dBZ]', verticalalignment='top', horizontalalignment='center',transform=ax3.transAxes, fontsize=50)    
    if fi == 4:
        plt.xlabel('Kilometers', fontsize=40,labelpad=-3)
        cb=fig.colorbar(vplot,cax=cax1,ticks=[-30,-20,-10,0,10,20,30],orientation='horizontal',extend='both')
        t3 = ax3.text(0.5, -0.5, 'Radar Reflectivity [dBZ]', verticalalignment='top', horizontalalignment='center',transform=ax3.transAxes, fontsize=50)    
        plt.xticks([2,4,6,8,10,12])

    else:
        plt.xlabel('')
        plt.xticks([2,4,6,8,10,12],labels=[])
    plt.xlim(13.5,0)
    plt.ylim(0,2.5)
    plt.yticks([0,0.5,1,1.5,2,2.5])
    plt.ylabel('Kilometers', fontsize=40)
    ax3.text(-0.23, 0.5, datetime.strptime('20'+ka2_file_RHIs[fi].split("/")[-1][-15:-3], "%Y%m%d%H%M%S").strftime("%H:%M UTC"),verticalalignment='center', horizontalalignment='left', rotation=90,transform=ax3.transAxes, fontsize=50)
    c+=1

    
    ax4 = plt.subplot2grid((8,4),(fi+3,1))
    vplot=ax4.pcolormesh(dist/1000, height/1000, radar_RHI.fields['sw_fix']['data'][:-1,:], vmin=0,vmax=4, cmap='cubehelix_r')
    plt.xlim(0, ((radar_RHI.range['data'][-1]-1500.)/1000.))
    t4=ax4.text(0.025, 0.985, letters[c], verticalalignment='top', horizontalalignment='left',transform=ax4.transAxes, fontsize=90)
    t4.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])

    plt.grid('on')
#     if fi == 0:
#         t4 = ax4.text(0.5, 1.18, 'Spectrum Width' r' [m s$^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax4.transAxes, fontsize=50)
    if fi == 4:
        plt.xlabel('Kilometers', fontsize=40,labelpad=-3)
        cb=fig.colorbar(vplot,cax=cax2,ticks=[0,0.5,1,1.5,2,2.5,3,3.5,4],orientation='horizontal',extend='both')
        t4 = ax4.text(0.5, -0.5, 'Spectrum Width' r' [m s$^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax4.transAxes, fontsize=50)
        plt.xticks([2,4,6,8,10,12])
    else:
        plt.xlabel('')
        plt.xticks([2,4,6,8,10,12],labels=[])
    plt.xlim(13.5,0)
    plt.ylim(0,2.5)
    plt.yticks([0,0.5,1,1.5,2,2.5],labels=[])
    plt.ylabel('')
    c+=1
    
    ax5 = plt.subplot2grid((8,4),(fi+3,2))
    vplot=ax5.pcolormesh(dist/1000, height/1000, radar_RHI.fields['vel_fix']['data'][:-1,:], vmin=-40,vmax=40, cmap=cmocean.cm.balance)
    plt.xlim(0, ((radar_RHI.range['data'][-1]-1500.)/1000.))
    t5=ax5.text(0.025, 0.985, letters[c], verticalalignment='top', horizontalalignment='left',transform=ax5.transAxes, fontsize=90)
    t5.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])
    plt.grid('on')
#     if fi == 0:
#         t5 = ax5.text(0.5, 1.18, 'Radial Velocity' r' [m s$^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax5.transAxes, fontsize=50)
    if fi == 4:
        plt.xlabel('Kilometers', fontsize=40,labelpad=-3)
        cb=fig.colorbar(vplot,cax=cax3,ticks=[-40,-30,-20,-10,0,10,20,30,40],orientation='horizontal',extend='both')
        t5 = ax5.text(0.5, -0.5, 'Radial Velocity' r' [m s$^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax5.transAxes, fontsize=50)
        plt.xticks([2,4,6,8,10,12])
    else:
        plt.xlabel('')
        plt.xticks([2,4,6,8,10,12],labels=[])
    plt.xlim(13.5,0)
    plt.ylim(0,2.5)
    plt.yticks([0,0.5,1,1.5,2,2.5],labels=[])
    plt.ylabel('')
    c+=1
    
    ax6 = plt.subplot2grid((8,4),(fi+3,3))
    vplot=ax6.pcolormesh(dist/1000, height/1000, vort, vmin=-0.15,vmax=0.15, cmap="PuOr_r")
    plt.xlim(0, ((radar_RHI.range['data'][-1]-1500.)/1000.))
    t6=ax6.text(0.025, 0.985, letters[c], verticalalignment='top', horizontalalignment='left',transform=ax6.transAxes, fontsize=90)
    t6.set_path_effects([PathEffects.withStroke(linewidth=10,foreground='white')])
    plt.grid('on')
#     if fi == 0:
#         t6 = ax6.text(0.5, 1.18, 'Inferred Vorticity' r' [$s^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax6.transAxes, fontsize=50)
    if fi == 4:
        plt.xlabel('Kilometers', fontsize=40,labelpad=-3)
        cb=fig.colorbar(vplot,cax=cax4,ticks=[-0.15,-0.1,-0.05,0,0.05,0.1,0.15],orientation='horizontal',extend='both')
        t6 = ax6.text(0.5, -0.5, 'Inferred Vorticity' r' [$s^{-1}$]', verticalalignment='top', horizontalalignment='center',transform=ax6.transAxes, fontsize=50)
        plt.xticks([2,4,6,8,10,12])
    else:
        plt.xlabel('')
        plt.xticks([2,4,6,8,10,12],labels=[])
    plt.xlim(13.5,0)
    plt.ylim(0,2.5)
    plt.yticks([0,0.5,1,1.5,2,2.5],labels=[])
    plt.ylabel('')
    c+=1
    
ax5.text(0.07, 0.41, r'Ka2 Azimuth 293$^{\circ}$',rotation=90, verticalalignment='top', horizontalalignment='left',transform=fig.transFigure, fontsize=60, fontweight='bold')
fig.subplots_adjust(wspace=0.1)
tit=plt.suptitle(r'Ka1 1.0$^{\circ}$ PPI 06/08/18 23:11 UTC',y=0.92, fontsize=70, fontweight='bold')
# plt.tight_layout()#pad=3.0)

plt.savefig(f'/home/aschueth/pythonscripts/Schuethetal2021/0608.png')