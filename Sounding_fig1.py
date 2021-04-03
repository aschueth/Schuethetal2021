import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units
from scipy import signal
import xarray as xr

def bunkers(u,v,z,bottom=500,top=5500):
    z=z.to('m')
    z=z.m
    
    max_dz = np.max(np.gradient(z))
    
    u = u.to('m/s')
    v = v.to('m/s')
    u=u.m
    v=v.m

    if max_dz > 1:
        u=np.interp(np.arange(np.amin(z),np.amax(z)),z,u)
        v=np.interp(np.arange(np.amin(z),np.amax(z)),z,v)

    shear=[(u[top]-u[bottom]),(v[top]-v[bottom])]
    newslope=-1/(shear[1]/shear[0])
    bnew=np.mean(v[bottom:top])-(newslope*np.mean(u[bottom:top]))
    bold=v[bottom]-((shear[1]/shear[0])*u[bottom])
    uint=(bnew-bold)/((shear[1]/shear[0])-newslope)
    vint=newslope*uint+bnew
    ang=np.arctan2(newslope,1)
    if newslope>0:
        ubnk=uint-7.5*np.cos(ang)
        vbnk=vint-7.5*np.sin(ang)
    else:
        ubnk=uint+7.5*np.cos(ang)
        vbnk=vint+7.5*np.sin(ang)
    return ubnk*units('m/s'),vbnk*units('m/s')

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels


rawfile = '/home/aschueth/pythonscripts/Schuethetal2021/100mb_20170612_1931.txt'
col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']

df = pd.read_csv(rawfile, usecols=[0,1,2,3,4,5],names=col_names,delim_whitespace=True)

raw_p = df['pressure'].astype('float64').values * units.hPa
raw_T = df['temperature'].values * units.degC
raw_Td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
raw_hght = df['height'].values * units.meter
raw_u, raw_v = mpcalc.wind_components(wind_speed, wind_dir)

filename = '/lustre/research/weiss/schueth/resdep125m/cm1out_000001.nc'
ds = xr.open_dataset(filename,chunks={'ni':288,'nj':288})


u = (ds["uinterp"][0,:,0,0].values+10.5)* units('m/s')
v = (ds["vinterp"][0,:,0,0].values+8.7)* units('m/s')
theta = ds["th"][0,:,0,0].values* units.kelvin
qv = ds["qv"][0,:,0,0].values* units('kg/kg')
p = (ds["prs"][0,:,0,0].values* units.Pa).to('hPa')
hght = ds["z"].values * units.kilometer

T=(theta / ((1000.* units.hPa) / p).to('dimensionless')**0.286).to('degC')
e = mpcalc.vapor_pressure(p, qv)
Td = mpcalc.dewpoint(e)
Tv = mpcalc.virtual_temperature(T, qv)

# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(10, 10))

# Grid for plots
skew = SkewT(fig, rotation=30,aspect='auto')

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, Tv, 'purple',linewidth=1)
skew.plot(p, T, 'r')
skew.plot(raw_p, raw_T, 'r--')
skew.plot(p, Td, 'g')
skew.plot(raw_p, raw_Td, 'g--')

def pressure_interval(p,u,v,upper=100,lower=1000,spacing=50):
    intervals = list(range(upper,lower,spacing))
    ix = []
    for center in intervals:
        index = (np.abs(p-center)).argmin()
        if index not in ix:
            ix.append(index)
    return p[ix],u[ix],v[ix]
p_,u_,v_ = pressure_interval(p.magnitude,u.to('knots'),v.to('knots'))
skew.plot_barbs(p_, u_, v_)
skew.ax.set_ylim(1000, 100)

# skew2=skew.ax.twinx()
std_hghts=np.array([0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])*units.meter
for z in range(len(std_hghts)):
    skew.ax.axhline(mpcalc.add_height_to_pressure(p[0],std_hghts)[z],linestyle='--',color='k',alpha=0.5)
    
# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, Tv[0], Td[0]).to('degC')
skew.plot(p, prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
skew.shade_cin(p, Tv, prof,alpha=0.2)
skew.shade_cape(p, Tv, prof,alpha=0.2)

# Add the relevant special lines
skew.plot_dry_adiabats(t0=np.array([-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100])* units.degC,color='red',linewidth=1,linestyle='-',alpha=0.25)
skew.plot_moist_adiabats(color='k',linewidth=1.25,linestyle=':')
skew.plot_mixing_lines(color='green',linewidth=1,linestyle=':')
skew.ax.axhline(p[0], color='tab:blue')

# Good bounds for aspect ratio
skew.ax.set_xlim(-40, 60)
skew.ax.set_title('Modified input sounding based on the \nobserved 061217-1930 UTC P1 sounding',fontsize=25)
skew.ax.set_xlabel(r'Temperature [$^\circ$C]')
skew.ax.set_ylabel('Pressure [hPa]')


rect=[Rectangle((0.75,0.22), 0.2, 0.25)]
pc = PatchCollection(rect, facecolor='white', alpha=0.8,transform=skew.ax.transAxes,zorder=5)
skew.ax.add_collection(pc)

# Create a hodograph
ax_hod = inset_axes(skew.ax, '45%', '40%', loc=1)
h = Hodograph(ax_hod, component_range=80.)
for z in std_hghts:
    ax_hod.plot(u[np.abs(hght-z).argmin()].to('knots'),v[np.abs(hght-z).argmin()].to('knots'),'o',color='k',markersize=8.5)
h.add_grid(increment=20)
hodo_mask=[p>200.*units.hPa]
raw_hodo_mask=[raw_p>200.*units.hPa]
h.plot(u[hodo_mask].to('knots'), v[hodo_mask].to('knots'),color='k')
h.plot(raw_u[raw_hodo_mask].to('knots'), raw_v[raw_hodo_mask].to('knots'),color='b',linestyle='--')
ax_hod.plot(raw_u[0].to('knots'),raw_v[0].to('knots'),'o',color='b',markersize=8.5)
bunkersu,bunkersv=bunkers(u.to('knots'), v.to('knots'), hght)
ax_hod.plot(bunkersu,bunkersv,'o',color='red')
ax_hod.plot(10.5,8.7,'o',color='green')
# mpcalc.critical_angle(pressure, u, v, heights, stormu, stormv)
ax_hod.set_xlim(-20,55)
ax_hod.set_ylim(-10,65)
ax_hod.set_xticks([-20,-10,0,10,20,30,40,50])
ax_hod.set_yticks([-10,0,10,20,30,40,50,60])
ax_hod.text(5,4,'model',color='green')
ax_hod.text(10,22,'bunkers',color='red')
ax_hod.set_ylabel('knot',fontsize=15,labelpad=-10)
ax_hod.set_xlabel('knot',fontsize=15)

bunkerSRH=mpcalc.storm_relative_helicity(u.to('knots'), v.to('knots'), hght, 3*units.km, bottom=0*units.km, storm_u=bunkersu, storm_v=bunkersv)
realSRH=mpcalc.storm_relative_helicity(u.to('knots'), v.to('knots'), hght, 3*units.km, bottom=0*units.km, storm_u=10.5*units('m/s'), storm_v=8.7*units('m/s'))
bunkerCA=mpcalc.critical_angle(p, u.to('knots'), v.to('knots'), hght, bunkersu, bunkersv)
realCA=mpcalc.critical_angle(p, u.to('knots'), v.to('knots'), hght, 10.5*units('m/s'), 8.7*units('m/s'))
skew.ax.text(0.85, 0.45, f'0-3 km SRH:', verticalalignment='top', horizontalalignment='center',transform=skew.ax.transAxes, fontsize=15,zorder=6)
skew.ax.text(0.85, 0.41, str(np.around(bunkerSRH[-1],decimals=1).m)+r' m$^2$ s$^{-2}$', verticalalignment='top', horizontalalignment='center',transform=skew.ax.transAxes,color='red', fontsize=15,zorder=6)
skew.ax.text(0.85, 0.38, str(np.around(realSRH[-1],decimals=1).m)+r' m$^2$ s$^{-2}$', verticalalignment='top', horizontalalignment='center',transform=skew.ax.transAxes,color='green', fontsize=15,zorder=6)
skew.ax.text(0.85, 0.33, f'Critical Angle:', verticalalignment='top', horizontalalignment='center',transform=skew.ax.transAxes, fontsize=15,zorder=6)
skew.ax.text(0.85, 0.29, str(np.around(bunkerCA,decimals=1).m)+r'$^{\circ}$', verticalalignment='top', horizontalalignment='center',transform=skew.ax.transAxes,color='red', fontsize=15,zorder=6)
skew.ax.text(0.85, 0.26, str(np.around(realCA,decimals=1).m)+r'$^{\circ}$', verticalalignment='top', horizontalalignment='center',transform=skew.ax.transAxes,color='green', fontsize=15,zorder=6)

plt.savefig('/home/aschueth/pythonscripts/Schuethetal2021/sounding.pdf',dpi=300)