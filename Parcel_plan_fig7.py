from CM1calc import *
import matplotlib.pyplot as plt
import cmocean 

fil=625
filename = '/lustre/research/weiss/schueth/resdep125m/cm1out_000'+str(fil)+'.nc'
ds = xr.open_dataset(filename,chunks={'ni':288,'nj':288})
time = ds['time'].values
x = ds['xh']
y = ds['yh']
z = ds['z']

X,Y=np.meshgrid(x.values,y.values)

ref =ds['dbz'].isel(time=0,nk=0)

th_plan = ds['th'].isel(time=0,nk=0)
qr_plan = ds['qr'].isel(time=0,nk=0)
qi_plan = ds['qi'].isel(time=0,nk=0)
qc_plan = ds['qc'].isel(time=0,nk=0)
qs_plan = ds['qs'].isel(time=0,nk=0)
qg_plan = ds['qg'].isel(time=0,nk=0)
qv_plan = ds['qv'].isel(time=0,nk=0)

thrho=trho(th_plan,qv_plan,qi_plan,qc_plan,qs_plan,qr_plan,qg_plan)
trhop_plan=perturbation(thrho,thrho[0,-1])

fils='/lustre/research/weiss/schueth/resdep125m/cm1out_pdata.nc'
dsp = xr.open_dataset(fils)

rawfile = '/home/aschueth/pythonscripts/Schuethetal2021/RHIplane_parcels.txt'
col_names = ['pi', 't', 'h', 'x','y','z','baro','stretch']
df = pd.read_csv(rawfile, usecols=[0,1,2,3,4,5,6,7],names=col_names)

baro_mask=df.baro>np.percentile(df.baro[subset],99)
stretch_mask=df.stretch>np.percentile(df.stretch[subset],99)


fig = plt.figure(figsize=(25, 20),facecolor='white')
#row,column
ax1 = plt.subplot2grid((1,1),(0,0))
plan=plt.contourf(X,Y,trhop_plan,np.linspace(-9,0,100),cmap=cmocean.cm.rain_r,extend='both')
plt.contour(X, Y, ref, [40], colors='black',linewidths=3,linestyles='dashed')
cbar = plt.colorbar(plan, ticks=[-9,-8,-7,-6,-5,-4,-3,-2,-1,0])
cbar.ax.tick_params(labelsize=30)
plt.xlabel('Kilometers', fontsize=40)
plt.ylabel('Kilometers', fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=30)
t1=ax1.text(0.5, 0.99, r"$\theta'_\rho$ [K]", verticalalignment='top', horizontalalignment='center',transform=ax1.transAxes, fontsize=55)
t1.set_path_effects([PathEffects.withStroke(linewidth=6,foreground='white')])
plt.xlim(-20,30)
plt.ylim(-15,35)
plt.title('Time= '+str(int(time[0]/np.timedelta64(1, 's')))+' s   ',fontsize=50)

plt.plot(dsp.x[0,:1000]/1000,dsp.y[0,:1000]/1000,'ko')
plt.plot(dsp.x[0,:1000]/1000,dsp.y[0,:1000]/1000,'ko')
ax1.text(16.5, 9.5, "Parcel Release \n plane", verticalalignment='top', horizontalalignment='left', fontsize=40)

plt.plot(df.x[subset],df.y[subset],'ko')
ax1.text(-0.5, 5, "Parcel Analysis \n plane", verticalalignment='top', horizontalalignment='left', fontsize=40)

baro_parcels = np.array(df.pi[subset][baro_mask])
plt.scatter(dsp.x[:1850,baro_parcels]/1000,dsp.y[:1850,baro_parcels]/1000,c='darkviolet', s=0.000025)#,alpha=0.005)

stretch_parcels = np.array(df.pi[subset][stretch_mask])
plt.scatter(dsp.x[:1850,stretch_parcels]/1000,dsp.y[:1850,stretch_parcels]/1000,c='royalblue', s=0.000025)#,alpha=0.005);

svort_parcels = np.array(df.pi[somega_mask])
plt.scatter(dsp.x[:1850,svort_parcels]/1000,dsp.y[:1850,svort_parcels]/1000,c='firebrick', s=0.000025)#,alpha=0.005);

ax1.text(5, 17.25, "Baroclinic", c='darkviolet',verticalalignment='top', horizontalalignment='left', fontsize=30)
ax1.text(2.5, 18.5, "Max Vorticity",c='firebrick', verticalalignment='top', horizontalalignment='left', fontsize=30)
ax1.text(-0.25, 19.75, "Stretching",c='royalblue', verticalalignment='top', horizontalalignment='left', fontsize=30)
plt.savefig(f'/home/aschueth/pythonscripts/Schuethetal2021/parcel_plan.png')