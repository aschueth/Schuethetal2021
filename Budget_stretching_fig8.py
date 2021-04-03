import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


rawfile = '/home/aschueth/pythonscripts/research/RHIplane_parcels.txt'
col_names = ['pi', 't', 'h', 'x','y','z','baro','stretch']
df = pd.read_csv(rawfile, usecols=[0,1,2,3,4,5,6,7],names=col_names)
subset=np.logical_and(np.logical_and(df.h>11,df.h<16),df.z<0.5)
stretch_mask=df.stretch>np.percentile(df.stretch[subset],99)

path='/lustre/research/weiss/schueth/resdep125m/'
dsparcels = xr.open_dataset('/lustre/research/weiss/schueth/new/cm1out_pdata.nc')

t = dsparcels.t.values

tilt_3d=[]
stretch_3d=[]
pgf_3d=[]
sgs_3d=[]
idiff_3d=[]
int_3d=[]
somega_3d=[]
omega_3d=[]
for parcelid in np.array(df.pi[subset][stretch_mask]):
    #read in tendency arrays
    xiten_sgs = dsparcels.xiten_sgs[:,parcelid].values
    xiten_idiff = dsparcels.xiten_idiff[:,parcelid].values
    xiten_ediff = dsparcels.xiten_ediff[:,parcelid].values
    xiten_pgf = dsparcels.xiten_pgf[:,parcelid].values
    xiten_cor = dsparcels.xiten_cor[:,parcelid].values

    etaten_sgs = dsparcels.etaten_sgs[:,parcelid].values
    etaten_idiff = dsparcels.etaten_idiff[:,parcelid].values
    etaten_ediff = dsparcels.etaten_ediff[:,parcelid].values
    etaten_pgf = dsparcels.etaten_pgf[:,parcelid].values
    etaten_cor = dsparcels.etaten_cor[:,parcelid].values

    zetaten_sgs = dsparcels.zetaten_sgs[:,parcelid].values
    zetaten_idiff = dsparcels.zetaten_idiff[:,parcelid].values
    zetaten_ediff = dsparcels.zetaten_ediff[:,parcelid].values
    zetaten_pgf = dsparcels.zetaten_pgf[:,parcelid].values
    zetaten_cor = dsparcels.zetaten_cor[:,parcelid].values

    #read in other prognostic variables
    xi = dsparcels.xi[:,parcelid].values
    eta = dsparcels.eta[:,parcelid].values
    zeta = dsparcels.zeta[:,parcelid].values

    u = dsparcels.u[:,parcelid].values
    v = dsparcels.v[:,parcelid].values
    w = dsparcels.w[:,parcelid].values

    dudx = dsparcels.dudx[:,parcelid].values
    dudy = dsparcels.dudy[:,parcelid].values
    dudz = dsparcels.dudz[:,parcelid].values

    dvdx = dsparcels.dvdx[:,parcelid].values
    dvdy = dsparcels.dvdy[:,parcelid].values
    dvdz = dsparcels.dvdz[:,parcelid].values

    dwdx = dsparcels.dwdx[:,parcelid].values
    dwdy = dsparcels.dwdy[:,parcelid].values
    dwdz = dsparcels.dwdz[:,parcelid].values

    xi_tilt    = zeta * dudz + eta * dudy
    xi_stretch = -xi * (dvdy+dwdz)
    xi_ts = xi_tilt + xi_stretch

    eta_tilt    = xi * dvdx + zeta * dvdz
    eta_stretch = -eta * (dudx+dwdz)
    eta_ts = eta_tilt + eta_stretch

    zeta_tilt    = xi * dwdx + eta * dwdy
    zeta_stretch = -zeta * (dudx+dvdy)
    zeta_ts = zeta_tilt + zeta_stretch

    wind = np.sqrt(u**2+v**2+w**2)
    windh = np.sqrt(u**2+v**2)

    omegah = np.sqrt(xi**2+eta**2)
    omega = np.sqrt(xi**2+eta**2+zeta**2)

    somegah = (u*xi+v*eta)/windh
    somega = (u*xi+v*eta+w*zeta)/wind

    horten_sgs = (dsparcels.etaten_sgs[:,parcelid].values*eta+dsparcels.xiten_sgs[:,parcelid].values*xi)/omegah
    horten_idiff = (dsparcels.etaten_idiff[:,parcelid].values*eta+dsparcels.xiten_idiff[:,parcelid].values*xi)/omegah
    horten_ediff = (dsparcels.etaten_ediff[:,parcelid].values*eta+dsparcels.xiten_ediff[:,parcelid].values*xi)/omegah
    horten_pgf = (dsparcels.etaten_pgf[:,parcelid].values*eta+dsparcels.xiten_pgf[:,parcelid].values*xi)/omegah
    horten_cor = (dsparcels.etaten_cor[:,parcelid].values*eta+dsparcels.xiten_cor[:,parcelid].values*xi)/omegah
    horten_tilt = ((zeta * dudz + eta * dudy)*xi+(xi * dvdx + zeta * dvdz)*eta)/omegah
    horten_stretch = ((-xi * (dvdy+dwdz))*xi+(-eta * (dudx+dwdz))*eta)/omegah

    totten_sgs = (dsparcels.zetaten_sgs[:,parcelid].values*zeta+dsparcels.etaten_sgs[:,parcelid].values*eta+dsparcels.xiten_sgs[:,parcelid].values*xi)/omega
    totten_idiff = (dsparcels.zetaten_idiff[:,parcelid].values*zeta+dsparcels.etaten_idiff[:,parcelid].values*eta+dsparcels.xiten_idiff[:,parcelid].values*xi)/omega
    totten_ediff = (dsparcels.zetaten_ediff[:,parcelid].values*zeta+dsparcels.etaten_ediff[:,parcelid].values*eta+dsparcels.xiten_ediff[:,parcelid].values*xi)/omega
    totten_pgf = (dsparcels.zetaten_pgf[:,parcelid].values*zeta+dsparcels.etaten_pgf[:,parcelid].values*eta+dsparcels.xiten_pgf[:,parcelid].values*xi)/omega
    totten_cor = (dsparcels.zetaten_cor[:,parcelid].values*zeta+dsparcels.etaten_cor[:,parcelid].values*eta+dsparcels.xiten_cor[:,parcelid].values*xi)/omega
    totten_tilt = ((zeta * dudz + eta * dudy)*xi+(xi * dvdx + zeta * dvdz)*eta+(xi * dwdx + eta * dwdy)*zeta)/omega
    totten_stretch = ((-xi * (dvdy+dwdz))*xi+(-eta * (dudx+dwdz))*eta+(-zeta * (dudx+dvdy))*zeta)/omega

    #set integrated variables to zero arrays initially
    xi_int         = np.zeros(np.shape(t))
    xi_tilt_int    = np.zeros(np.shape(t))
    xi_stretch_int = np.zeros(np.shape(t))
    xi_pgf_int     = np.zeros(np.shape(t))
    xi_sgs_int     = np.zeros(np.shape(t))
    xi_idiff_int   = np.zeros(np.shape(t))
    xi_ediff_int   = np.zeros(np.shape(t))
    xi_cor_int     = np.zeros(np.shape(t))
    xi_int_sum     = np.zeros(np.shape(t))
    xi_ts_int      = np.zeros(np.shape(t))

    eta_int         = np.zeros(np.shape(t))
    eta_tilt_int    = np.zeros(np.shape(t))
    eta_stretch_int = np.zeros(np.shape(t))
    eta_pgf_int      = np.zeros(np.shape(t))
    eta_sgs_int     = np.zeros(np.shape(t))
    eta_idiff_int   = np.zeros(np.shape(t))
    eta_ediff_int   = np.zeros(np.shape(t))
    eta_cor_int   = np.zeros(np.shape(t))
    eta_int_sum   = np.zeros(np.shape(t))
    eta_ts_int   = np.zeros(np.shape(t))

    zeta_int         = np.zeros(np.shape(t))
    zeta_tilt_int    = np.zeros(np.shape(t))
    zeta_stretch_int = np.zeros(np.shape(t))
    zeta_pgf_int      = np.zeros(np.shape(t))
    zeta_sgs_int     = np.zeros(np.shape(t))
    zeta_idiff_int   = np.zeros(np.shape(t))
    zeta_ediff_int   = np.zeros(np.shape(t))
    zeta_cor_int   = np.zeros(np.shape(t))
    zeta_int_sum   = np.zeros(np.shape(t))
    zeta_ts_int   = np.zeros(np.shape(t))

    hor_int         = np.zeros(np.shape(t))
    hor_tilt_int    = np.zeros(np.shape(t))
    hor_stretch_int = np.zeros(np.shape(t))
    hor_pgf_int      = np.zeros(np.shape(t))
    hor_sgs_int     = np.zeros(np.shape(t))
    hor_idiff_int   = np.zeros(np.shape(t))
    hor_ediff_int   = np.zeros(np.shape(t))
    hor_cor_int   = np.zeros(np.shape(t))
    hor_int_sum   = np.zeros(np.shape(t))

    tot_int         = np.zeros(np.shape(t))
    tot_tilt_int    = np.zeros(np.shape(t))
    tot_stretch_int = np.zeros(np.shape(t))
    tot_pgf_int      = np.zeros(np.shape(t))
    tot_sgs_int     = np.zeros(np.shape(t))
    tot_idiff_int   = np.zeros(np.shape(t))
    tot_ediff_int   = np.zeros(np.shape(t))
    tot_cor_int   = np.zeros(np.shape(t))
    tot_int_sum   = np.zeros(np.shape(t))

    #set the beginning of the integrated array the ceter difference between the first two values
    xi_int [0]          = 0.5 * (xi[0] + xi[1])
    xi_tilt_int[0]      = 0.5 * (xi_tilt[0] + xi_tilt[1])
    xi_stretch_int[0]   = 0.5 * (xi_stretch[0] + xi_stretch[1])
    xi_pgf_int [0]      = 0.5 * (xiten_pgf[0] + xiten_pgf[1])
    xi_sgs_int[0]       = 0.5 * (xiten_sgs[0] + xiten_sgs[1])
    xi_ediff_int[0]     = 0.5 * (xiten_ediff[0] + xiten_ediff[1])
    xi_idiff_int[0]     = 0.5 * (xiten_idiff[0] + xiten_idiff[1])
    xi_cor_int[0]       = 0.5 * (xiten_cor[0] + xiten_cor[1])
    xi_ts_int[0]        = 0.5 * (xi_ts[0] + xi_ts[1])

    eta_int [0]          = 0.5 * (eta[0] + eta[1])
    eta_tilt_int[0]      = 0.5 * (eta_tilt[0] + eta_tilt[1])
    eta_stretch_int[0]   = 0.5 * (eta_stretch[0] + eta_stretch[1])
    eta_pgf_int [0]      = 0.5 * (etaten_pgf[0] + etaten_pgf[1])
    eta_sgs_int[0]       = 0.5 * (etaten_sgs[0] + etaten_sgs[1])
    eta_ediff_int[0]     = 0.5 * (etaten_ediff[0] + etaten_ediff[1])
    eta_idiff_int[0]     = 0.5 * (etaten_idiff[0] + etaten_idiff[1])
    eta_cor_int[0]       = 0.5 * (etaten_cor[0] + etaten_cor[1])
    eta_ts_int[0]        = 0.5 * (eta_ts[0] + eta_ts[1])

    zeta_int [0]          = 0.5 * (zeta[0] + zeta[1])
    zeta_tilt_int[0]      = 0.5 * (zeta_tilt[0] + zeta_tilt[1])
    zeta_stretch_int[0]   = 0.5 * (zeta_stretch[0] + zeta_stretch[1])
    zeta_pgf_int [0]      = 0.5 * (zetaten_pgf[0] + zetaten_pgf[1])
    zeta_sgs_int[0]       = 0.5 * (zetaten_sgs[0] + zetaten_sgs[1])
    zeta_ediff_int[0]     = 0.5 * (zetaten_ediff[0] + zetaten_ediff[1])
    zeta_idiff_int[0]     = 0.5 * (zetaten_idiff[0] + zetaten_idiff[1])
    zeta_cor_int[0]       = 0.5 * (zetaten_cor[0] + zetaten_cor[1])
    zeta_ts_int[0]        = 0.5 * (zeta_ts[0] + zeta_ts[1])

    hor_int [0]          = 0.5 * (np.sqrt(xi**2+eta**2)[0] + np.sqrt(xi**2+eta**2)[1])
    hor_tilt_int[0]      = 0.5 * (horten_tilt[0] + horten_tilt[1])
    hor_stretch_int[0]   = 0.5 * (horten_stretch[0] + horten_stretch[1])
    hor_pgf_int [0]      = 0.5 * (horten_pgf[0] + horten_pgf[1])
    hor_sgs_int[0]       = 0.5 * (horten_sgs[0] + horten_sgs[1])
    hor_ediff_int[0]     = 0.5 * (horten_ediff[0] + horten_ediff[1])
    hor_idiff_int[0]     = 0.5 * (horten_idiff[0] + horten_idiff[1])
    hor_cor_int[0]       = 0.5 * (horten_cor[0] + horten_cor[1])

    tot_int [0]          = 0.5 * (np.sqrt(xi**2+eta**2+zeta**2)[0] + np.sqrt(xi**2+eta**2+zeta**2)[1])
    tot_tilt_int[0]      = 0.5 * (totten_tilt[0] + totten_tilt[1])
    tot_stretch_int[0]   = 0.5 * (totten_stretch[0] + totten_stretch[1])
    tot_pgf_int [0]      = 0.5 * (totten_pgf[0] + totten_pgf[1])
    tot_sgs_int[0]       = 0.5 * (totten_sgs[0] + totten_sgs[1])
    tot_ediff_int[0]     = 0.5 * (totten_ediff[0] + totten_ediff[1])
    tot_idiff_int[0]     = 0.5 * (totten_idiff[0] + totten_idiff[1])
    tot_cor_int[0]       = 0.5 * (totten_cor[0] + totten_cor[1])

    #     xi_tilt_int[j] = xi_tilt_int[j-1] + xi_tilt[j] * dt

    def integrate(var,varint,j,dt):
    #     print(varint[j-1])
    #     print(dt*(var[j-1] + var[j])/2.)
        return varint[j-1] + dt*(var[j-1] + var[j])/2.

    #integrate through time
    for j in range(1,len(t)):
        dt=(t[j]-t[j-1])

        xi_tilt_int[j] = integrate(xi_tilt,xi_tilt_int,j,dt)
        xi_stretch_int[j] = integrate(xi_stretch,xi_stretch_int,j,dt)
        xi_pgf_int[j] = integrate(xiten_pgf,xi_pgf_int,j,dt)
        xi_sgs_int[j] = integrate(xiten_sgs,xi_sgs_int,j,dt)
        xi_ediff_int[j] = integrate(xiten_ediff,xi_ediff_int,j,dt)
        xi_idiff_int[j] = integrate(xiten_idiff,xi_idiff_int,j,dt)
        xi_cor_int[j] = integrate(xiten_cor,xi_cor_int,j,dt)
        xi_ts_int[j] = integrate(xi_ts,xi_ts_int,j,dt)
        xi_int[j] = xi_int[j-1]+xi_tilt[j]*dt+xi_stretch[j]*dt+xiten_pgf[j]*dt+xiten_sgs[j]*dt+xiten_ediff[j]*dt+xiten_idiff[j]*dt+xiten_cor[j]*dt

        eta_tilt_int[j] = integrate(eta_tilt,eta_tilt_int,j,dt)
        eta_stretch_int[j] = integrate(eta_stretch,eta_stretch_int,j,dt)
        eta_pgf_int[j] = integrate(etaten_pgf,eta_pgf_int,j,dt)
        eta_sgs_int[j] = integrate(etaten_sgs,eta_sgs_int,j,dt)
        eta_ediff_int[j] = integrate(etaten_ediff,eta_ediff_int,j,dt)
        eta_idiff_int[j] = integrate(etaten_idiff,eta_idiff_int,j,dt)
        eta_cor_int[j] = integrate(etaten_cor,eta_cor_int,j,dt)
        eta_ts_int[j] = integrate(eta_ts,eta_ts_int,j,dt)
        eta_int[j] = eta_int[j-1]+eta_tilt[j]*dt+eta_stretch[j]*dt+etaten_pgf[j]*dt+etaten_sgs[j]*dt+etaten_ediff[j]*dt+etaten_idiff[j]*dt+etaten_cor[j]*dt

        zeta_tilt_int[j] = integrate(zeta_tilt,zeta_tilt_int,j,dt)
        zeta_stretch_int[j] = integrate(zeta_stretch,zeta_stretch_int,j,dt)
        zeta_pgf_int[j] = integrate(zetaten_pgf,zeta_pgf_int,j,dt)
        zeta_sgs_int[j] = integrate(zetaten_sgs,zeta_sgs_int,j,dt)
        zeta_ediff_int[j] = integrate(zetaten_ediff,zeta_ediff_int,j,dt)
        zeta_idiff_int[j] = integrate(zetaten_idiff,zeta_idiff_int,j,dt)
        zeta_cor_int[j] = integrate(zetaten_cor,zeta_cor_int,j,dt)
        zeta_ts_int[j] = integrate(zeta_ts,zeta_ts_int,j,dt)
        zeta_int[j] = zeta_int[j-1]+zeta_tilt[j]*dt+zeta_stretch[j]*dt+zetaten_pgf[j]*dt+zetaten_sgs[j]*dt+zetaten_ediff[j]*dt+zetaten_idiff[j]*dt+zetaten_cor[j]*dt

        hor_tilt_int[j] = integrate(horten_tilt,hor_tilt_int,j,dt)
        hor_stretch_int[j] = integrate(horten_stretch,hor_stretch_int,j,dt)
        hor_pgf_int[j] = integrate(horten_pgf,hor_pgf_int,j,dt)
        hor_sgs_int[j] = integrate(horten_sgs,hor_sgs_int,j,dt)
        hor_ediff_int[j] = integrate(horten_ediff,hor_ediff_int,j,dt)
        hor_idiff_int[j] = integrate(horten_idiff,hor_idiff_int,j,dt)
        hor_cor_int[j] = integrate(horten_cor,hor_cor_int,j,dt)
        hor_int[j] = hor_int[j-1]+horten_tilt[j]*dt+horten_stretch[j]*dt+horten_pgf[j]*dt+horten_sgs[j]*dt+horten_ediff[j]*dt+horten_idiff[j]*dt+horten_cor[j]*dt

        tot_tilt_int[j] = integrate(totten_tilt,tot_tilt_int,j,dt)
        tot_stretch_int[j] = integrate(totten_stretch,tot_stretch_int,j,dt)
        tot_pgf_int[j] = integrate(totten_pgf,tot_pgf_int,j,dt)
        tot_sgs_int[j] = integrate(totten_sgs,tot_sgs_int,j,dt)
        tot_ediff_int[j] = integrate(totten_ediff,tot_ediff_int,j,dt)
        tot_idiff_int[j] = integrate(totten_idiff,tot_idiff_int,j,dt)
        tot_cor_int[j] = integrate(totten_cor,tot_cor_int,j,dt)
        tot_int[j] = tot_int[j-1]+totten_tilt[j]*dt+totten_stretch[j]*dt+totten_pgf[j]*dt+totten_sgs[j]*dt+totten_ediff[j]*dt+totten_idiff[j]*dt+totten_cor[j]*dt

    tilt_3d.append(tot_tilt_int)
    stretch_3d.append(tot_stretch_int)
    pgf_3d.append(tot_pgf_int)
    sgs_3d.append(tot_sgs_int)
    idiff_3d.append(tot_idiff_int)
    int_3d.append(tot_int)
    somega_3d.append(somega)
    omega_3d.append(omega)
    
SMALL_SIZE = 20
MEDIUM_SIZE = 40
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('legend', handlelength=5)
# rcParams["legend.handlelength"] = 2.0

fig, ax1 = plt.subplots(figsize=(17,8),facecolor='white')
l1=plt.plot(dsparcels.t.values[:1850],np.mean(np.array(tilt_3d),axis=0)[:1850]+np.mean(np.array(stretch_3d),axis=0)[:1850], label='Stretching',linestyle='dashed',linewidth=4)
l2=plt.plot(dsparcels.t.values[:1850],np.mean(np.array(pgf_3d),axis=0)[:1850], label='Baroclinic',linestyle='dotted',linewidth=4)
l3=plt.plot(dsparcels.t.values[:1850],np.mean(np.array(sgs_3d),axis=0)[:1850], label='SGS',linestyle=(0,(3,1,1,1,1,1)),linewidth=4)
l4=plt.plot(dsparcels.t.values[:1850],np.mean(np.array(idiff_3d),axis=0)[:1850], label='Implicit Diffusion',linewidth=4)
l5=plt.plot(dsparcels.t.values[:1850],np.mean(np.array(int_3d),axis=0)[:1850],c='k', label='Total',linewidth=5)
l6=plt.plot(dsparcels.t.values[:1850],np.mean(np.array(somega_3d),axis=0)[:1850],c='k', label='Streamwise',linewidth=0.5)

plt.fill_between(dsparcels.t.values[:1850], np.amin(np.array(tilt_3d),axis=0)[:1850]+np.amin(np.array(stretch_3d),axis=0)[:1850], np.amax(np.array(tilt_3d),axis=0)[:1850]+np.amax(np.array(stretch_3d),axis=0)[:1850],color=matplotlib.cm.tab10(0),alpha=0.1)
plt.fill_between(dsparcels.t.values[:1850], np.amin(np.array(pgf_3d),axis=0)[:1850], np.amax(np.array(pgf_3d),axis=0)[:1850],color=matplotlib.cm.tab10(1),alpha=0.1)
plt.fill_between(dsparcels.t.values[:1850], np.amin(np.array(sgs_3d),axis=0)[:1850], np.amax(np.array(sgs_3d),axis=0)[:1850],color=matplotlib.cm.tab10(2),alpha=0.1)
plt.fill_between(dsparcels.t.values[:1850], np.amin(np.array(idiff_3d),axis=0)[:1850], np.amax(np.array(idiff_3d),axis=0)[:1850],color=matplotlib.cm.tab10(3),alpha=0.1)
plt.fill_between(dsparcels.t.values[:1850], np.amin(np.array(int_3d),axis=0)[:1850], np.amax(np.array(int_3d),axis=0)[:1850],color='k',alpha=0.1)

plt.title('3D Vorticity Production in the Stretching Regime',fontsize=25)
plt.ylabel(r'Vorticity Magnitude [$s^{-1}$]')
plt.xlabel('Model Time [s]')
plt.grid()
plt.ylim(-0.08,0.08)

ax2 = ax1.twinx()
plt.ylabel('Altitude of Parcel [m AGL]')
l7=plt.plot(dsparcels.t.values[:1850],np.mean(dsparcels.z[:1850,np.array(df.pi[subset][stretch_mask])],axis=1),c='lightgray',linewidth=4,label='Parcel Altitude')
plt.fill_between(dsparcels.t.values[:1850], np.amin(dsparcels.z[:1850,np.array(df.pi[subset][stretch_mask])],axis=1), np.amax(dsparcels.z[:1850,np.array(df.pi[subset][stretch_mask])],axis=1),color='lightgray',alpha=0.3)
plt.ylim(0,4000)
ax1.set_zorder(1)  # default zorder is 0 for ax1 and ax2
ax1.patch.set_visible(False)  # prevents ax1 from hiding ax2
plt.axvline(x=9360,c='brown',alpha=0.5)
plt.text(9360,-200,'9360',c='brown',fontsize=20,horizontalalignment='right')

lns = l1+l2+l3+l4+l5+l6+l7
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='lower left', fontsize=18)

plt.savefig(f'/home/aschueth/pythonscripts/research/plots/stretching_budget.pdf',dpi=300)

int_3d=np.stack(int_3d,axis=0)
omega_3d=np.stack(omega_3d,axis=0)
RMSE_h=[]
RMSPE_h=[]
RMSE=[]
RMSPE=[]
for i, parcelid in enumerate(np.array(df.pi[subset][stretch_mask])):
    RMSE_h.append(np.sqrt(np.mean((int_3d[i,:find_nearest(dsparcels.z[:,parcelid],1000)]-omega_3d[i,:find_nearest(dsparcels.z[:,parcelid],1000)])**2)))
    RMSE.append(np.sqrt(np.mean((int_3d[i,:1850]-omega_3d[i,:1850])**2)))
    RMSPE_h.append(np.sqrt(np.mean(((int_3d[i,:find_nearest(dsparcels.z[:,parcelid],1000)]-omega_3d[i,:find_nearest(dsparcels.z[:,parcelid],1000)])/int_3d[i,:find_nearest(dsparcels.z[:,parcelid],1000)])**2))*100)
    RMSPE.append(np.sqrt(np.mean(((int_3d[i,:1850]-omega_3d[i,:1850])/int_3d[i,:1850])**2))*100)

print(np.mean(np.array(RMSE)), np.mean((np.array(RMSPE))))
print(np.mean(np.array(RMSE_h)),np.mean(np.array(RMSPE_h)))