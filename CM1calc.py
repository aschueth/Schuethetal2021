import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numba import jit
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import operator

def sftemp():
    '''
    Function to create pretty, non-uniform temperature colormapself.

    Parameters:
    Returns: cmap

    Example: plt.contourf(X,Y,trhop, cmap=CM1calc.sftemp())
    '''

    sfc_cdict ={'red':      ((0.00, 0.20, 0.20),
                             (0.08, 0.40, 0.40),
                             (0.17, 0.27, 0.27),
                             (0.25, 0.80, 0.80),
                             (0.33, 0.20, 0.20),
                             (0.42, 0.20, 0.20),
                             (0.50, 0.00, 0.00),
                             (0.58, 0.99, 0.99),
                             (0.67, 1.00, 1.00),
                             (0.75, 0.82, 0.82),
                             (0.83, 0.53, 0.53),
                             (0.92, 0.95, 0.95),
                             (1.00, 1.00, 1.00)),

            'green':        ((0.00, 0.20, 0.20),
                             (0.08, 0.40, 0.40),
                             (0.17, 0.00, 0.00),
                             (0.25, 0.60, 0.60),
                             (0.33, 0.40, 0.40),
                             (0.42, 0.60, 0.60),
                             (0.50, 0.39, 0.39),
                             (0.58, 0.76, 0.76),
                             (0.67, 0.36, 0.36),
                             (0.75, 0.02, 0.02),
                             (0.83, 0.00, 0.00),
                             (0.92, 0.03, 0.03),
                             (1.00, 0.60, 0.60)),

            'blue':         ((0.00, 0.60, 0.60),
                             (0.08, 0.60, 0.60),
                             (0.17, 0.65, 0.65),
                             (0.25, 1.00, 1.00),
                             (0.33, 1.00, 1.00),
                             (0.42, 0.40, 0.40),
                             (0.50, 0.07, 0.07),
                             (0.58, 0.02, 0.02),
                             (0.67, 0.00, 0.00),
                             (0.75, 0.01, 0.01),
                             (0.83, 0.00, 0.00),
                             (0.92, 0.52, 0.52),
                             (1.00, 0.80, 0.80))}


    sfc_coltbl = LinearSegmentedColormap('SFC_COLTBL',sfc_cdict)
    return sfc_coltbl

def RHIvort(radar,swp_id):
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
    
    vel = radar.get_field(swp_id,'vel_fix')
    phi = np.deg2rad(radar.get_elevation(swp_id))

    rangearray = np.tile(radar.range['data'],(len(phi),1))
        
    vort = np.zeros_like(vel)

    for i in range(1,25):    # iterate radii from 1 to 25 bins apart, to get a smoother wider field
        vortmp = np.zeros_like(vel)
        for j in range(i):    #iterate by possible slices to fill the full array
            slices = slice(j, None, i)
            vortmp[slices,:]=(1/rangearray[slices,:])*np.gradient(vel[slices,:],phi[slices],axis=0)
        
        #algorithm that takes the absolute largest value elementwise and keeps that in the vort array for all radii
        pinds = vortmp > np.absolute(vort)
        ninds = vortmp < -np.absolute(vort)

        vort[pinds]=vortmp[pinds]
        vort[ninds]=vortmp[ninds]
        
    return vort

def find_nearest(array, value):
    '''
    Function to find index of the array in which the value is closest to

    Parameters: array (array), value (number)
    Returns: index (int)

    Example: xind = CM1calc.find_nearest(x,5)
    '''

    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

@jit
def find_parcels(hvort,z,w):
    '''
    Function to filter out the unwanted parcels

    Parameters: hvort (array), z (array), w (array)
    Returns: a list of indices in which the parcels are 'good'

    This function is technically optimized by jit, more testing will be needed to see if it helps
    '''

    for i in range(len(hvort[0,:])):#filters out those parcels that don't meet the necessary conditions and saves time opening the data
        for j in range(len(hvort[:,0])):
            if np.logical_and(np.logical_and(np.logical_and((hvort[j,i]>0.03),(z[j,i]<=1)),(w[j,i]<=1)),(w[j,i]>0)):
                indlist=np.append(indlist,int(i))
                break
    return(indlist)

def label_count(array_label,num,data):
    '''
    Function to count label sizes and return the label that is the biggest and how big it is

    Parameters: array_label (array), num of labels in array_label (int), data (array) in which labels were created
    Returns: list of zipped numbers, first is the id of biggest label, second is the size of that label
    '''

    unique, counts = np.unique(array_label, return_counts=True)
    dicounts=(dict(zip(unique.compute(), counts.compute())))
    dicountsorted = sorted(dicounts.items(), key=operator.itemgetter(1),reverse=True)
    idbiggest= [i[0] for i in dicountsorted[1:]]
    sizebiggest= [i[1] for i in dicountsorted[1:]]
    return list(zip(idbiggest,sizebiggest))

def vector_gradient_components(u,v,w,resh,resv):
    '''
    Function calculates the gradients of a vector field given components in all directions.

    Parameter: u (x component vector field), v (y component vector field), w (z component vector field), resh (int), resv (1d array of dz)
    Returns: dudx,dudy,(dudz), dvdx,dvdy,(dvdz), dwdx,dwdy,(dwdz)
    '''

    d=len(np.shape(u))
    
    if d == 4:
        print('time,z,y,x')
        [dudx,dudy,dudz]=np.gradient(u.T,axis=(0,1,2))
        [dvdx,dvdy,dvdz]=np.gradient(v.T,axis=(0,1,2))
        [dwdx,dwdy,dwdz]=np.gradient(w.T,axis=(0,1,2))

        dudx = dudx/resh
        dudy = dudy/resh

        dvdx = dvdx/resh
        dvdy = dvdy/resh

        dwdx = dwdx/resh
        dwdy = dwdy/resh
        for i in range(len(resv)):
            dudz[:,:,i,:] = dudz[:,:,i,:]/resv[i]
            dvdz[:,:,i,:] = dvdz[:,:,i,:]/resv[i]
            dwdz[:,:,i,:] = dwdz[:,:,i,:]/resv[i]
        return (dudx.T,dudy.T,dudz.T, dvdx.T,dvdy.T,dvdz.T, dwdx.T,dwdy.T,dwdz.T)
    if d == 3:
        [dudx,dudy,dudz]=np.gradient(u.T)
        [dvdx,dvdy,dvdz]=np.gradient(v.T)
        [dwdx,dwdy,dwdz]=np.gradient(w.T)

        dudx = dudx/resh
        dudy = dudy/resh

        dvdx = dvdx/resh
        dvdy = dvdy/resh

        dwdx = dwdx/resh
        dwdy = dwdy/resh
        for i in range(len(resv)):
            dudz[:,:,i] = dudz[:,:,i]/resv[i]
            dvdz[:,:,i] = dvdz[:,:,i]/resv[i]
            dwdz[:,:,i] = dwdz[:,:,i]/resv[i]
        return (dudx.T,dudy.T,dudz.T, dvdx.T,dvdy.T,dvdz.T, dwdx.T,dwdy.T,dwdz.T)
    elif d == 2:
        [dudx,dudy]=np.gradient(u.T)
        [dvdx,dvdy]=np.gradient(v.T)
        [dwdx,dwdy]=np.gradient(w.T)

        dudx = dudx/resh
        dudy = dudy/resh

        dvdx = dvdx/resh
        dvdy = dvdy/resh

        dwdx = dwdx/resh
        dwdy = dwdy/resh
        return (dudx.T,dudy.T, dvdx.T,dvdy.T, dwdx.T,dwdy.T)
    else:
        raise Exception('Dimension needs to be 2 or 3 or 4')

def scalar_gradient_components(m,resh,resv):
    '''
    Function calculates the gradients of a scalar field.

    Parameter: m (scalar field), resh (int), resv (1d array of dz)
    Returns: dmdz, dmdy, (dmdz)
    '''

    d=len(np.shape(m))
    if d == 4:
        print('time,z,y,x')
        [dmdx,dmdy,dmdz]=np.gradient(m.T,axis=(0,1,2))

        dmdx = dmdx/resh
        dmdy = dmdy/resh
        for i in range(len(resv)):
            dmdz[:,:,i,:] = dmdz[:,:,i,:]/resv[i]
        return (dmdx.T,dmdy.T,dmdz.T)
    if d == 3:
        [dmdx,dmdy,dmdz]=np.gradient(m.T)

        dmdx = dmdx/resh
        dmdy = dmdy/resh
        for i in range(len(resv)):
            dmdz[:,:,i] = dmdz[:,:,i]/resv[i]
        return (dmdx.T,dmdy.T,dmdz.T)
    elif d == 2:
        [dmdx,dmdy]=np.gradient(m.T)

        dmdx = dmdx/resh
        dmdy = dmdy/resh
        return (dmdx.T,dmdy.T)
    else:
        raise Exception('Dimension needs to be 2 or 3 or 4')

def acceleration(u,v,w,prevu,prevv,prevw,x,z,dt=60):
    '''function calculates lagrangian acceleration by the material derivative
    
    Parameter:u,v,w straight from CM1
              prevu,prevv,prevw, the u, v, and w from the previous time step (assumed to be a minute)
              x,z straight from CM1
              
    acceleration = dvi/dt+v dot del v
    
    Returns: ax,ay,az,accel
    '''
    xm=x*1000 #units from km to m
    zm=z*1000 #units from km to m
    
    dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz=vector_gradient_components(u,v,w,np.gradient(xm)[0],np.gradient(zm))
    dudt=(u-prevu)/dt
    dvdt=(v-prevv)/dt
    dwdt=(w-prevw)/dt

    ax=dudt+u*dudx+v*dudy+w*dudz
    ay=dvdt+u*dvdx+v*dvdy+w*dvdz
    az=dwdt+u*dwdx+v*dwdy+w*dwdz

    wind = np.sqrt(u**2+v**2+w**2)
    a = np.sqrt(ax**2+ay**2+az**2)
    
    win = [list(i) for i in zip(list(u.values.flatten()/wind.values.flatten()),list(v.values.flatten()/wind.values.flatten()),list(w.values.flatten()/wind.values.flatten()))]
    acc = [list(i) for i in zip(list(ax.values.flatten()/a.values.flatten()),list(ay.values.flatten()/a.values.flatten()),list(az.values.flatten()/a.values.flatten()))]

    #aligned with wind
    acceldir = np.reshape([np.dot(win[b],acc[b]) for b in range(len(win))],np.shape(u))

    accel=a*acceldir
    return ax,ay,ax,accel

def thetae(qv, theta, P):
    '''
    Function calculates the equivalent potential temperature

    Parameter: qv, theta, P straight from CM1
    Returns: Equivalent Potential Temperature in K
    '''

    temp = (theta/np.power((100000/(P)),0.286))-273.15
    e = qv*P/(100.*0.622)
    tlcl = 55.0+(2840.0/(3.5*np.log(temp+273.15)-np.log(e)-4.805))
    tm = theta*np.power(((temp+273.15)/theta),(0.286*qv))
    thetae = tm*np.exp(((3376.0/tlcl)-2.54)*qv*(1.0+0.81*qv))
    return thetae

def trho(th,qv,qi,qc,qs,qr,qg,type='density'):
    '''
    Function calculates density or virtual potential temperature

    Parameters: th, qv, qi, qc, qs, qr, qg, type
    Returns: trho
    '''
    if type == 'density':
        return th*(1.+0.61*qv-qi-qc-qs-qr-qg)
    elif type == 'virtual':
        return th*(1.+0.61*qv)
    else:
        raise Exception('type needs to be density or virtual')

def perturbation(m,corner):
    '''
    Function calculates the perturbation of an array (m); typically for theta_rho, but could be anything. Can handle 2d or 3d arrays. 
    Corner is value or vetrical array as the base state

    Parameters: m, corner, type
    Returns: trho'
    '''

    d=len(np.shape(m))
    if d == 3:
        return (m.transpose() - corner).transpose()
    elif d ==2:
        return (m - corner)
    else:
        raise Exception('Dimension needs to be 2 or 3')

def thrgrad(th,qv,qi,qc,qs,qr,qg,x,z,type='density'):
    '''
    Function calculates the density or virtual potential temperature gradients. Change type to virtual for thetav, make sure all arrays are same shape and size.

    Parameters: th, qv, qi, qc, qs, qr, qg, x, z, type
    Returns: xtrgrad, ytrgrad, ztrgrad
    '''
    resh=dx(x)
    resv=dz(z)
    return scalar_gradient_components(trho(th,qv,qi,qc,qs,qr,qg),resh,resv)

def svort(u,v,w,xvort,yvort,zvort):
    '''
    Function calculates windspeed, streamwise vorticity, and vorticity magnitude both in 3d and horizontally

    Paramters: u,v,w,xvort,yvort,zvort
    Returns: zip(wndspd,wndspdh,svort,svorth,vortmag,vortmagh)
    '''

    wndspd=np.sqrt(u**2.+v**2.+w**2.)
    wndspdh=np.sqrt(u**2.+v**2.)
    svort=(u*xvort+v*yvort+w*zvort)/wndspd
    svorth=(u*xvort+v*yvort)/wndspdh
    vortmag = np.sqrt(xvort**2.+yvort**2.+zvort**2.)
    vortmagh = np.sqrt(xvort**2.+yvort**2.)
    return zip(wndspd,wndspdh,svort,svorth,vortmag,vortmagh)

def dx(x):
    '''
    Function finds the horizontal grid spacing, assuming no stretching

    Parameters: x [km] (array)
    Returns: dx [km] (int)
    '''
    return (x[int(len(x)/2)]-x[int(len(x)/2)-1]).values

def dz(z):
    '''
    Function calculates a list of dz, used due to stretching

    Parameters: z [km] (array)
    Returns: dz [km] (array)
    '''
    return np.gradient(z)

def UHcenter(x,z,w,zvort):
    '''
    Function finds the centroid of the gaussian smoothed 2-5 km UH, if fields are zero, return middle of domain

    Paramters: x (array), z (array), w (array), zvort (array)
    Returns: indices of centroid (y,x)

    Example: x0, y0 = CM1calc.UHcenter(x,z,w,zvort)
    '''
    
    gausfactor=np.around((1./dx(x))/0.16)
    UH = np.sum(w[find_nearest(z,2):find_nearest(z,5),:,:]*zvort[find_nearest(z,2):find_nearest(z,5),:,:],axis=0)
    UHfil=gaussian_filter(UH,sigma=gausfactor)>(0.75*np.amax(gaussian_filter(UH,sigma=gausfactor)))
    labelsmaxUH,numlabelsmaxUH=ndimage.label(UHfil)
    biggest=label_count(labelsmaxUH,numlabelsmaxUH,UHfil)
    mask=np.zeros_like(labelsmaxUH)
    try:
        mask[labelsmaxUH==biggest[0][0]]=1
        centroidsUH=ndimage.measurements.center_of_mass(mask)
    except:
        centroidsUH=(np.shape(UH)[0]/2,np.shape(UH)[0]/2)

    return centroidsUH

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

    for i in range(len(xh)):
        for j in range(len(yh)):
            if d == 3:
                for k in range(len(zh)):
                    var[k,j,i]=abs(array[k,j,i]-np.nanmean(array[k,j-delta:j+delta,i-delta:i+delta]))
                return(var)
            elif d == 2:
                var[i,j]=abs(array[i,j]-np.nanmean(array[i-delta:i+delta,j-delta:j+delta]))
                return(var)

@jit
def anglefix(array):
    '''If the angle is above or below 360, bring it down to the unit circle'''
    var=np.zeros_like(array)
    for i in range(len(xh)):
        for j in range(len(yh)):
            if array[i,j]>360:
                var[i,j]=array[i,j]-360
            elif array[i,j]<360:
                var[i,j]=array[i,j]+360
            else:
                var[i,j]=array[i,j]
    return(var)

@jit
def dotvectors(u1,v1,u2,v2):
    '''takes in u and v component arrays for vector 1 and 2 and calculates the dot product.
       Make sure that all are the same size'''
    var=np.zeros_like(u1)
    array1=np.array([u1, v1])
    array2=np.array([u2, v2])
    for i in range(len(xh)):
        for j in range(len(yh)):
            var[i,j]=np.dot(array1[:,i,j], array2[:,i,j])
    return(var)

def normalize(dat):
    return (dat-np.amin(dat))/(np.amax(dat)-np.amin(dat))

def KVN(u,v,w,x,z):
    '''
        This routine follows Lisa Schielicke's algorithm ported from matlab, and their publication
        (Schielicke et al., 2016). Velocity gradient tensor components are computed (Sii, Sjj, Skk, Sij, Sik, Sji, Sjk, Ski, Skj)
        to find the full deformation field. The Kinematic Vorticity number is then computed as a metric to identify
        individual vorticies coincident with regions of increased turblence and vorticity.

        Describes the excess or deficit of rotation relative to the deformation (including divergence).
        A Wk value of 1 is related to the balance of rotation and strain rate (i.e. a pure shearing motion)

        Parameters: u (3d array), v (3d array), w (3d array), x (1d array), z(1d array)
        Returns: KVN (3d array)
    '''

    resh=dx(x)*1000.
    resv=dz(z)*1000.

    #gradients:
    dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz = vector_gradient_components(u,v,w,resh,resv)

    #strain rate:
    sii=2.*dudx
    sjj=2.*dvdy
    skk=2.*dwdz
    sij=dvdx+dudy
    sik=dwdx+dudz
    sjk=dvdz+dwdy

    #vorticity tensor
    wij=-dvdx+dudy
    wik=-dwdx+dudz
    wjk=-dwdy+dvdz

    s3d =np.sqrt((sii/2.)**2 + (sjj/2.)**2 + (skk/2.)**2 + sij**2 + sik**2 + sjk**2)
    om3d=np.sqrt(wij**2 + wik**2 + wjk**2)

    wk3d=om3d/s3d

    return wk3d

def deformation_terms(u,v,w,x,z):
    '''
       Function for extracting the components of the velocity gradient tensor including the Rotation Rate and Strain Rate terms.

       Parameters: u (array), v (array), w (array), x (1d array), z (1d array)
       Returns: divergence, stretching, and shearing in z, y, and x respectively
   '''

    resh=dx(x)*1000.
    resv=dz(z)*1000.

    #gradients:
    dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz = vector_gradient_components(u,v,w,resh,resv)

    divz = dudx + dvdy
    stretchingz = dudx - dvdy
    shearingz   = dudy + dvdx

    divy = dudx + dwdz
    stretchingy = dudx - dwdz
    shearingy   = dwdx + dudz

    divx = dvdy + dwdz
    stretchingx = dvdy - dwdz
    shearingx   = dwdy + dvdz

    return(divz, stretchingz, shearingz, divy, stretchingy, shearingy, divx, stretchingx, shearingx)

def okubo_weiss(u,v,w,xvort,yvort,zvort,x,z):
    '''
    Function to calculate okubo-weiss parameter defined as stretching^2+shearing^2-vorticity^2

    Parameters: u (array), v (array), w (array), xvort (array), yvort (array), zvort (array), x (1d array), z (1d array)
    Returns: Okubo-Weiss Parameter (array)
    '''
    d=len(np.shape(u))
    if d == 3:
        divz, stretchingz, shearingz, divy, stretchingy, shearingy, divx, stretchingx, shearingx = deformation_terms(u,v,w,x,z)
        return ((stretchingx**2.+stretchingy**2.+stretchingz**2.)+(shearingx**2.+shearingy**2.+shearingz**2.)-(xvort**2.+yvort**2.+zvort**2.))
    elif d ==2:
        divz, stretchingz, shearingz, divy, stretchingy, shearingy, divx, stretchingx, shearingx = deformation_terms(u,v,w,x,z)
        return (stretchingz**2.+shearingz**2.-zvort**2.)
    else:
        raise Exception('Dimension needs to be 2 or 3')

def frontogenesis(u,v,w,m,x,z):
    '''
    Function to calculate frontogenesis, primarily used for thermodynamics but any scalar will work

    Parameters: u (array), v (array), w (array), m (array), x (1d array), z (1d array)
    Returns: frontogenesis (array)
    '''

    resh=dx(x)*1000.
    resv=dz(z)*1000.

    d=len(np.shape(u))
    if d == 3:
        dmdx, dmdy, dmdz = scalar_gradient_components(m, resh, resv)
        dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz = vector_gradient_components(u,v,w,resh,resv)

        mmag = np.sqrt(dmdx**2.+dmdy**2.+dmdz**2.)
        return (1./mmag)*((dmdx*(-dudx*dmdx-dvdx*dmdy-dwdx*dmdz))+(dmdy*(-dudy*dmdx-dvdy*dmdy-dwdy*dmdz))+(dmdz*(-dudz*dmdx-dvdz*dmdy-dwdz*dmdz)))*1000.

    elif d ==2:
        dmdx, dmdy = scalar_gradient_components(m, resh, resv)
        dudx,dudy,dvdx,dvdy,dwdx,dwdy = vector_gradient_components(u,v,w,resh,resv)

        mmag = np.sqrt(dmdx**2.+dmdy**2.)
        return (1./mmag)*((dmdx*(-dudx*dmdx-dvdx*dmdy))+(dmdy*(-dudy*dmdx-dvdy*dmdy)))*1000.
    else:
        raise Exception('Dimension needs to be 2 or 3')

# Todo:
# #solid body rotation
# #RHI
#
# poisson
# skewt
#     from raw
#     from input
#     from output
