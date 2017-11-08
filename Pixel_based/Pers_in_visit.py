
# coding: utf-8

# # Notebook to study short term persistence from multiple exposures in a single visit 
# 

# In[ ]:

from astropy.io import fits
import glob, os, shutil, pickle, bz2, gc
import numpy as np
import pandas as pd
from scipy.stats import sigmaclip
from scipy.optimize import curve_fit
from scipy.special import gammaincc, gamma
from astropy.wcs import WCS
from astropy.stats import histogram
from itertools import product
from multiprocessing import Pool
from crds import bestrefs


# In[ ]:

# The project dir 
pdir = '/user/gennaro/Functional_work/WFC3_persistence/py_progs/short_term_persistence/'

#The mosaic dir
mdir = pdir+'/Mosaic_hi_res_folder/'

#The dir to save/load the Persistence curves dataframes
sdir = pdir+'/PD_dataframes_dir/'


# In[ ]:

# conversion factor from days to seconds
daytosec = 24.*3600.


# In[ ]:

#Single and double exponential models to be fitted to the data

def decay1(t,a1,t1):
    e1 = a1*np.exp(-t/t1)
    return e1

def intdec1(t,a1,t1):
    tu = t[1:]
    td = t[:-1]
    k  = -a1*t1
    return k*(np.exp(-tu/t1)-np.exp(-td/t1))/(tu-td)
    
def decay2(t,a1,t1,a2,t2):
    e1 = a1*np.exp(-t/t1)
    e2 = a2*np.exp(-t/t2)
    return e1+e2

def intdec2(t,a1,t1,a2,t2):
    tu = t[1:]
    td = t[:-1]
    k1,k2  = -a1*t1, - a2*t2
    
    return k1*(np.exp(-tu/t1)-np.exp(-td/t1))/(tu-td) + k2*(np.exp(-tu/t2)-np.exp(-td/t2))/(tu-td)

#Single exponential models plus a constant

def intdec1_plusconst(t,a1,t1,q):
    tu = t[1:]
    td = t[:-1]
    k  = -a1*t1
    return k*(np.exp(-tu/t1)-np.exp(-td/t1))/(tu-td) +q

def dec1_plusconst(t,a1,t1,q):
    e1 = a1*np.exp(-t/t1)
    return e1+q


#Shifted power law model

def shpwl(t,t0,A,index):
    return A * ((t+t0)/1000)**index

def intshpwl(t,t0,A,index):
    tu = t[1:]
    td = t[:-1]

    if (index == -1.):
        return A*np.log( (tu+t0)/(td+t0) )
    else:
        return A/(1+index) * ( ((tu+t0)/1000)**(1+index) - ((td+t0)/1000)**(1+index) )/(tu-td)
    
    
#Schechter like model

def schechter(t,phi,alpha,tstar):
    x = t/tstar
    return phi*(x**alpha)*np.exp(-x)

def intschechter(t,phi,alpha,tstar):
    x = t/tstar

    tu = x[1:]
    td = x[:-1]

    g1 = gammaincc(alpha+1,td)
    g2 = gammaincc(alpha+1,tu)
    
    diff = gamma(alpha+1)*(g1-g2)
    
    return phi*diff


#Geometric median calculation function

from scipy.spatial.distance import cdist, euclidean

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1, D

        y = y1



# In[ ]:

# Define a function that takes the vsflts list, the current flt that is being used as stimulus
# and looks for all the pixels with valid stimulus values AND that have valid ramps (i.e. no source, only sky)
# in the following exposures AND that where not stimulated more than prior-stim-factor (psf)% of the current stimulus in 
# ANY past exposure up to a certain look back exposure (lb).
# Mario: added the option of multiplying the stimulus by the Pixel Area Map

def find_ramps(istim,flts,lev_u,lev_d,lb=None,PAM=None,psf=0.1):
        
    stimdata  = flts[istim,:,:]*(tendMJDs[istim]-tstrMJDs[istim])*daytosec
    
    if PAM is not None:
        stimdata *= PAM
    
    istimgood = (stimdata > lev_d) & (stimdata < lev_u)  
    print('Pixels with potentially right stimuli:',np.sum(istimgood) )

    if lb is not None:
        if (istim-lb)>0:
            st = istim-lb
        else:
            st = 0
    else:
        st = 0
    
    for i in range(st,istim,1):
        persdata = flts[i,:,:] * (tendMJDs[i]-tstrMJDs[i])*daytosec 
        if PAM is not None:
            persdata *= PAM

        if (imtyps[i] == 'EXT'):
            istimgood = istimgood & (persdata < psf*stimdata) 
            
    print('Pixels with really right stimuli:',np.sum(istimgood) )
    
    
    icount    = np.zeros_like(stimdata,dtype=np.int_)
    iprev     = istimgood
    for i in range(istim+1,len(imtyps),1):
    
        persdata = flts[i,:,:] * (tendMJDs[i]-tstrMJDs[i])*daytosec
        if PAM is not None:
            persdata *= PAM
    
        if (imtyps[i] == 'EXT'):
            msky = np.nanmean(sigmaclip(persdata,2.,2.)[0])
            ssky = np.nanstd(sigmaclip(persdata,2.,2.)[0])
            iskycurr = (persdata <msky+2*ssky) & (persdata >msky-2*ssky)    
        elif (imtyps[i] == 'DARK'):
            iskycurr = np.ones_like(persdata,dtype=np.bool_)
        else:
            print('Wrong image type')
            assert False
        
        igood = istimgood & iskycurr & iprev
        iprev = igood
        icount[igood] += 1

        print('Pixels with ramps extending for at least',i-istim,' exposures:', igood.sum())
        
        if (np.sum(igood) == 0):
            break
                 
    return icount


# In[ ]:

# Dedfine a function to get the sky value in the cureent flt pixel, but measured form the AD mosaic

def getskyfrommosaic(wcsAD, wcsFLT, x, y, dxgrid, dygrid, skyrad_o,skyrad_i,mask_sky_contam,mosaic):

    coords = wcsAD.all_world2pix(wcsFLT.all_pix2world(np.array([[x,y]],dtype=np.float_),0),0) 
    dx = coords[0,0]
    dy = coords[0,1]
    
    dst = np.sqrt((dxgrid-dx)**2 + (dygrid-dy)**2)
    msk = (dst<skyrad_o) & (dst > skyrad_i) & mask_sky_contam 
    skyarr = mosaic[1].data[msk]
    cskyarr,l,u = sigmaclip(skyarr,2.,2.)
    return np.nanmean(cskyarr)

# Similar but for getting background values from the smae image

def getlocalbackground (x, y, xgrid, ygrid, skyrad_o,skyrad_i, fltdata):

    dst = np.sqrt((xgrid-x)**2 + (ygrid-y)**2)
    msk = (dst<skyrad_o) & (dst > skyrad_i)
    skyarr = fltdata[msk]
    cskyarr,l,u = sigmaclip(skyarr,2.,2.)
    return np.nanmean(cskyarr)

#def get_dark_rate(x, y, istim, j)


# In[ ]:

def get_sky_and_indices(nz0, nz1, j, istim):
    # Function to get the sky value, as well as calculate the indices needed in the large data cube of IMA reads
       
    #Get the sky from the drizzled image
    imtyp = imtyps[istim+j]

    if(imtyp == 'EXT'):
        skyhere = getskyfrommosaic(w_mosaic, w_vsflts[istim+j], nz1, nz0, dxgrid, dygrid, skyrad_o,skyrad_i,mask_sky_contam,mosaic)
#        skyhere = getlocalbackground(nz1, nz0, xgrid, ygrid, skyrad_o,skyrad_i, flts[istim+j,:,:])
    elif(imtyp == 'DARK'):
        skyhere = 0.0 #Do not subtract any sky, as superdark is subtracted from IMA already
    else:
        print('Wrong image type')
        assert False
    
    offset = ( tstrMJDs[istim+j] - tendMJDs[istim])*daytosec
    ioff = np.sum(nsamps[0:istim+j])
    nsamp = nsamps[istim+j]
    
    k_product = product([nz0], [nz1], [skyhere], [offset], [ioff], [nsamp], [j], range(nsamp-1))
    return list(k_product)


# In[ ]:

def get_pixel_values(inputs, istim, PAM=None):
    # function to extract the values from the IMA cube and IMA metadata arrays
    
    nz1, nz0, skyhere, offset, ioff, nsamp, j, k = inputs
    te = ima_times[ioff+k]
    ts = ima_times[ioff+k+1]

    tfromstim = te + offset
    tdenom    = te - ts  
    meancurr  = (ima_scis[ioff+k,nz1,nz0]*te - ima_scis[ioff+k+1,nz1,nz0]*ts)/tdenom
    stdvcurr  = np.sqrt(np.sum(np.square([ima_errs[ioff+k,nz1,nz0]*te,ima_errs[ioff+k+1,nz1,nz0]*ts])))/tdenom

    if np.isnan(meancurr) == True:
        
        print(te,ts,ima_scis[ioff+k,nz1,nz0],ima_scis[ioff+k+1,nz1,nz0],inputs)
        assert False
    
    if ((PAM is not None) & (imtyps[istim+j] == 'EXT')):
        meancurr = (meancurr-skyhere)*PAM[nz1,nz0]
        stdvcurr *= PAM[nz1,nz0]
        
    exptime = (tendMJDs[istim]-tstrMJDs[istim])*daytosec
    return [flts[istim,nz1,nz0]*exptime,
            exptime,
            nz0,
            nz1,
            tfromstim,
            tdenom,
            nsamp-k-1,
            nsamp,
            meancurr,
            stdvcurr,
            istim,
            istim+j,
            imtyps[istim],
            imtyps[istim+j],
            flts_dqs[istim,nz1,nz0],
            ima_dqs[ioff+k,nz1,nz0]
           ]


# In[ ]:

#Read files header, make sure they are sorted by EXPSTART
#This now copies the files into the mosaic hi res directory, keeping visit structure

sflts= []

for vis in ['1','2','3']:
    qldir = pdir+'/14016_data/Visit0'+vis+'/'
    wdir = mdir+'/Visit0'+vis+'/'
    if not os.path.isdir(wdir):
        os.mkdir(wdir)
    flts = glob.glob(qldir+'*_flt.fits')
    print('***************')
    starttimes = []
    endtimes   = []
    imagetypes = []
    for flt in flts:
        starttimes.append(fits.getheader(flt,0)['EXPSTART'])
        endtimes.append(fits.getheader(flt,0)['EXPEND'])
        imagetypes.append(fits.getheader(flt,0)['IMAGETYP'])
        filename = os.path.split(flt)[-1]
        if not os.path.exists(wdir+filename):
            shutil.copy(flt, wdir)
            shutil.copy(flt.replace('_flt','_ima'), wdir)
            
    flts = glob.glob(wdir+'*_flt.fits')    
    ii = np.argsort(starttimes)
    for jj in range(len(flts)):
        print(starttimes[ii[jj]],endtimes[ii[jj]],(-starttimes[ii[jj]]+endtimes[ii[jj]])*daytosec,imagetypes[ii[jj]],flts[ii[jj]][-18:])

    sflts.append([flts[i] for i in ii])


# In[ ]:

# Choose which visit to work on

visit_index = 2
vsflts = sflts[visit_index]

# If in a hurry, shorten the list for faster analysis

#vsflts = vsflts[0:7]


# Create the wcs objects for the AD mosaic and flts 
# For the external flts use the wcs info in the AD 4th extension to update
# A WCS object created from the QL flts. This is needed because the WCS header of the flts copyied from QL
# may not have the up-to-date WCS that have been updated by TWREG/AD to prodcue the mosaic

mosaic = fits.open(mdir+'/F140W_Mosaic_WFC3_IR_drz.fits')
flt2mosaic = list(mosaic[4].data['FILENAME']) #List of ALL the flts that contribute to the AD mosaic
w_mosaic = WCS(mosaic[1].header)

w_vsflts = []
for vsflt in vsflts:
    if (fits.getheader(vsflt,0)['IMAGETYP'] != 'EXT'):
        w_vsflts.append(WCS(fits.getheader(vsflt,1)))
    elif (fits.getheader(vsflt,0)['IMAGETYP'] == 'EXT'):

        try:
            index_element = flt2mosaic.index(vsflt[-18:])
            w = WCS(fits.getheader(vsflt,1))
            
            crval1, crval2 = mosaic[4].data['CRVAL1'][index_element],mosaic[4].data['CRVAL2'][index_element]
            cd1_1, cd1_2   = mosaic[4].data['CD1_1'][index_element],mosaic[4].data['CD1_2'][index_element]
            cd2_1, cd2_2   = mosaic[4].data['CD2_1'][index_element],mosaic[4].data['CD2_2'][index_element]
            
            w.wcs.crval = [crval1,crval2]
            w.wcs.cd    = np.array([[cd1_1, cd1_2],[cd2_1,cd2_2]])
            
        except ValueError:
            print('Flt not in the mosaic!')
            assert False
        
        w_vsflts.append(w)


# Read in the pixel-area map

PAM = fits.getdata(pdir+'/Pixel_based/ir_wfc3_map.fits')


#From the current AD mosaic, get the sky values offsets that need to be used in the flts

MDRIZSKYs = []

for vsflt in vsflts:
    try:
        index_element = flt2mosaic.index(vsflt[-18:])
        MDRIZSKYs.append(mosaic[4].data['MDRIZSKY'][index_element])
    except ValueError:
        MDRIZSKYs.append(0.)
    print(vsflt,MDRIZSKYs[-1])


# In[ ]:

#Preprocess darks- determine corresponding superdark
def update_dark(vsflt):
    if 'N/A' in fits.getval(vsflt,'DARKFILE',ext=0):
        fits.setval(vsflt,'DARKCORR',value='PERFORM',ext=0)
        errors = bestrefs.BestrefsScript('BestrefsScript --update-bestrefs -s 1 -f ' + vsflt)()
        
def get_superdark_plane(darkfile, nsamp):
    darkfile = darkfile.replace('iref$','/grp/hst/cdbs/iref/')
    hdu = fits.open(darkfile)
    planes = []
    for k in range(nsamp):
        samp = 16-nsamp+k+1
        if (hdu[1].header['BUNIT'] == 'COUNTS/S'):
            planes.append(hdu['SCI',samp].data)
        else:
            if samp != 16:
                planes.append( (hdu['SCI',samp].data - hdu['SCI',16].data )/hdu['TIME',samp].header['PIXVALUE'])
            else:
                planes.append( 0.*(hdu['SCI',samp].data))
            
    hdu.close()
    return planes


# In[ ]:

# Create the numpy arrays containg the ima and flt data as well
# as the arrays of metadata.
# Also subtract the MDRIZSKY from the flt for a 1-to-1 comaprison with the AD mosaic
# Also bring the darks into e/s


ima_scis  = []
ima_errs  = []
ima_dqs   = []
ima_times = [] 
flts      = []
flts_dqs  = []
tendMJDs  = []
tstrMJDs  = []
imtyps    = []
nsamps    = []
sampseqs  = []

for vsflt,MDS in zip(vsflts,MDRIZSKYs):
    print('Appending '+vsflt+' to the datacube')

    ima = fits.open(vsflt.replace('_flt','_ima'))
    nsamps.append(ima[0].header['NSAMP'])
    sampseqs.append(ima[0].header['SAMP_SEQ'])

    hdr = fits.getheader(vsflt)
    if (hdr['IMAGETYP'] == 'DARK'):
        update_dark(vsflt)
        print(fits.getval(vsflt,'DARKFILE'))
        superdark_planes = get_superdark_plane(fits.getval(vsflt,'DARKFILE'),nsamps[-1])
    
    flt = fits.open(vsflt)
    
    
    if (flt[1].header['BUNIT'] == 'COUNTS/S'):
        fdt = flt['SCI'].data*flt[0].header['CCDGAIN']
    elif (flt[1].header['BUNIT'] == 'ELECTRONS/S'):
        fdt = flt['SCI'].data
    else:
        print('BUNITS not supported')
        assert False
    
    
    for k in range(nsamps[-1]):
        if (ima['SCI',k+1].header['BUNIT'] == 'COUNTS/S' and (hdr['IMAGETYP'] == 'DARK')):
            dark_sub_ima = ima['SCI',k+1].data-superdark_planes[k]
            imas = dark_sub_ima[5:-5,5:-5]*ima[0].header['CCDGAIN']
            imae = ima['ERR',k+1].data[5:-5,5:-5]*ima[0].header['CCDGAIN']
        elif (ima['SCI',k+1].header['BUNIT'] == 'COUNTS/S'):
            imas = ima['SCI',k+1].data[5:-5,5:-5]*ima[0].header['CCDGAIN']
            imae = ima['ERR',k+1].data[5:-5,5:-5]*ima[0].header['CCDGAIN']
            
        elif (ima['SCI',k+1].header['BUNIT'] == 'ELECTRONS/S'):
            imas = ima['SCI',k+1].data[5:-5,5:-5]
            imae = ima['ERR',k+1].data[5:-5,5:-5]
        else:
            print('BUNITS not supported')
            assert False
            
        ima_scis.append(imas)
        ima_errs.append(imae)
        ima_dqs.append(ima['DQ',k+1].data[5:-5,5:-5])
        ima_times.append(ima['TIME',k+1].header['PIXVALUE'])
   
    
    tendMJDs.append(flt[0].header['EXPEND'])
    tstrMJDs.append(flt[0].header['EXPSTART'])
    imtyps.append(flt[0].header['IMAGETYP'])
    flts.append(fdt - MDS)
    flts_dqs.append(flt['DQ'].data)
    
    flt.close()
    ima.close()
    

print('Done1')
ima_scis  = np.asarray(ima_scis)
print('Done2')
ima_errs  = np.asarray(ima_errs)
print('Done3')
ima_times = np.asarray(ima_times)
print('Done4')
ima_dqs = np.asarray(ima_dqs)
print('Done5')
flts      = np.asarray(flts)
print('Done6')
flts_dqs = np.asarray(flts_dqs)
print('Done7')
tendMJDs  = np.asarray(tendMJDs)
print('Done8')
tstrMJDs  = np.asarray(tstrMJDs)
print('Done9')
imtyps    = np.asarray(imtyps)
print('Done10')
nsamps    = np.asarray(nsamps)
print('Done11')


# In[ ]:

#Define the stimuli e-/s level to identify the ramps

lev_u = np.inf
lev_d = 4e4

# Define the pixel grid (to trasform indices in x,y positions)
xgrid,ygrid = np.meshgrid( np.arange(fits.getdata(vsflts[0],1).shape[1]) ,np.arange(fits.getdata(vsflts[0],1).shape[0]))
dxgrid,dygrid = np.meshgrid( np.arange(mosaic[1].data.shape[1]) ,np.arange(mosaic[1].data.shape[0]))

drz_fin = np.isfinite(mosaic[1].data)

msky_d = np.nanmean(sigmaclip(mosaic[1].data[drz_fin],2.,2.)[0])
ssky_d = np.nanstd(sigmaclip(mosaic[1].data[drz_fin],2.,2.)[0])
mask_sky_contam = (mosaic[1].data <msky_d+3*ssky_d) & (mosaic[1].data >msky_d-3*ssky_d) & drz_fin

skyrad_o = 12
skyrad_i = 3
lookback = None
psf = 0.2
numcores = 8


# In[ ]:

#before running the big step perform garbage collection
gc.collect()

df = pd.DataFrame()
cols = ['Stim','EXPTIME_stim','xpix','ypix','tfromstim','deltat','Read index','NSAMP','meancurr','stdvcurr','Ind_stim','Ind_pers','Stim_type','Pers_type','DQ_stim','DQ_pers']

mypool = Pool(numcores)

# Parallelized version
for istim,stim in enumerate(vsflts[:-1]):

    print('**********************')
    print('Doing: ',stim)

    if imtyps[istim] == 'EXT':
    
        icount    = find_ramps(istim,flts,lev_u,lev_d,lb=lookback,PAM=PAM,psf=psf)
    
        nz     = np.nonzero(icount)    
        nnexts = icount[nz]
    
        nlines = 0
        for nnext in nnexts:
            nlines = nlines+np.sum((nsamps-1)[istim+1:istim+1+nnext])

        print('Number of entries: ',nlines)
        modulus = np.trunc(nlines/10)
    
        if (nlines > 0) :
            flt_big_index = []
            for nz0,nz1,nnext in list(zip(nz[0],nz[1],nnexts)):
                prod = product([nz0], [nz1], range(1,nnext+1,1),[istim])
                flt_big_index += list(prod)
    
            derp = mypool.starmap(get_sky_and_indices, flt_big_index)
            ima_big_index = []
            for block in derp:
                ima_big_index += block
        
            biglist = mypool.starmap(get_pixel_values,zip(ima_big_index, [istim]*nlines, [PAM]*nlines))
            df = df.append(pd.DataFrame(biglist,columns=cols),ignore_index=True)
 



# Rearrange the dataframe to save only non-redundant info

print('Creating df2')
df2 = df.set_index(['xpix', 'ypix','Ind_stim','Stim','EXPTIME_stim','Stim_type','DQ_stim'])
print('Done')



iuniq  = df2.index.unique()
df2['Uniq_multiindex'] = np.empty(len(df2),dtype=np.int_)

nuniq = len(iuniq)
print('Number of points:',len(df))
print('Number of unique ramps:', nuniq)

for i,ind in enumerate(iuniq):
    if (i%500 == 0):
        print(i,' out of', nuniq)
    df2.loc[ind,'Uniq_multiindex'] = i


# In[ ]:

#Make sure that the data types are set to more space-efficient ones

print('Recasting up df2')
df2['Ind_pers'] = df2['Ind_pers'].astype(np.uint8)
df2['Read index'] = df2['Read index'].astype(np.uint8)
df2['NSAMP'] = df2['NSAMP'].astype(np.uint8)
df2[['tfromstim','deltat','meancurr','stdvcurr']] = df2[['tfromstim','deltat','meancurr','stdvcurr']].astype(np.float32)
df2['Uniq_multiindex'] = df2['Uniq_multiindex'].astype(np.uint32)
print('Done')

# In[ ]:

# The dataframe mapping the uniqe ramps indices
print('Defining the lookup dataframe')
df_lookup = df2[['Uniq_multiindex']].copy().drop_duplicates()
print('Done')

# In[ ]:

# The dataframe with persistence values

print('Defining the values dataframe')
df_values=pd.DataFrame()
df_values['tfromstim']       = df2['tfromstim'].values
df_values['deltat']          = df2['deltat'].values
df_values['Read index']      = df2['Read index'].values
df_values['meancurr']        = df2['meancurr'].values
df_values['stdvcurr']        = df2['stdvcurr'].values
df_values['NSAMP']           = df2['NSAMP'].values
df_values['Ind_pers']        = df2['Ind_pers'].values
df_values['Pers_type']       = df2['Pers_type'].values 
df_values['DQ_pers']         = df2['DQ_pers'].values 
df_values['Uniq_multiindex'] = df2['Uniq_multiindex'].values 
print('Done')


print('Saving')
df_values.to_hdf(sdir+'DF.h5', 'Visit'+'{:0>2}'.format(str(visit_index+1))+'_values', mode='a',format = 't')
df_lookup.to_hdf(sdir+'DF.h5', 'Visit'+'{:0>2}'.format(str(visit_index+1))+'_lookup', mode='a',format = 't')
print('Done')

