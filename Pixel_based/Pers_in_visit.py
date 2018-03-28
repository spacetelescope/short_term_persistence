
# coding: utf-8


# In[1]:

from astropy.io import fits
import glob, os, shutil, pickle, bz2, gc, sys
import matplotlib.pyplot as plt
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


# In[3]:

# The project dir 
pdir = '/user/gennaro/Functional_work/WFC3_persistence/py_progs/short_term_persistence/'

#The mosaic dir
mdir = pdir+'/Mosaic_hi_res_folder/'

#The dir to save/load the Persistence curves dataframes
sdir = pdir+'/PD_dataframes_dir/'

# In[4]:

# conversion factor from days to seconds
daytosec = 24.*3600.


# In[6]:

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
        earlierdata = flts[i,:,:] * (tendMJDs[i]-tstrMJDs[i])*daytosec 
        if PAM is not None:
            earlierdata *= PAM

        if (imtyps[i] == 'EXT'):
            istimgood = istimgood & (earlierdata < psf*stimdata) 
            
    print('Pixels with really right stimuli:',np.sum(istimgood) )
    
    
    icount    = np.zeros_like(stimdata,dtype=np.int_)
    iprev     = istimgood
    for i in range(istim+1,len(imtyps),1):
    
        persdata = flts[i,:,:] * (tendMJDs[i]-tstrMJDs[i])*daytosec
        if PAM is not None:
            persdata *= PAM
    
        if (imtyps[i] == 'EXT'):
            pfinite = persdata[np.isfinite(persdata)]
            msky = np.nanmean(sigmaclip(pfinite,2.5,2.5)[0])
            ssky = np.nanstd(sigmaclip(pfinite,2.5,2.5)[0])
            print('Mean',msky)
            print('Stdv',ssky)
            
            iskycurr = (persdata <msky+3.*ssky) & (persdata >msky-2*ssky)   
#            iskycurr = (persdata < (25*(tendMJDs[istim]-tstrMJDs[istim])*daytosec)) & (persdata >(5*(tendMJDs[istim]-tstrMJDs[istim])*daytosec) ) 
#            print(msky,ssky)
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


# In[7]:

# Dedfine a function to get the sky value in the cureent flt pixel, but measured form the AD mosaic

def getskyfrommosaic(wcsAD, wcsFLT, x, y, dxgrid, dygrid,mask_sky_contam,mosaic):

    coords = wcsAD.all_world2pix(wcsFLT.all_pix2world(np.array([[x,y]],dtype=np.float_),0),0) 
    dx = coords[0,0]
    dy = coords[0,1]
    
    dst = np.sqrt((dxgrid-dx)**2 + (dygrid-dy)**2)
    msk = (dst<skyrad_o) & (dst > skyrad_i) & mask_sky_contam 
    skyarr = mosaic[1].data[msk]
    cskyarr,l,u = sigmaclip(skyarr[np.isfinite(skyarr)],2.5,2.5)
    return np.nanmean(cskyarr)

# Similar but for getting background values from the same image

def getlocalbackground (x, y, xgrid, ygrid, image):

    dst = np.sqrt((xgrid-x)**2 + (ygrid-y)**2)
    msk = (dst<skyrad_o) & (dst > skyrad_i)
    skyarr = image[msk]
    cskyarr,l,u = sigmaclip(skyarr[np.isfinite(skyarr)],scsky,scsky)
    return np.nanmean(cskyarr)

#msky = np.median(sigmaclip(meancurr_ima,1.75,1.75)[0])
#def get_dark_rate(x, y, istim, j)


# In[8]:

def get_sky_and_indices(nz0, nz1, j, istim):
    # Function to get the sky value, as well as calculate the indices needed in the large data cube of IMA reads
       
    #Get the sky from the drizzled image
    imtyp = imtyps[istim+j]

    if(imtyp == 'EXT'):
#        skyhere = getskyfrommosaic(w_mosaic, w_vsflts[istim+j], nz1, nz0, dxgrid, dygrid, mask_sky_contam,mosaic)
        skyhere = getlocalbackground(nz1, nz0, xgrid, ygrid, flts[istim+j,:,:])
    elif(imtyp == 'DARK'):
        skyhere = 0.0 #Do not subtract any sky, as the superdark is subtracted from IMA already
    else:
        print('Wrong image type')
        assert False
    
    offset = ( tstrMJDs[istim+j] - tendMJDs[istim] )*daytosec
    ioff = np.sum(nsamps[0:istim+j])
    nsamp = nsamps[istim+j]
    
    k_product = product([nz0], [nz1], [skyhere], [offset], [ioff], [nsamp], [j], range(nsamp-1))
    return list(k_product)


# In[9]:

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
        meancurr = (meancurr-skyhere)*pflats[pflats_names[istim+j]][nz1,nz0]
        stdvcurr *= pflats[pflats_names[istim+j]][nz1,nz0]
    
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


# In[10]:

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

#vsflts = vsflts[0:len(vsflts)-16]

# If doing darks only (visit0 has 7 ext, visit1 has 5, visit2 has 3):
# Just keep the 2 last ext, the last one is the useful one, the second last is needed to exclude that the pixel
# had previous large stimuli

startflt = 5 - visit_index*2  
vsflts = vsflts[startflt:]
        
# Read the mosaic file

mosaic = fits.open(mdir+'/F140W_Mosaic_WFC3_IR_drz.fits')


# In[ ]:

# Create the wcs objects for the AD mosaic and flts 
# For the external flts use the wcs info in the AD 4th extension to update
# A WCS object created from the QL flts. This is needed because the WCS header of the flts copyied from QL
# may not have the up-to-date WCS that have been updated by TWREG/AD to prodcue the mosaic

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


# In[ ]:

# Read in the pixel-area map

PAM = fits.getdata(pdir+'/Pixel_based/ir_wfc3_map.fits')


#Get the flt files to de-flatten the processed IMAs

def get_flat(ext_file):
    flatfile = fits.getval(ext_file,'PFLTFILE').replace('iref$','/grp/hst/cdbs/iref/')
    return flatfile

pflats_names = []
for flt in vsflts:
    if fits.getheader(flt)['IMAGETYP'] == 'EXT':
        pflats_names.append(get_flat(flt))
    else:
        pflats_names.append(None)
        

pflats_unique = []
for i in pflats_names:
    if i is not(None):
        if i not in pflats_unique:
            pflats_unique.append(i)

pflats = {}           
for f in pflats_unique:
    pflats[f] = fits.open(f)[1].data[5:-5,5:-5]
    



# In[ ]:

#lowlim, bs = 0., 0.2

DRZdt = mosaic[1].data
DRZ_avg_sky = np.nanmean(sigmaclip(DRZdt[np.isfinite(DRZdt)],2.5,2.5)[0])

# In[ ]:

#From the current AD mosaic, get the sky values offsets that need to be used in the flts

flt2mosaic = list(mosaic[4].data['FILENAME']) #List of ALL the flts that contribute to the AD mosaic
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

#In order to match the image coordinates with the astropy convention,
#the first coordinate in the following numpy array is the y, the second the x 

gn_im = np.ones([1014,1014])

hd = fits.open(vsflts[0])

gn_im[0:507,0:507] = hd[0].header['ATODGNA'] #2.252  #q1
gn_im[507:,0:507]  = hd[0].header['ATODGNB'] #2.203  #q2
gn_im[507:,507:]   = hd[0].header['ATODGNC'] #2.188  #q3
gn_im[0:507,507:]  = hd[0].header['ATODGND'] #2.265  #q4

print(fits.open(vsflts[0])[0].header['*TODGN*'])


# In[ ]:

# Create the numpy arrays containg the ima and flt data as well
# as the arrays of metadata.
# Also subtract the MDRIZSKY from the flt for a 1-to-1 comaprison with the AD mosaic
# Also bring the darks into e/s


ima_scis  = []
ima_errs  = []
ima_dqs   = []
ima_times = [] 
ima_avg_sky  = []

flts      = []
flts_dqs  = []
flts_avg_sky  = []


tendMJDs  = []
tstrMJDs  = []
imtyps    = []
nsamps    = []
sampseqs  = []

ima_offset = 0

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
            imas = dark_sub_ima[5:-5,5:-5]*gn_im
            imae = ima['ERR',k+1].data[5:-5,5:-5]*gn_im

        elif (ima['SCI',k+1].header['BUNIT'] == 'COUNTS/S'):
            imas = ima['SCI',k+1].data[5:-5,5:-5]*gn_im
            imae = ima['ERR',k+1].data[5:-5,5:-5]*gn_im
            
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

#Avergae sky computation
    if (hdr['IMAGETYP'] == 'EXT'):
        afs = np.nanmean(sigmaclip(fdt[np.isfinite(fdt)],2.5,2.5)[0])
        flts_avg_sky.append(afs)
    else:
        flts_avg_sky.append(0.)
    
    
    for k in range(nsamps[-1]):    
        if (hdr['IMAGETYP'] == 'EXT'):
            if k == (nsamps[-1] - 1):
                ima_avg_sky.append(0.)
            else:
                meancurr_ima = (ima_scis[ima_offset+k]*ima_times[ima_offset+k] - ima_scis[ima_offset+k+1]*ima_times[ima_offset+k+1])/(ima_times[ima_offset+k] - ima_times[ima_offset+k+1])
                ais = np.nanmean(sigmaclip(meancurr_ima[np.isfinite(meancurr_ima)],2.5,2.5)[0])
                ima_avg_sky.append(ais)
        else:
            ima_avg_sky.append(0.)
        
    ima_offset += nsamps[-1]

    tendMJDs.append(flt[0].header['EXPEND'])
    tstrMJDs.append(flt[0].header['EXPSTART'])
    imtyps.append(flt[0].header['IMAGETYP'])
    flts.append(fdt)
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
flts_avg_sky =  np.asarray(flts_avg_sky)
print('Done8')
tendMJDs  = np.asarray(tendMJDs)
print('Done9')
tstrMJDs  = np.asarray(tstrMJDs)
print('Done10')
imtyps    = np.asarray(imtyps)
print('Done11')
nsamps    = np.asarray(nsamps)
print('Done12')
ima_avg_sky   = np.asarray(ima_avg_sky)
print('Done13')


# In[ ]:

#Define the stimuli e-/s level to identify the ramps

lev_u = np.inf
lev_d = 1e4

# Define the pixel grid (to trasform indices in x,y positions)
xgrid,ygrid = np.meshgrid( np.arange(fits.getdata(vsflts[0],1).shape[1]) ,np.arange(fits.getdata(vsflts[0],1).shape[0]))
dxgrid,dygrid = np.meshgrid( np.arange(mosaic[1].data.shape[1]) ,np.arange(mosaic[1].data.shape[0]))

drz_finite = np.isfinite(mosaic[1].data)

msky_d = np.nanmean(sigmaclip(mosaic[1].data[drz_finite],2.5,2.5)[0])
ssky_d = np.nanstd(sigmaclip(mosaic[1].data[drz_finite],2.5,2.5)[0])
mask_sky_contam = (mosaic[1].data <msky_d+1*ssky_d) & (mosaic[1].data >msky_d-2*ssky_d) & drz_finite

scsky = 2.1
skyrad_o = 15
skyrad_i = 6
lookback = None
psf = 0.2
numcores = 12

#namesuff = '_sig'+'{:05.2f}'.format(scsky) + '_ri' + '{:05.2f}'.format(skyrad_i) + '_ro' + '{:05.2f}'.format(skyrad_o)
namesuff = '_dark_only'

print('Doing: ',namesuff)
sys.stdout.flush()

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
    sys.stdout.flush()
    
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
 


# In[ ]:

df.head()


# In[ ]:

# Rearrange the dataframe to save only non-redundant info

df2 = df.set_index(['xpix', 'ypix','Ind_stim','Stim','EXPTIME_stim','Stim_type','DQ_stim'])
df2.head()


# In[ ]:

iuniq  = df2.index.unique()
df2['Uniq_multiindex'] = np.empty(len(df2),dtype=np.int_)

print('Number of points:',len(df))
print('Number of unique ramps:',len(iuniq))

for i,ind in enumerate(iuniq):
    if (i%1000 == 0):
        print(i)
        sys.stdout.flush()
    df2.loc[ind,'Uniq_multiindex'] = i


# In[ ]:

#Make sure that the data types are set to more space-efficient ones

df2['Ind_pers'] = df2['Ind_pers'].astype(np.uint8)
df2['Read index'] = df2['Read index'].astype(np.uint8)
df2['NSAMP'] = df2['NSAMP'].astype(np.uint8)
df2[['tfromstim','deltat','meancurr','stdvcurr']] = df2[['tfromstim','deltat','meancurr','stdvcurr']].astype(np.float32)
df2['Uniq_multiindex'] = df2['Uniq_multiindex'].astype(np.uint32)


# In[ ]:

# The dataframe mapping the uniqe ramps indices
df_lookup = df2[['Uniq_multiindex']].copy().drop_duplicates()
df_lookup.head()


# In[ ]:

# The dataframe with persistence values

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

df_values.head()


# In[ ]:

df_values.to_hdf(sdir+'DF'+namesuff+'.h5', 'Visit'+'{:0>2}'.format(str(visit_index+1))+'_values', mode='a',format = 't')
df_lookup.to_hdf(sdir+'DF'+namesuff+'.h5', 'Visit'+'{:0>2}'.format(str(visit_index+1))+'_lookup', mode='a',format = 't')


# In[ ]:



