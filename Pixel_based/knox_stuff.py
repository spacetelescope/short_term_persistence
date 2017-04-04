'''
This module contains some of Knox's persistence pipeline routines,
modified to work for the short time persistence project
(hence the suffix _spv, short-time persistence version)
'''

import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

def get_persistence_spv(model,exp=300.,dt=1000.):
    '''
    Calculate the persistence curve using the tabulated A_gamma model

    where:

    exp    the exposure of the stimulus image
    dt    the time since the stimulus image was taken
    models    the file containing times and links to the persitance
        curves

    We need to interpolate both for the actual stimulus and
    for the fact that we don't have a grid of exposures.

    This just returns the persistence curve (on a specific grid) for a given exposure
    time exp and delta time dt.

    Note:

    Although this was initially developed for a phenomenalogical model where an 
    amplitude and a power law index had be calculated but fitting observatonal data, 
    any model with includes a power law decay can be case in the from of the so-called
    A-gama model, including a fermi-type model.

    History:

    170331    gennaro This needs the model in input as dictionary

    140630    ksl    Added a variable models to allow one to read files in any directory
            of interest

    '''


    
    # Now we need to interpolate so we have a single model given
    # an exposure time
    i=0
    while i<len(model['exp']) and model['exp'][i]<exp:
        i=i+1

    # print 'get_persitence:',i,len(model_exp),model_exp[i],exp

    if i==0:
        persist=model['a'][0]*(dt/1000.)**-model['g'][0]
    elif i==len(model['exp']):
        i=i-1
        persist=model['a'][i]*(dt/1000.)**-model['g'][i]
    else:
        frac=(exp-model['exp'][i-1])/(model['exp'][i]-model['exp'][i-1])
        persist1=model['a'][i-1]*(dt/1000.)**-model['g'][i-1]
        persist2=model['a'][i]*(dt/1000.)**-model['g'][i]
        persist=(1.-frac)*persist1+frac*persist2

    return persist


def read_models_spv():
    '''
    Read in the fits models for which can be expressed
    in terms of an amplitude A and a power law decay.
    

    Notes:

    The interpolated Fermi models are also read in 
    via this procedure.


    History

    170331  gennaro  made it return a dictionary and removed the test to see whether the function has been called
                     also: this only read the a-gama models by choice (and their path is hardcoded)
    140803    ksl    Coded
    140805    ksl    Replaced older read_models routine

    '''

    try:
        xpath=os.environ['PERCAL']
    except KeyError:
        print('Error: subtract_persist.locate_file: Environment variable PERCAL not defined and %s not in local directory or PerCal subdirectory')
        return ''

    models_file = xpath+'/a_gamma.fits'
 
    try:
        x=fits.open(models_file)
    except IOError:
        print('read_models: file %s does not appear to exist' % models)
        return 'NOK'

    i=1
    model_exp=[]
    model_a=[]
    model_stim=[]
    model_g=[]
    while i<len(x):
        model_exp.append(x[i].header['exp'])
        tabdata=x[i].data

        one_stim=tabdata['stim']
        model_stim.append(one_stim)

        one_a=tabdata['a']
        model_a.append(one_a)

        one_gamma=tabdata['gamma']
        model_g.append(one_gamma)
        i=i+1

    model_stim=np.array(model_stim[0]) # Use only the first row for the stimulus
    model_a=np.array(model_a)
    model_g=np.array(model_g)
    model_exp=np.array(model_exp)

    return  {'stim':model_stim,
             'a':model_a,
             'g':model_g,
             'exp':model_exp,
             'file':models_file
             }




def make_persistence_image_spv(x,model,exptime=500,dt=300):
    '''
    Make the persistence image for the A gamma model, given 
    
    an image array x,
    an exposure time for the stimulus image exptime.
    a time since the end of the (last) stimulus image
    a fits table that contains the spatially averaged persistence model.

    Note that this routine calls get_persistenee, which returns a persistence curve
    for a particular exptime and dt. The persistence curve is on a fixed grid. Here we
    use scipy.inter1d to produce the persisence image.

    History:

    170331    gennaro This needs the model in input as dictionary


    '''

    persist_curve=get_persistence_spv(model,exptime,dt)

    # Now we need to interpolate this curve
    f=interp1d(model['stim'],persist_curve,fill_value=0,bounds_error=False)

    persist=f(x)

    return persist
