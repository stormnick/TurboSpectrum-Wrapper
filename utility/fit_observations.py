import numpy as np
from sys import argv, exit
import os
import glob
import pickle
from scipy import interpolate
from observations import readSpectrumTSwrapper, spectrum, read_observations
from convolve_JG import conv_res, conv_macroturbulence, conv_rotation
from multiprocessing import Pool
from scipy.optimize import curve_fit, fmin_bfgs
from PayneModule import restore, restoreFromNormLabels, readNN
import sys
import cProfile
import pstats
import time

def callNN(wavelength, obsSpec, NN, p0, freeLabels, setLabels, quite=True):
    """
     To ensure the best convergence this function needs to be called on normalised labels (in p0)
     maybe it would work withput normalisation? it would make the code so much nicer
    """
  #  setLabels[i] = (setLabels[i] - norm['min'][i] ) / ( norm['max'][i] - norm['min'][i] ) - 0.5    

    labels = setLabels.copy()
    labels[freeLabels] = p0
    #labels[-1] = np.random.random()*50
    #labels[-2] = np.random.random()*5

    Vrot = labels[-1] 
    Vmac = labels[-2] 

    #labels[~freeLabels][1:-2] = labels[~freeLabels][1:-2] + labels[3] # feh
    flux = restore(wavelength, NN, labels[:-2]) 
    fwhm = (np.mean(wavelength) / obsSpec.R) * 1e3
    #wavelength, flux = conv_res(wavelength, flux, fwhm)
    #wavelength, flux = conv_macroturbulence(wavelength, flux, Vmac)
#    wavelength, flux = conv_rotation(wavelength, flux, Vrot)
    spec = spectrum(wavelength, flux, res = np.inf)
    
    spec.convolve_resolution( obsSpec.R )
    spec.convolve_macroturbulence( labels[-2] , quite=quite)
    spec.convolve_rotation( labels[-1], quite=quite)
    chi2 = np.sqrt(np.sum(obsSpec.flux - flux)**2)
    print(labels, chi2)
    #return flux
    return spec.flux

def fitToNeuralNetwork(obsSpec, NN, prior = None, quite = True):

    freeLabels = np.full(len(NN['labelsKeys'])+2, True)
    setLabels = np.full(len(NN['labelsKeys'])+2, 0.0)
    if isinstance(prior, type(None)):
        pass
    else:
        if len(prior)  < len(NN['labelsKeys']) + 2:
            for i, l in enumerate( np.hstack( (NN['labelsKeys'], ['vmac', 'vrot']))):
                l = l.lower()
                if l in prior:
                    freeLabels[i] = False
                    setLabels[i] = prior[l]
        elif prior.keys() != NN['labelsKeys']:
            print(f"Provided prior on the labels {prior} does not match labels ANN was trained on: {NN['labelsKeys']}")
            exit()

    """
    Initialise the labels if not provided
    Extra dimension is for macro-turbulence and rotation
    """

    initLabels = []
    norm = {'min' : np.hstack( [NN['x_min'], [0, 0]] ), 'max': np.hstack( [NN['x_max'], [100, 10]] ) }
    for i, l in enumerate( np.hstack( (NN['labelsKeys'], ['vmac', 'vrot']))):
        if freeLabels[i]:
            initLabels.append( np.mean( (norm['min'][i], norm['max'][i] ) )  + prior['feh'])
    """
    Resampled (and cut if needed)  observed spectrum to the wavelength points 
    provided in the ANN
    """
    w_new = NN['wvl'][np.logical_and( NN['wvl']>min(obsSpec.lam ), NN['wvl']<max(obsSpec.lam) )]
    obsSpec.flux = np.interp(w_new,  obsSpec.lam, obsSpec.flux)
    obsSpec.lam = w_new

    """
    Lambda function for fitting 
    """
    fitFunc = lambda wavelength, *labels : callNN(
                                                wavelength, obsSpec,
                                                NN, labels, freeLabels, setLabels, quite = quite
                                                )
    bounds = ( norm['min'][freeLabels], norm['max'][freeLabels] )
    popt,_ = curve_fit(
                    fitFunc, obsSpec.lam, \
                    obsSpec.flux, p0=initLabels,\
                    bounds = bounds
                    )
    " restore normalised labels "
    setLabels[freeLabels] = popt
  #  setLabels = (setLabels+ 0.5)*( norm['max'] - norm['min'] ) + norm['min']
    flux =  restore(obsSpec.lam, NN, setLabels[:-2])
    wavelength = obsSpec.lam
    wavelength, flux = conv_res(wavelength, flux, 1e3 * np.mean(wavelength)/obsSpec.R )
    wavelength, flux = conv_macroturbulence(wavelength, flux, setLabels[-2])
#    wavelength, flux = conv_rotation(wavelength, flux, setLabels[-1])

    #spec = spectrum(
    #                obsSpec.lam,
    #                restore(obsSpec.lam, NN, setLabels[:-2]), res = np.inf
    #                )
    #spec.convolve_resolution(obsSpec.R)
    #spec.convolve_macroturbulence( setLabels[-2] , quite=quite)
    #spec.convolve_rotation( setLabels[-1], quite=quite)

    np.savetxt(f"./{obsSpec.ID}_modelFlux.dat", np.vstack([obsSpec.lam, obsSpec.flux, flux]).T )
    chi2 = np.sqrt(np.sum(obsSpec.flux - flux)**2)
    return setLabels, chi2


def internalAccuracyFitting():
    if len(argv) < 4:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra or payne NN> <path to observed spectra> fit-for-key e.g. 'mg'")
        exit()
    "Fit using Payne neural network"
    nnPath = argv[1]
    NNs = glob.glob(nnPath)
    print(f"found {len(NNs)} ANNs")
    obsPath = argv[2]
    specList = glob.glob(obsPath)
    print(f"found {len(specList)} observed spectra")
  
    solveFor = None
    if len (argv)>3:
        solveFor = argv[3]
        print(f"Solving for {solveFor}...")
    
    for nnPath in NNs:
        NN = readNN(nnPath)
        NNid = nnPath.split('/')[-1].replace('.npz', '').strip() 
        if not isinstance(solveFor, type(None)):
            if solveFor not in NN['labelsKeys']:
                print(f"No key {solveFor} in requested NN {nnPath}")
                exit()

        out = {'file':[], 'chi2':[], 'vmac':[], 'vrot':[], f"diff_{solveFor}":[]}
        with open(f"./fittingResults_{NNid}_fitFor{solveFor}.dat", 'w') as LogResults:
            LogResults.write( "#" + '\t'.join(NN['labelsKeys']) + ' Vmac    Vrot  chi2\n' )
            for obsSpecPath in specList:
                print(obsSpecPath)
                out['file'].append(obsSpecPath)
                obsSpec = readSpectrumTSwrapper(obsSpecPath)
                if solveFor not in obsSpec.__dict__.keys():
                    print(f"No key {solveFor} in spectrum {obsSpecPath}")
                    exit()
                obsSpec.convolve_resolution(NN['res'])
                # resolution is considered constant, therefore FWHM will be bigger for smaller wavelngth range
                # be careful with the resolution convolution for both observations and ANN restored fluxes
                obsSpec.cut([min(NN['wvl']), max(NN['wvl'])] )
                #obsSpec.cut([5182, 5185] )
                obsSpec.ID = obsSpecPath.split('/')[-1].replace('.dat', '')
    
                prior = None
                if not isinstance(solveFor, type(None)):
                    prior = {}
                    for l in NN['labelsKeys']:
                        if l.lower() != solveFor.lower():
                            prior[l.lower()] = obsSpec.__dict__[l]
                    #prior['vmac'] = 0
                    #prior['vrot'] = 0
    
                labelsFit, bestFitChi2 = fitToNeuralNetwork(obsSpec, NN, prior = prior, quite=True)
                for i, l in enumerate(NN['labelsKeys']):
                    if l not in out:
                        out.update({l:[]})
                    out[l].append(labelsFit[i])
                out[f"diff_{solveFor}"].append( obsSpec.__dict__[solveFor] - out[solveFor][-1] )
                d =  out[f"diff_{solveFor}"][-1]
                print(f"Difference in {solveFor} is {d:.3f}")
                out['vmac'].append(labelsFit[-2])
                out['vrot'].append(labelsFit[-1])
                out['chi2'].append(bestFitChi2)
                   
                LogResults.write( f"{obsSpec.ID} " + '\t'.join(f"{l:.3f}" for l in labelsFit) + f"{bestFitChi2 : .3f}\n")
        for k in out.keys():
            out[k] = np.array(out[k])
        with open(f'./fittingResults_{NNid}_fitFor{solveFor}.pkl', 'wb') as f:
            pickle.dump(out, f)

if __name__ == '__main__':
    if len(argv) < 3:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra or payne NN> <path to observed spectra> ")
        exit()
    "Fit using Payne neural network"
    nnPath = argv[1]
    NNs = glob.glob(nnPath)
    print(f"found {len(NNs)} ANNs")
    obsPath = argv[2]
    #specList = glob.glob(obsPath)
    specList = glob.glob('./uves_test/ksi_Hya/uvu_11325994-3151279_520.0_25_reNormCont.asc')
    print(f"found {len(specList)} observed spectra")
    
    Jofre = {
        'HD107328': {
            'key':'12202074+0318445',
            'teff':4496, 
            'logg':2.09,
            'feh':-0.38,
            'vturb':1.65},
        'HD220009': {
            'key':'23202065+0522519',
            'teff':4275,
            'logg':1.47,
            'feh':-0.79, 
            'vturb':1.49},
        'ksi_Hya': {
            'key':'11325994-3151279',
            'teff':5044, 
            'logg':2.87,
            'feh':0.11,
            'vturb':1.40},
        'mu_Leo': {
            'key':'09524561+2600243',
            'teff':4474, 
            'logg':2.51,
            'feh':0.20, 
            'vturb':1.28}, 
         'HD122563': {
            'key':'14023168+0941090',
            'teff':4636, 
            'logg':1.42,
            'feh':-2.52,
            'vturb':1.92}
    }
    
    with open('./fittingResults.dat', 'w') as fOut:
        for nnPath in NNs:
            NN = readNN(nnPath)
            fOut.write('#  ' + '   '.join(f"{k}" for k in NN['labelsKeys']) + ' Vmac  Vrot  chi  SNR\n' )
            lim = [5300, 5600]
            #lim = [5526, 5530]
            
            #profiler = cProfile.Profile()
            #profiler.enable()
            for sp in specList:
                starID = sp.split('/')[-2]
                print(starID)
                w, f, snr = np.loadtxt(sp, unpack=True, usecols=(0,1,2))
                snr = snr[0]
                if  lim[0] >= min(w) and  lim[1] <= max(w):
                    print(sp)
                    specObs = spectrum(w, f, res=47000)
                    #specObs.convolve_resolution(18000)
                    specObs.ID = sp.split('/')[-1].replace('.asc', '')
                    specObs.cut(lim)
                    #prior = None
                    prior = { k: Jofre[starID][k] for k in ['teff', 'logg', 'feh', 'vturb']}
                    print(prior)
                    labelsFit, bestFitChi2 = fitToNeuralNetwork(specObs, NN, prior = prior, quite=True)
                    for i, k in  enumerate(NN['labelsKeys']):
                        print(f"{k} = {labelsFit[i]:.2f}")
                    print(f"Vmac = {labelsFit[-2]:.2f}")
                    print(f"Vrot = {labelsFit[-1]:.2f}")
                    fOut.write(f"{specObs.ID}  " +  '   '.join( f"{l:.3f}" for l in labelsFit  ) + f"  {bestFitChi2:.3f}   {snr:.1f}"  + '\n')
            #profiler.disable()
            #stats = pstats.Stats(profiler).sort_stats('cumulative')
            #stats.print_stats()
