import numpy as np
from sys import argv, exit
import os
import glob
import pickle
from scipy import interpolate
from .observations import readSpectrumTSwrapper, spectrum, read_observations, convolve_gauss
from multiprocessing import Pool
from scipy.optimize import curve_fit, fmin_bfgs
from .PayneModule import restore, restoreFromNormLabels, readNN
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
    Vbroad = labels[-1] 

    flux = restore(wavelength, NN, labels[:-1]) 
    if Vbroad > 0.0:
        flux = convolve_gauss(wavelength, flux, Vbroad, mode='broad')
    if NN['res'] < np.inf:
        flux = convolve_gauss(wavelength, flux, NN['res'], mode='res')
    #chi2 = np.sqrt(np.sum(obsSpec.flux - flux)**2)
    #print(labels, chi2)
    return flux

def fitToNeuralNetwork(obsSpec, NN, prior = None, quite = True):

    freeLabels = np.full(len(NN['labelsKeys'])+1, True)
    setLabels = np.full(len(NN['labelsKeys'])+1, 0.0)
    if isinstance(prior, type(None)):
        pass
    else:
        if len(prior)  < len(NN['labelsKeys']) + 1:
            for i, l in enumerate( np.hstack( (NN['labelsKeys'], ['vbroad']))):
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
    norm = {'min' : np.hstack( [NN['x_min'], [0]] ), 'max': np.hstack( [NN['x_max'], [100]] ) }
    for i, l in enumerate( np.hstack( (NN['labelsKeys'], ['vbroad']))):
        if freeLabels[i]:
            initLabels.append( np.mean( (norm['min'][i], norm['max'][i] ) ) )
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

    wavelength = obsSpec.lam
    flux =  restore(wavelength, NN, setLabels[:-1])
    if setLabels[-1] > 0.0: # vbroad
        flux = convolve_gauss(wavelength, flux, setLabels[-1], mode='broad')
    if NN['res'] < np.inf:
        flux = convolve_gauss(wavelength, flux, NN['res'], mode='res')
#    np.savetxt(f"./{obsSpec.ID}_modelFlux.dat", np.vstack([wavelength, flux, flux]).T )
    chi2 = np.sqrt(np.sum(obsSpec.flux - flux)**2)
    return setLabels, chi2


def internalAccuracyFitting(nnPath, specList, solveFor=None):
    print(f"Solving for {solveFor}...")

    print(f"found {len(specList)} observed spectra")
    
    NN = readNN(nnPath)
    NNid = nnPath.split('/')[-1].replace('.npz', '').strip() 
    if not isinstance(solveFor, type(None)):
        if solveFor not in NN['labelsKeys']:
            print(f"No key {solveFor} in requested NN {nnPath}")
            exit()

    out = {'file':[], 'chi2':[], 'vmac':[], 'vrot':[], f"diff_{solveFor}":[]}
    with open(f"./fittingResults_{NNid}_fitFor{solveFor}.dat", 'w') as LogResults:
        LogResults.write( "#" + '\t'.join(NN['labelsKeys']) + f' Vbroad chi2 {solveFor}_bestFit\n' )
        for obsSpecPath in specList:
            out['file'].append(obsSpecPath)
            obsSpec = readSpectrumTSwrapper(obsSpecPath)
            obsSpec.ID = obsSpecPath.split('/')[-1].replace('.dat', '')
            if solveFor not in obsSpec.__dict__.keys():
                print(f"No key {solveFor} in spectrum {obsSpecPath}")
                exit()

            obsSpec.cut([min(NN['wvl']), max(NN['wvl'])] )
            if np.isfinite(NN['res']):
                f = convolve_gauss(obsSpec.lam, obsSpec.flux, NN['res'], mode='res')
                obsSpec = spectrum(obsSpec.lam, f, res=NN['res'])

            prior = None
            if not isinstance(solveFor, type(None)):
                prior = {}
                for l in NN['labelsKeys']:
                    if l.lower() != solveFor.lower():
                        prior[l.lower()] = obsSpec.__dict__[l]
                prior['vbroad'] = 0.0

            labelsFit, bestFitChi2 = fitToNeuralNetwork(obsSpec, NN, prior = prior, quite=True)
            for i, l in enumerate(NN['labelsKeys']):
                if l not in out:
                    out.update({l:[]})
                out[l].append(labelsFit[i])
            out[f"diff_{solveFor}"].append( obsSpec.__dict__[solveFor] - out[solveFor][-1] )
            d =  out[f"diff_{solveFor}"][-1]
            print(f"Difference in {solveFor} is {d:.2f}")
            out['chi2'].append(bestFitChi2)
               
            LogResults.write( f"{obsSpec.ID} " + '\t'.join(f"{l:.3f}" for l in labelsFit) + f"{bestFitChi2 : .3f} {d:.3f}\n")
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
            lim = [5526, 5530]
            
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
