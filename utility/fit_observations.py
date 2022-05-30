import numpy as np
from sys import argv, exit
import os
import glob
import pickle
from scipy import interpolate
from observations import readSpectrumTSwrapper, spectrum, read_observations
from multiprocessing import Pool
from scipy.optimize import curve_fit, fmin_bfgs
import sys
sys.path.append('/home/semenova/codes/PayneNN')
from source.validationChecks import restore, readNN, restoreFromNormLabels

def readModelSpectralGrid(ListOfModelSpectra):
    "Assuming the same wavelength because there is too much going on with cutting the spectra and so on"
    spectraGrid = {
                  'path' : [],
                  'coordinates' : []
}
    for spFile in ListOfModelSpectra:
        spec = readSpectrumTSwrapper(spFile)
        if not isinstance(spec, type(None)):
            spectraGrid['path'].append( spFile )
            for lab in spec.labels:
                if lab not in spectraGrid:
                    spectraGrid.update( { lab : [] } )
                spectraGrid[lab].append(spec.__dict__[lab])
                if lab not in spectraGrid['coordinates']:
                    spectraGrid['coordinates'].append(lab)
    return spectraGrid

def gatherModelData(Path):
    pickledGridPath = Path.replace(Path.split('/')[-1], '') + 'gridCoordinates.pkl'
    if os.path.isfile(pickledGridPath) and os.path.getsize(pickledGridPath) > 0:
        with open(pickledGridPath, 'rb') as f:
            modelGrid = pickle.load(f)
    else:
        ncpu = 50
        specList = glob.glob(Path)
        print(f"reading model {len(specList)} spectra under {Path} ...")
        args = [ specList[i::ncpu] for i in range(ncpu)]
        modelGrid = { }

        with Pool(processes=ncpu) as pool:
            out = pool.map(readModelSpectralGrid, args )
        " Gather output from parallel processes "
        for grid in out:
            for k in grid:
                if not k in modelGrid:
                    modelGrid.update( { k : [] } )
                modelGrid[k].extend(grid[k])
        for k in modelGrid:
            modelGrid[k] = np.array( modelGrid[k] )
        modelGrid['coordinates'] = np.unique(modelGrid['coordinates'])
        with open(pickledGridPath, 'wb') as f:
            pickle.dump(modelGrid, f)

    return modelGrid

def getClosestModel(specObs, modelGrid, coords, labels):
    dist = 0
    for i,k in enumerate(coords):
        print(f"{k} = {labels[i]}")
        dist += ( modelGrid[k] - labels[i] )**2
    pos = np.where(dist == min(dist))[0]
    if len(pos) > 1:
       print(f"found {len(pos)} 'closest' spectra:")
       for p in pos:
           print(spectraGrid['path'][p].split('/')[-1])
    pos = pos[0]
    modSpec = readSpectrumTSwrapper( modelGrid['path'][pos] )
    modSpec.convolve_resolution(specObs.R)
    modSpec.convolve_macroturbulence(labels[-1])

    print(modelGrid['path'][pos].split('/')[-1])
    print()

    flux = np.interp(specObs.lam, modSpec.lam, modSpec.flux )
    chi2 = np.sqrt(np.sum( ( flux - specObs.flux ) ** 2 ))

    return chi2

def fitToModelGrid(obsSpec, modelGrid, fitForLabels = None):

    if isinstance(fitForLabels, type(None)):
        fitForLabels = modelGrid['coordinates']

    " Initialise the labels if not provided "
    initLabels = []
    for k in fitForLabels:
        print(k, np.mean( modelGrid[k] ))
        initLabels.append( np.mean( modelGrid[k] ) )
    " Add macro-turbulence "
    initLabels.append(0)
    print(initLabels)

    fitFunc = lambda label : getClosestModel(obsSpec, modelGrid, fitForLabels, label)
    popt = fmin_bfgs( fitFunc, initLabels)
    print(popt)
    return popt


def callNN(wavelength, obsSpec, NN, labels):
    #fileOut = "./iterFlux.dat"
    #if os.path.isfile(fileOut):
    #    data = np.loadtxt(fileOut)
    #else:
    #    data = None
    
    spec = spectrum(
                    wavelength,
                    restoreFromNormLabels(wavelength, NN, labels[:-2]), res = np.inf
                    )
    spec.convolve_resolution(obsSpec.R)
    spec.convolve_macroturbulence(labels[-2]*100)
    spec.convolve_rotation(labels[-1]*100)
    #if not isinstance(data, type(None)):
    #    data = np.vstack( ( data, spec.flux) )
    #else: data = spec.flux.copy()
    #np.savetxt(fileOut, data)
    return spec.flux

def fitToNeuralNetwork(obsSpec, NN, fitForLabels = None):

    if isinstance(fitForLabels, type(None)):
        fitForLabels = NN['labelsKeys']

    """
    Initialise the labels if not provided
    Extra dimension for macro-turbulence
    """

    initLabels = np.zeros(len(fitForLabels)+2)


    fitFunc = lambda wavelength, *labels : callNN(
                                                wavelength, obsSpec,
                                                NN, labels
                                                )
    popt,_ = curve_fit(
                    fitFunc, obsSpec.lam, \
                    obsSpec.flux, p0=initLabels,\
                    #bounds = (-0.5, 0.5)
                    )

    popt[:-2] = (popt[:-2] + 0.5)*( NN['x_max'] - NN['x_min'] ) + NN['x_min']
    popt[-2:] =  popt[-2:]*100
    for i in range(len(fitForLabels)):
        print(f" {fitForLabels[i]} = {popt[i]:.2f} ")
    print(f"Vmac = {popt[-2]:.3f}")
    print(f"Vrot = {popt[-1]:.3f}")
    spec = spectrum(
                    obsSpec.lam,
                    restore(obsSpec.lam, NN, popt[:-2]), res = np.inf
                    )
    spec.convolve_resolution(obsSpec.R)
    spec.convolve_macroturbulence(popt[-1])
    np.savetxt(f"./{obsSpec.ID}_modelFlux.dat", spec.flux)
    return popt

if __name__ == '__main__':
    if len(argv) < 4:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra or payne NN> <path to observed spectra> <mode (grid or payne)>")
        exit()

    if argv[3].lower().strip() == 'grid':
        "Fit to the grid"

        modelPath = argv[1]
        obsPath = argv[2]

        modelGrid = gatherModelData(modelPath)

        fitForLabels = ['teff', 'logg', 'feh', 'vturb']
        for obsSpecPath in glob.glob(obsPath):
            w, f = read_observations(obsSpecPath, format = 'ascii')
            obsSpec = spectrum(
                            w, f,\
                            res = np.inf
                            )
            obsSpec.convolve_resolution(24)
            labelsFit = fitToModelGrid(obsSpec, modelGrid, fitForLabels )
    elif argv[3].lower().strip() == 'payne':
        "Fit using Payne neural network"
        nnPath = argv[1]
        obsPath = argv[2]

        NN = readNN(nnPath)
        fout = open(f"./fittingResults.dat", 'w')
        for obsSpecPath in glob.glob(obsPath):
            print(obsSpecPath)
            w, f = read_observations(obsSpecPath, format = 'ascii')
            mask = np.logical_and(f> 0, f<1.5)
            w, f = w[mask], f[mask]
            obsSpec = spectrum(
                            w, f,\
                            res = 15e3
                            )
            obsSpec.ID = obsSpecPath.split('/')[-1].replace('.dat', '')
            labelsFit = fitToNeuralNetwork(obsSpec, NN)
            fout.write( f"{obsSpec.ID} " + '\t'.join(f"{l:.3f}" for l in labelsFit) + '\n')
        fout.close()

    else:
        print(f"Mode {argv[4]} not understood. 'Grid' or 'Payne' are supported.")
        exit()
