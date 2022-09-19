# external
from sys import argv
import numpy as np
from multiprocessing import Pool
# local
from configure_setup import setup
from run_ts import compute_babsma
from atmos_package import model_atmosphere


def babsma_worker(args):
    setup, indices = args
    for i, atmFile in enumerate(setup.atmos_list[indices]):
        print(atmFile)
        atmos = model_atmosphere()
        atmos.read(atmFile, setup.atmos_format)
       
        " Where to write opacities "
        modelOpacFile = f"{setup.cwd}_{atmos.id}_{setup.jobID}"
        # TODO: add specific keywords to ts_input and update TS to write requested opacities
        if setup.atmos_format.strip() == 'marcs':
            setup.ts_input['MARCS-FILE'] = '.true.'
            atmos.path = atmFile
        else: 
            atmos.path = f"{setup.cwd}/{atmos.ID}.dat"
            write_atmos_m1d4TS(atmos,  atmos.path)
            
        compute_babsma(setup.ts_input, atmos, modelOpacFile, quite=setup.debug)


if __name__ == '__main__':
    if len(argv) > 2:
        conf_file = argv[1]
    else:
        print("Usage: $ pytnon generate_random_grid.py ./configFile.txt jobName")
        exit()

    setup = setup(file = conf_file, mode = 'MAprovided')
    setup.jobID = argv[2]

    if 'ncpu' not in setup.__dict__:
        setup.ncpu = 1

    ind = np.arange(len(setup.atmos_list))
    args = [ [setup, ind[i::setup.ncpu]] for i in range(setup.ncpu)]
    with Pool(processes=setup.ncpu) as pool:
        pool.map(babsma_worker, args )
