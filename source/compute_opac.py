# external
from sys import argv
import numpy as np
from multiprocessing import Pool
# local
from configure_setup import setup
from run_ts import compute_babsma


def babsma_worker(setup, indices):
    for i, atmFile in enumerate(setup.atm_list[indices]):
        atmos = model_atmosphere()
        atmos.read(atmFile, setup.atm_format)
        " Where to write opacities "
        modelOpacFile = f"{setup.cwd}_{atmos.id}_{setup.jobID}"
        # TODO: add specific keywords to ts_input and update TS to write requested opacities
        compute_babsma(ts_input, atmos, modelOpacFile, quite=setup.debug)


if __name__ == '__main__':
    if len(argv) > 2:
        conf_file = argv[1]
    else:
        print("Usage: $ pytnon generate_random_grid.py ./configFile.txt jobName")
        exit()

    setup = setup(file = conf_file)
    setup.jobID = argv[2]

    if 'ncpu' not in setup.__dict__:
        setup.ncpu = 1

    ind = np.arange(len(setup.atm_list))
    args = [ [setup, ind[i::setup.ncpu]] for i in range(set.ncpu)]
    unpackFunc = lambda arg : parallel_worker(arg[0], arg[1])
    with Pool(processes=set.ncpu) as pool:
        pool.map(babsma_worker, args )
