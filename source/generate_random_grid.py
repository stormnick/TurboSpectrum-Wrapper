# external
from sys import argv
import numpy as np
from multiprocessing import Pool
# local
from configure_setup import setup
from run_ts import parallel_worker

def run_TS_parallel(set):
    """
    Splits requested input parameters into N_CPU chunks and calls
    parallel_worker N_CPU times in parallel with respective input

    Parameters
    ----------
    set : setup
        Configuration of requested computations
    """

    if set.ncpu > set.inputParams['count']:
        set.ncpu = set.inputParams['count']
        print(f"Requested more CPUs than jobs. \
Will use {set.ncpu} CPUs instead")

    ind = np.arange(set.inputParams['count'])
    args = [ [set, ind[i::set.ncpu]] for i in range(set.ncpu)]

    unpackFunc = lambda arg : parallel_worker(arg[0], arg[1])
    with Pool(processes=set.ncpu) as pool:
        pool.map(parallel_worker, args )



if __name__ == '__main__':
    if len(argv) > 2:
        conf_file = argv[1]
    else:
        conf_file = "config.txt"
        #print("Usage: $ pytnon generate_random_grid.py ./configFile.txt jobName")
        #exit()

    set = setup(file = conf_file)
    # TODO: assign random name / cwd name if empty or not provided
    try:
        set.jobID = argv[2]
    except IndexError:
        set.jobID = f"jobID_{np.random.random()}"

    run_TS_parallel(set)
