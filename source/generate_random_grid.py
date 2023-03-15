# external
import socket
from sys import argv
import numpy as np
#from multiprocessing import Pool
from dask.distributed import Client
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

    print("Preparing workers")  # TODO check memory issues? set higher? give warnings?
    client = Client(threads_per_worker=1, n_workers=set.ncpu)
    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "gemini-login.mpia.de"
    print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

    # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
    print(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

    print("Worker preparation complete")

    futures = []

    for one_index in ind:
        scattered_future = client.scatter([set, one_index])
        future = client.submit(parallel_worker, scattered_future)
        futures.append(future)

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    print("Worker calculation done")  # when done, save values

    """
    args = [ [set, ind[i::set.ncpu]] for i in range(set.ncpu)]

    unpackFunc = lambda arg : parallel_worker(arg[0], arg[1])
    with Pool(processes=set.ncpu) as pool:
        pool.map(parallel_worker, args )"""



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
