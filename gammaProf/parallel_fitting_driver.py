import sys
import pdb
import time
import glob
import h5py
import numpy as np
from mpi4py import MPI
import example_run as fitter

def parallel_profile_fit(lensing_dir):
    
    # toggle this on to test communication without actually performing fits
    dry_run = False
    # toggle this on to redo fits even if files exist
    overwrite = False

    # -----------------------------------------
    # ---------- define communicator ----------
    comm= MPI.COMM_WORLD
    rank = comm.Get_rank()
    numranks = comm.Get_size()
    if(rank==0):
        print('\n---------- starting with {} MPI processes ----------'.format(numranks))
        sys.stdout.flush()
    comm.Barrier()


    # --------------------------------------
    # ---------- find all cutouts ----------
    all_cutouts = np.array(glob.glob('{}/halo*'.format(lensing_dir)))
  

    # ----------------------------------------------------------------------
    # ---------- remove cutouts with missing or incomplete mocks -----------
    if(rank == 0): 
        print('found {} total halo mocks'.format(len(all_cutouts))) 
        print('removing empty and truncated cutouts') 
    empty_mask = np.array([len(glob.glob('{}/*mock*'.format(c))) != 0 for c in all_cutouts])
    all_cutouts = all_cutouts[empty_mask]
    
    cutout_dens_depth = np.array( [max([int(s.split('plane')[-1].split('_')[0]) 
                                        for s in glob.glob('{}/dtfe_dens/*plane*'.format(c))])
                                        for c in all_cutouts] )
    cutout_mock_depth = np.array( [max([int(s.split('plane')[-1]) for s in 
                                        list(h5py.File(glob.glob('{}/*lensing_mocks.hdf5'.format(c))[0]).keys())])
                                        for c in all_cutouts] )
    truncated_mask = np.array( [cutout_dens_depth[i] == cutout_mock_depth[i] 
                                for i in range(len(all_cutouts))] )
    all_cutouts = all_cutouts[truncated_mask]
    
    if(rank == 0):
        print('distributing {} mocks to {} ranks'.format(len(all_cutouts), numranks))
        sys.stdout.flush()


    # -------------------------------------------------
    # ---------- distribute cutouts to ranks ----------
    this_rank_halos = np.array_split(all_cutouts, numranks)[rank]
    print("rank {} gets {} mocks".format(rank, len(this_rank_halos)), flush=True)
    comm.Barrier()
    #sys.stdout.flush()
    #comm.Barrier()
   
    
    # ----------------------------------------------------------------------------
    # ---------------- do profile fitting on mocks for this rank -----------------
    start = time.time()
    for i in range(len(this_rank_halos)):
        
        if(rank==0): 
            print('\n---------- working on halo {}/{} ----------'.format(i+1, len(this_rank_halos)))
            sys.stdout.flush()
        cutout = this_rank_halos[i] 
        
        if( (len(glob.glob('{}/profile_fits/*.npy'.format(cutout))) < 4 or overwrite) and not dry_run):
            fitter.sim_example_run(halo_cutout_dir = cutout, showfig=False, stdout=(rank==0))
    end = time.time()

    # all done
    comm.Barrier()
    if(rank == 0): print('\n')
    comm.Barrier()
    print('rank {} finished {} halos in {:.2f} s'.format(
          rank, len(this_rank_halos), end-start))


if __name__ == '__main__':
    parallel_profile_fit(sys.argv[1])
