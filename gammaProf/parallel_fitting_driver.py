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
    overwrite = True

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
    #all_cutouts = np.array(glob.glob('{}/halo*'.format(lensing_dir)))
    all_cutouts = np.array(glob.glob('{}/halo_244960324069_0'.format(lensing_dir)))
    all_cutouts = np.hstack([all_cutouts, 
                            np.array(glob.glob('{}/halo_404385825240_0'.format(lensing_dir)))])


    # ----------------------------------------------------------------------
    # ---------- remove cutouts with missing or incomplete mocks -----------
    if(rank == 0): 
        print('found {} total halo mocks'.format(len(all_cutouts))) 
        print('removing empty and truncated cutouts') 
    nofile_mask = np.array([len(glob.glob('{}/*mock*'.format(c))) != 0 for c in all_cutouts])
    all_cutouts = all_cutouts[nofile_mask]
    cutout_mock_planes = np.array( [[int(s.split('plane')[-1]) 
                                        for s in list(h5py.File(glob.glob(
                                        '{}/*lensing_mocks.hdf5'.format(c))[0], 'r').keys())]
                                        for c in all_cutouts] )
    empty_mask = [len(plane) > 0 for plane in cutout_mock_planes]
    all_cutouts = all_cutouts[empty_mask]
    cutout_mock_planes = cutout_mock_planes[empty_mask]
    
    cutout_dens_depth = np.array( [max([int(s.split('plane')[-1].split('_')[0]) 
                                        for s in glob.glob('{}/dtfe_dens/*plane*'.format(c))])
                                        for c in all_cutouts] )
    cutout_mock_depth = np.array( [max(planes) for planes in cutout_mock_planes] )

    truncated_mask = np.array( [cutout_dens_depth[i] == cutout_mock_depth[i] 
                                for i in range(len(all_cutouts))] )
    all_cutouts = all_cutouts[truncated_mask]
    
    if(rank == 0): 
        print('removing cutouts missing concentrations')
   
    conc_mask = np.array( [len(np.genfromtxt('{}/properties.csv'.format(c), 
                               delimiter=',', names=True).dtype ) == 11 
                           for c in all_cutouts] ) 
    all_cutouts = all_cutouts[conc_mask]
    
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

        cutout = this_rank_halos[i]
        mass = np.genfromtxt('{}/properties.csv'.format(cutout), delimiter=',', names=True)['sod_halo_mass']
        if(np.log10(mass) > 14.5) : makeplot=True
        else: makeplot=False
        
        if(rank==0): 
            print('\n---------- working on halo {}/{} with mass {:.2E}----------'.format(
                    i+1, len(this_rank_halos), mass))
            sys.stdout.flush()

        if( (len(glob.glob('{}/profile_fits/*0.3rmin.npy'.format(cutout))) < 4 or overwrite) and not dry_run):
            fitter.sim_example_run(halo_cutout_dir = cutout, makeplot=makeplot, showfig=False, 
                                   stdout=(rank==0), bin_data=True, rbins=30, rmin=0.3)
    
    # -------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------
    
    end = time.time()

    # all done
    comm.Barrier()
    if(rank == 0): print('\n')
    comm.Barrier()
    print('rank {} finished {} halos in {:.2f} s'.format(
          rank, len(this_rank_halos), end-start))


if __name__ == '__main__':
    parallel_profile_fit(sys.argv[1])
