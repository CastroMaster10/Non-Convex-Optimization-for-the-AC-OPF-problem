import matlab.engine as mat_eng
import numpy as np
import matlab

def read_matpower_powergrid(dir_file):
    eng = mat_eng.start_matlab()
    # Add path
    eng.addpath(dir_file, nargout=0)
    # Load the MATPOWER case
    mpc = eng.pglib_opf_case5_pjm(nargout=1)
    # Convert to internal indexing to get 'order' field
    base_mva = mpc["baseMVA"]
    
    #areas = np.array(mpc["areas"])
    # Access bus data
    bus = np.array(mpc['bus'])
    # Generator data
    gen = np.array(mpc['gen'])
    # Generator cost data
    gencost = np.array(mpc['gencost'])
    # Branch data
    branch = np.array(mpc['branch'])
    return {
        "baseMVA": base_mva,
        #"areas": areas,
        "bus": bus,
        "gen": gen,
        "gencost": gencost,
        "branch": branch,
    }
