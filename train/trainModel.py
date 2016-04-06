from trainconfig import *
from utils.loadDataUtils import convertMatToHDF5

def trainToySEA():
    cfg = getConfig()
    convertMatToHDF5(cfg.toyTrainData, cfg.hdf5ToyDir, 1)  #set readMode to 0, in case file already exists for faster execution
    print 'hello'
        
def trainFullSEA():
    #download data from a file in the internet, or skip if the files already exist.
    pass