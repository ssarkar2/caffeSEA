from trainconfig import *
from utils.loadDataUtils import convertMatToHDF5
import caffe

def trainToySEA():
    cfg = getConfig()
    convertMatToHDF5(cfg.toyTrainData, cfg.hdf5ToyDir, 1)  #set readMode to 0, in case file already exists for faster execution
	caffe.set_mode_gpu()
	solver = caffe.get_solver("solver.prototxt")
	solver.solve()
    print 'hello'
        
def trainFullSEA():
    #download data from a file in the internet, or skip if the files already exist.
    pass