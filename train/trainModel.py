from trainconfig import *
from utils.loadDataUtils import *
from utils.caffeUtils import *




def trainToySEA():
    cfg = getConfig()
    convertMatToHDF5(cfg.toyTrainData, cfg.hdf5ToyDir, 0)  #set readMode to 0, in case file already exists for faster execution
    convertMatToHDF5(cfg.toyTestData, cfg.hdf5ToyDir, 0)
    solver = initCaffe([('toySolver', cfg.toySolver)])
    loss, weights = run_solvers(100000, solver)
    del solver
    print 'hello'
        
def trainFullSEA(dataDir):
    #download data from a file in the internet, or skip if the files already exist.
    getFullTrainData(dataDir)
    #convertMatToHDF5_(dataDir + 'train.mat', dataDir + 'hdf5FullInputDir', 0)  #doesnot work due to huge size of input file. need to fix
    convertMatToHDF5_(dataDir + 'valid.mat', dataDir + 'hdf5FullInputDir', 0)
    #convertMatToHDF5_(dataDir + 'test.mat', dataDir + 'hdf5FullInputDir', 0)
    print 'helloagain'
    pass
