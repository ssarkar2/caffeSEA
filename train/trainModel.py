from trainconfig import *
from utils.loadDataUtils import *
from utils.caffeUtils import *
from utils.editUtils import *



def trainToySEA():
    cfg = getConfig()
    convertMatToHDF5(cfg.toyTrainData, cfg.hdf5ToyDir, 0)  #set readMode to 0, in case file already exists for faster execution
    convertMatToHDF5(cfg.toyTestData, cfg.hdf5ToyDir, 0)
    solver = initCaffe([('toySolver', cfg.toySolver)])
    loss, weights = run_solvers(100000, solver)
    del solver
    print 'hello'

def trainFullSEA(dataDir, snapshotLoc):
    #download data from a file in the internet, or skip if the files already exist.
    getFullTrainData(dataDir)
    chunkSize = 10000

    validTxt = convertMatToHDF5(dataDir + 'valid.mat', dataDir + 'hdf5FullInputDir/', 0, chunkSize)
    testTxt = convertMatToHDF5(dataDir + 'test.mat', dataDir + 'hdf5FullInputDir/', 0, chunkSize)
    trainTxt = convertMatToHDF5(dataDir + 'train.mat', dataDir + 'hdf5FullInputDir/', 0, chunkSize)  #does not work due to huge size of input file. need to fix  #fixed

    createDir(dataDir + 'Model')
    outputModelProtoLoc = createModelPrototxt(dataDir + 'Model/', trainTxt, testTxt)  #currently it can only alter the train and test input files
    #'test_iter': '10', 'test_interval': '100',
    newSolverProtoLoc = createSolverPrototxt({ 'display':'10', 'snapshot': '100', 'max_iter':'500', 'snapshot_prefix':snapshotLoc, "net":'"' + outputModelProtoLoc + '"'}, dataDir + 'Model/')

    solver = initCaffe([('fullSolver', newSolverProtoLoc)])
    loss, weights = run_solvers(100000, solver)
    del solver

    print 'helloagain'
