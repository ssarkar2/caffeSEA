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
        
def trainFullSEA(dataDir):
    #download data from a file in the internet, or skip if the files already exist.
    getFullTrainData(dataDir)

    validTxt = convertMatToHDF5(dataDir + 'valid.mat', dataDir + 'hdf5FullInputDir', 0)
    testTxt = convertMatToHDF5(dataDir + 'test.mat', dataDir + 'hdf5FullInputDir', 0)
    trainTxt = convertMatToHDF5(dataDir + 'train.mat', dataDir + 'hdf5FullInputDir', 0, 20)  #does not work due to huge size of input file. need to fix  #fixed

    createDir(dataDir + 'Model')
    outputModelProtoLoc = createModelPrototxt(dataDir + 'Model/', trainTxt, testTxt)  #currently it can only alter the train and test input files
    newSolverProtoLoc = createSolverPrototxt({'snapshot_prefix':"\"../dumpModels/caffeSEAFull_\"", "net":'"' + outputModelProtoLoc + '"'}, dataDir + 'Model/')

    newSolverProtoLoc = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/fullData/Model/solver_new_hand.prototxt'    #HACK: FIX: CHECK WHY AUTO SOLVER PROTO GENERATION FAILS

    #ALSO, the train hdf5 file is too big... create small chunks of train files

    #solver = initCaffe([('fullSolver', newSolverProtoLoc)])
    #loss, weights = run_solvers(100000, solver)
    #del solver

    print 'helloagain'
    pass
