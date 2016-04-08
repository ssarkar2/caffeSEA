
class Config():
    caffeSEAHome = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/caffeSEA'
    
    #toy data 
    toyDataDir = ('/').join([caffeSEAHome, 'data/toyTrainingMatData'])
    toyModelDir = ('/').join([caffeSEAHome, 'models', 'toyModel'])
    toyTrainData = ('/').join([toyDataDir, 'train.mat'])
    toyValidData = ('/').join([toyDataDir, 'valid.mat'])
    toyTestData = ('/').join([toyDataDir, 'test.mat'])
    hdf5ToyDir = ('/').join([caffeSEAHome, 'data/hdf5ToyDir'])  #data is converted to hdf5 and dumped here
    toySolver = ('/').join([toyModelDir, 'solverToy.prototxt'])
    toyModelDefn = ('/').join([toyModelDir, 'trainModel_deepSEA_Toy.prototxt'])
    
    def __setattr__(self, *_):  #so that the values in the class cannot be changed externally
        pass
    
    
def getConfig():
    return Config()