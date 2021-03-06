#this file contains functions to read input data for training the network

import os
import scipy.io as sio
import h5py
import numpy as np
import urllib2
import tarfile
from train.trainconfig import *

def addSlash(dirname): #if a directory location does nto end with '/' add it at the end
    return dirname + ('/', '')[dirname[-1] == '/']

def convertMatToHDF5(matData, hdf5DataDir, readMode, split=10000):  #split into files each containing 'split' number of samples
    hdf5DataDir = addSlash(hdf5DataDir)
    createDir(hdf5DataDir)
    fileName = getFileName(matData)
    #fullFileName = hdf5DataDir + fileName + '.hdf5'
    if readMode == 0:
        txtFile = hdf5DataDir + fileName + '.txt'
        if os.path.isfile(txtFile): return txtFile

    print matData
    
    """
    short note on mat file loading in python (the try and except paths below)
     matlab usually saves in 'v7' format. this format can only store files < 2gb. hence test.mat and valid.mat are in 'v7' form
     train.mat is > 2gb, hence it is in 'v7.3' format.
     v7.3 is basically a hdf5 file (u can open .mat files saved in this format using hdf5viewers etc)

     v7 however is not in hdf5 format, (hence I was using scipy and not h5py to open that (the try path))

     A simple way to unify the try/except paths is to convert 'test.mat' and 'valid.mat' to 'v7.3' and use h5py for everything

    save('file.mat', 'var1', 'var2', '-v7.3')
    """
    try:
        #matdata = sio.loadmat(matData)
        #[matDataName, matLabelName] = getDataNames(matdata.keys())
        #with h5py.File(fullFileName,'w') as f:
        #    swappedData = swapDims(matdata[matDataName])
        #    dataH5 = f.create_dataset('data', swappedData.shape, dtype='i1')  #i1 indicates 1byte sized integer.
        #    dataH5[...] = swappedData  #matdata[matDataName] is (10000, 4, 1000). swappedData is (10000,1,4,1000)
        #    labelH5 = f.create_dataset('label', matdata[matLabelName].shape, dtype='i1')
        #    labelH5[...] = matdata[matLabelName]  #(10000, 919)


        matdata = sio.loadmat(matData)
        [matDataName, matLabelName] = getDataNames(matdata.keys())
        swappedData = swapDims(matdata[matDataName])
        datasize = swappedData.shape
        chunkCount = 0;
        while(1):
            chunkCount += 1
            with h5py.File(hdf5DataDir + fileName + str(chunkCount) + '.hdf5') as f:
                if chunkCount*split < datasize[0]:
                    numInThisChunk = split
                    endidx = chunkCount*split
                else:
                    numInThisChunk = datasize[0] - (chunkCount-1)*split
                    endidx = None  #till the end

                dataH5 = f.create_dataset('data', (numInThisChunk, datasize[1], datasize[2], datasize[3]), dtype='i1')  #i1 indicates 1byte sized integer.
                dataH5[...] = swappedData[(chunkCount-1)*split : endidx][:][:][:]
                labelH5 = f.create_dataset('label', (numInThisChunk, 919), dtype='i1')
                labelH5[...] = matdata[matLabelName][(chunkCount-1)*split : endidx][:]
                if endidx == None: break
    except:
        print 'scipy loading of mat file failed. using h5py'
        #matDataName = callMatlabDimensionConverter(matData)
        #callMatlabDimensionConverter(matData)
        #with h5py.File(matData,'r') as hf:
        #    [matDataName, matLabelName] = getDataNames(hf.keys())
        #    with h5py.File(fullFileName,'w') as f:
        #        print hf[matDataName].shape   #(4000, 1, 1, 4400000)   #WRONG. CHECK
        #        dataH5 = f.create_dataset('data', hf[matDataName].shape, dtype='i1')  #i1 indicates 1byte sized integer.
        #        dataH5[...] = hf[matDataName]
        #        labelH5 = f.create_dataset('label', hf[matLabelName].shape, dtype='i1')
        #        labelH5[...] = hf[matLabelName]  #(10000, 919)

        print 'call matlab'
        callMatlabDimensionConverter(matData, hdf5DataDir, split)
        chunkCount = sio.loadmat(hdf5DataDir + 'chunkCount.mat')['chunkCount']
        print 'matlab done. files created: ', chunkCount
        #print matData
        #with h5py.File(matData,'r') as hf:
        #    [matDataName, matLabelName] = getDataNames(hf.keys())
        #    print  matDataName, hf[matDataName].shape
        #    print  matLabelName, hf[matLabelName].shape

    with open(hdf5DataDir + fileName+'.txt', 'w') as ftxt:
        for i in range(chunkCount):
            ftxt.write(hdf5DataDir + fileName + str(i+1) + '.hdf5\n')

    return ('/').join([hdf5DataDir, fileName+'.txt' ])


def callMatlabDimensionConverter(matData, hdf5DataDir, chunksz):
    os.system(Config().matlabPath + ' -nodisplay -nosplash -nodesktop -r "addpath(\'./utils\'); matReshape(\'' + matData +',' + hdf5DataDir + ',' + str(chunksz)+ '\'); quit"')
    #return matData.replace('.mat', '_swap.mat')


def swapDims(nparr):
    shp = nparr.shape
    return np.reshape(nparr, (shp[0], 1, shp[1], shp[2]))


def createDir(dirname): #check if such a directory exists, else create a new one.
    try: os.stat(dirname)
    except: os.mkdir(dirname)


def getFileName(fullName):
    return fullName.split('/')[-1].split('.')[0]   #for ubuntu
    #return fullName.split('\\')[-1].split('.')[0]   #for windows

def getDataNames(names):
    varnames = [i for i in names if '__' not in i]  #remove things like __global__, __version__ etc. now varnames is of size 2 only
    flag = (1, 0)['x' in varnames[0]]  #check if 'x' is present in the variable name
    return [varnames[flag], varnames[1-flag]]


def getFullTrainData(dataDir):
    url = 'http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz'
    print dataDir + 'train.mat'
    if (os.path.isfile(dataDir + 'train.mat') and os.path.isfile(dataDir + 'valid.mat') and os.path.isfile(dataDir + 'test.mat')):
        return
    else:
        file_name = dataDir + url.split('/')[-1]
        if not os.path.isfile(dataDir + 'deepsea_train_bundle.v0.9.tar.gz'):
            downloadFile(file_name, url)
        tar = tarfile.open(file_name, "r:gz")
        print 'extraction started...'
        extractFile(tar, 'train', dataDir); print 'train extract done'
        extractFile(tar, 'valid', dataDir); print 'valid extract done'
        extractFile(tar, 'test', dataDir); print 'test extract done'
        tar.close()
        #os.remove(dataDir + 'deepsea_train_bundle.v0.9.tar.gz')

def extractFile(tar, fileName, outFolder):
    member = tar.getmember('deepsea_train/' + fileName + '.mat')
    member.name = os.path.basename(member.name)
    tar.extract(member, outFolder)

def downloadFile(file_name, url):
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()


