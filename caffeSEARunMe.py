from train.trainModel import *
import sys


if __name__ == '__main__':
    op = int(sys.argv[1])
    if op == 0:
        trainToySEA()
    elif op == 1:
        trainFullSEA('/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/fullData/')
        #trainFullSEA('C:\Sayantan\\acads\cmsc702\deepSEACaffe\\fullData\\')






