#this file contains functions that read in different file formats when using the network (eg FASTA etc)

from subprocess import *

def fasta2hdf5(input_file, output_directory):
    tempdir = output_directory
    check_call(['cp', input_file, tempdir + '/input_file.fasta'])
    print "Successfully copied input to working directory."
    check_call(["grep '>'  " + tempdir + '/input_file.fasta ' + ">" + tempdir + '/input_file.fasta.name'], shell=True)
    try:
        check_call(["python 1_fasta2input.nomut.py  " + tempdir + "/input_file.fasta"], shell=True)
    except:
        raise Exception("Fasta format error.")
    print "Successfully converted to input format"
