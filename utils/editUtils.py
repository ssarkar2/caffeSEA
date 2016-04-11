#this file contains functions to edit the standard prototexts

from models import caffe_pb2

# new_data = {'max_iter': 5000, 'snapshot': 500}
def createSolverPrototxt(new_data, save_loc):
    f = open('solverToy.prototxt')
    def_text = f.read().split('\n')
    def_text.remove('')
    solver_default_dict = {module.split(': ')[0]: module.split(': ')[1] for module in def_text}
    new_dictionary = solver_default_dict
    for key, value in new_data.iteritems():
        new_dictionary[key] = str(value)
    f.close()
    new_solver_text = ['{}: {}'.format(key, value) for key, value in new_dictionary.iteritems()]
    new_proto = '\n'.join(new_solver_text)
    new_file_name = save_loc + 'solverToy_new.prototxt'
    f_new = open(new_file_name, 'w')
    f_new.write(new_proto)
    f_new.close()
