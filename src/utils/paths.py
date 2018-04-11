from os.path import dirname, join


PATH_PROJECT = dirname(dirname(dirname(__file__)))
PATH_DATA = join(PATH_PROJECT, 'data')
PROCESSED_NPY_DATA = join(PATH_DATA, 'npy')
PATH_SRC = join(PATH_PROJECT, 'src')
PATH_WEIGHTS = join(PATH_DATA, 'models')
PATH_OUTPUT = join(PATH_DATA, 'output')
