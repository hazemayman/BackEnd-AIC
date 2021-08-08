import os
# from geotools.alert import Alert


def create_dir(dirname, path):
    '''
    creates a directory in the specified path
    -----
    params:
    dirname: the name of the new directory
    path: the path in which the directory will be created
    '''
    fullpath = os.path.join(path, dirname)
    if os.path.exists(fullpath):
        print(f"Path already exists at {fullpath}")
    else:
        os.mkdir(fullpath)
        print(f"Created new dir at {fullpath} ")


# @Alert.alert_server
# def predict():
#     '''demo function'''
#     return True
