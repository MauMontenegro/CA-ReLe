import json
import time
from pathlib import Path


def getDir(*args):
    dr = Path.home()
    adds = ["rl_results", *args]
    for s in adds:
        dr /= s
    return dr


def genDir(*args):
    dr = getDir(*args)
    dr.mkdir(parents=True, exist_ok=True)
    return dr


def getExpDir(expName, envName):
    expDir = getDir(expName, envName)
    folders = []
    for folder in expDir.iterdir():
        stem = folder.stem
        if stem != 'tensorboard' and folder.is_dir():
            print('{}: {}'.format(len(folders), stem))
            folders.append(folder)
    while True:
        select = input('Insert the number corresponding to Folder')
        select = int(select)
        if 0 <= select < len(folders):
            break
        print('Error: {} is not a valid option,  please try another'.format(select))
    return folders[select]


def configPath(path):
    return path / "config.json"


def loadconfig(expDir):
    fh = configPath(expDir).open("r")
    config = json.load(fh)
    fh.close()
    return config


def timeFormatedS():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())


class pathManager:
    def __init__(self, config):
        self.config = config
        self.expName = config['agent']['class']
        self.envName = config['env']['name_']
        self.load = config['manager']['load_model']
        self.path = None

        if self.load:
            self.__initLoad__(self.expName, self.envName)
        else:
            self.__initNew__(self.expName, self.envName)

    def __initLoad__(self, expName, envName):
        print('Select Experiment to Load:')
        path = getExpDir(expName, envName)
        self.timeID = path.stem
        self.config = loadconfig(path)

    def __initNew__(self, expName, envName):
        print('Starting new config file')
        self.timeID = t = timeFormatedS()
        if expName == 'unknown':
            print("Warning! Consider set algorithm a different name as it is unknown")
        self.path = genDir(expName, envName, t)
