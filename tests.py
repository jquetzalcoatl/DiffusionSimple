import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.utils as vutils
import json
import PIL
import logging
import sys
import argparse
import getpass

sys.path.insert(1, '/home/' + getpass.getuser() + '/Projects/DiffusionSimple/util')

from loaders import generateDatasets, inOut, saveJSON, loadJSON#, MyData
from tools import accuracy, tools, per_image_error, predVsTarget, errInDS, errInDS_2, errOverLat
from plotter import myPlots, plotSamp, plotSampRelative
from NNets import SimpleCNN, SimpleCNNConvT, SimpleCNN_L, SimpleCNN_S, UNet, LeakyUNet, SimpleCNNCat, SimpleCNNJules, SimpleCNNReflect, UNetGPT, UNetBias0, UNetPrelu, SimpleCNNJulesPB, UNetPB, UNetPreluPB, LeakyUNetPB, UNetBias0PB, SimpleCNNCatPB, UNetPrelu2PB

def select_nn(arg, d=None, num_samples=1):
    if arg == "SimpleCNN":
        class DiffSur(SimpleCNN):
            pass 
    elif arg == "UNet":
        class DiffSur(UNet):
            pass
    elif arg == "SimpleCNNConvT":
        class DiffSur(SimpleCNNConvT):
            pass
    elif arg == "SimpleCNN_L":
        class DiffSur(SimpleCNN_L):
            pass
    elif arg == "SimpleCNN_S":
        class DiffSur(SimpleCNN_S):
            pass
    elif arg == "LeakyUNet":
        class DiffSur(LeakyUNet):
            pass
    elif arg == "LeakyUNetPB":
        class DiffSur(LeakyUNetPB):
            pass
    elif arg == "SimpleCNNCat":
        class DiffSur(SimpleCNNCat):
            pass
    elif arg == "SimpleCNNJules":
        class DiffSur(SimpleCNNJules):
            pass
    elif arg == "SimpleCNNReflect":
        class DiffSur(SimpleCNNReflect):
            pass
    elif arg == "UNetGPT":
        class DiffSur(UNetGPT):
            pass
    elif arg == "UNetBias0":
        class DiffSur(UNetBias0):
            pass
    elif arg == "UNetBias0PB":
        class DiffSur(UNetBias0PB):
            pass
    elif arg == "UNetPrelu":
        class DiffSur(UNetPrelu):
            pass
    elif arg == "SimpleCNNJulesPB":
        class DiffSur(SimpleCNNJulesPB):
            pass
    elif arg == "UNetPB":
        class DiffSur(UNetPB):
            pass
    elif arg == "UNetPreluPB":
        class DiffSur(UNetPreluPB):
            pass
    elif arg == "SimpleCNNCatPB":
        class DiffSur(SimpleCNNCatPB):
            pass
    elif arg == "UNetPrelu2PB":
        class DiffSur(UNetPrelu2PB):
            pass
    return DiffSur()


class thelogger(object):
    def logFunc(self, PATH, dict, dir="0"):
            self.initTime = datetime.now()
            os.path.isdir(PATH + "Logs/") or os.mkdir(PATH + "Logs/")
            os.path.isdir(PATH + "Logs/" + dir) or os.mkdir(PATH + "Logs/" + dir)
            path = PATH + "Logs/" + dir + "/"

            self.logging = logging
            self.logging = logging.getLogger()
            self.logging.setLevel(logging.DEBUG)
            self.handler = logging.FileHandler(os.path.join(path, 'tests.log'))
            self.handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                fmt='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.handler.setFormatter(formatter)
            self.logging.addHandler(self.handler)

            self.logging.info(f'{str(self.initTime).split(".")[0]} - Log')

def selectNN(dict):
#     if dict["NN"] != "SimpleAdapCNN100":
#         diffSolv = select_nn(dict["NN"])
#         diffSolv = diffSolv().to(device)
#     else:
#         diffSolv = SimpleAdapCNN100(dict).to(device)
    diffSolv = select_nn(dict["NN"], d=dict)
    diffSolv = diffSolv.to(device)
    return diffSolv

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Deep Diffusion Solver")
    parser.add_argument('--path', dest="path", type=str, default='/home/' + getpass.getuser() +'/Projects/DiffusionSimple/Results/',
                        help="Specify path to dataset")   #Change
    parser.add_argument('--dataset', dest="dataset", type=str, default="Datav2",
                        help="Specify dataset")
    parser.add_argument('--dir', dest="dir", type=str, default="100DGX-" + str(datetime.now()).split(" ")[0],
                        help="Specify directory name associated to model")
    parser.add_argument('--bashtest', dest="bashtest", type=bool, default=False,
                        help="Leave default unless testing flow")
    
    parser.add_argument('--bs', dest="bs", type=int, default=50,
                        help="Specify Batch Size")
    parser.add_argument('--nw', dest="nw", type=int, default=8,
                        help="Specify number of workers")
    parser.add_argument('--ngpu', dest="ngpu", type=int, default=1,
                        help="Specify ngpu. (Never have tested >1)")
    parser.add_argument('--lr', dest="lr", type=float, default=0.0001,
                        help="Specify learning rate")
    parser.add_argument('--maxep', dest="maxep", type=int, default=100,
                        help="Specify max epochs")
    
    parser.add_argument('--newdir', dest="newdir", type=bool, default=False,
                        help="Is this a new model?")
    parser.add_argument('--newtrain', dest="newtrain", type=bool, default=False,
                        help="Are you starting training")
    
    
    parser.add_argument('--transformation', dest="transformation", type=str, default="linear",
                        help="Select transformation: linear, sqrt or log?")
    parser.add_argument('--loss', dest="loss", type=str, default="exp",
                        help="Select loss: exp, step, toggle or rand?")
    parser.add_argument('--wtan', dest="wtan", type=float, default=10.0,
                        help="Specify hparam wtan")
    parser.add_argument('--w', dest="w", type=float, default=1.0,
                        help="Specify hparam w")
    parser.add_argument('--alpha', dest="alpha", type=int, default=1,
                        help="Specify hparam alpha")
    parser.add_argument('--w2', dest="w2", type=float, default=4000.0,
                        help="Specify hparam w2")
    parser.add_argument('--delta', dest="delta", type=float, default=0.02,
                        help="Specify hparam delta")
    parser.add_argument('--toggle', dest="toggle", type=int, default=1,
                        help="Specify hparam toggle")
      
    args = parser.parse_args()
    
    
    ###Start Here
    PATH = args.path # "/raid/javier/Datasets/DiffSolver/"
    DATASETNAME = args.dataset # "All"
    os.path.isdir(PATH + "AfterPlots/") or os.mkdir(PATH + "AfterPlots/")
    os.path.isdir(PATH + "AfterPlots/errors/") or os.mkdir(PATH + "AfterPlots/errors/")
    os.path.isdir(PATH + "AfterPlots/Samples/") or os.mkdir(PATH + "AfterPlots/Samples/")
    os.path.isdir(PATH + "AfterPlots/Pred/") or os.mkdir(PATH + "AfterPlots/Pred/")

    dir = args.dir #'1DGX' #'Test'#str(21)
    BATCH_SIZE=args.bs #50
    NUM_WORKERS=args.nw #8
    ngpu = args.ngpu #1
    lr = args.lr #0.0001
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#     diffSolv = DiffSur().to(device)
    os.listdir(PATH + "Dict/")

    selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}}}

    datasetNameList = [DATASETNAME] # [f'{i}SourcesRdm' for i in range(1,21)]
    error, errorField, errorSrc = [], [], []

    for selectedDir in selectedDirs.keys():
        dir = selectedDir
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
#        print(dict, '\n')
        diffSolve = selectNN(dict)
        try:
            ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict, tag='Best')
        except:
            ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
#         ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
        theModel.eval();
        myLog = thelogger()
        myLog.logFunc(PATH, dict, dir)
        myLog.logging.info(f'Generating tests using MSE... for model {selectedDir}')
        for (j, ds) in enumerate(datasetNameList):
            myLog.logging.info(f'Dataset: {ds}')
            trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=256, transformation=dict["transformation"], val=False).getDataLoaders()
    #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=512)
            arr = errInDS(theModel, testloader, device, transformation=dict["transformation"], error_fnc=nn.MSELoss(reduction='none'))
            selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
            selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
            selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
            selectedDirs[selectedDir]["max"]["all"].append(arr[3])
            selectedDirs[selectedDir]["max"]["field"].append(arr[4])
            selectedDirs[selectedDir]["max"]["src"].append(arr[5])
            selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
            selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
            selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
            selectedDirs[selectedDir]["min"]["all"].append(arr[9])
            selectedDirs[selectedDir]["min"]["field"].append(arr[10])
            selectedDirs[selectedDir]["min"]["src"].append(arr[11])
            selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
            selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
            selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

            selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
            selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
            selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
            selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
            selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
            selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
            selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
            selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
            selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
            selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
            selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
            selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
            selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
            selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
            selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
            
            selectedDirs[selectedDir]["mean"]["ring4"].append(arr[30])
            selectedDirs[selectedDir]["max"]["ring4"].append(arr[31])
            selectedDirs[selectedDir]["maxmean"]["ring4"].append(arr[32])
            selectedDirs[selectedDir]["min"]["ring4"].append(arr[33])
            selectedDirs[selectedDir]["minmean"]["ring4"].append(arr[34])
        myLog.logging.info(f'Finished tests over datasets')

    modelName = next(iter(selectedDirs.keys()))
    saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}_MSE.json')
    myLog.logging.info(f'JSON object saved')
    
#     selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}}}

#     datasetNameList = [DATASETNAME]#[f'{i}SourcesRdm' for i in range(1,21)]
#     error, errorField, errorSrc = [], [], []

#     for selectedDir in selectedDirs.keys():
#         dir = selectedDir
#         os.listdir(os.path.join(PATH, "Dict", dir))[0]
#         dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
# #        print(dict, '\n')
#         diffSolve = selectNN(dict)
#         try:
#             ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict, tag='Best')
#         except:
#             ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
# #         ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
#         theModel.eval();
# #         myLog = thelogger()
# #         myLog.logFunc(PATH, dict, dir)
#         myLog.logging.info(f'Generating tests using MSE on training set... for model {selectedDir}')
#         for (j, ds) in enumerate(datasetNameList):
#             myLog.logging.info(f'Dataset: {ds}')
#             trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=256, transformation=dict["transformation"], val=True).getDataLoaders()
#     #     selectedDirs[selectedDir] = t.errorPerDataset(PATH, theModel, device, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, std_tr=0.0, s=512)
#             arr = errInDS(theModel, trainloader, device, transformation=dict["transformation"], error_fnc=nn.MSELoss(reduction='none'))
#             selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
#             selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
#             selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
#             selectedDirs[selectedDir]["max"]["all"].append(arr[3])
#             selectedDirs[selectedDir]["max"]["field"].append(arr[4])
#             selectedDirs[selectedDir]["max"]["src"].append(arr[5])
#             selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
#             selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
#             selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
#             selectedDirs[selectedDir]["min"]["all"].append(arr[9])
#             selectedDirs[selectedDir]["min"]["field"].append(arr[10])
#             selectedDirs[selectedDir]["min"]["src"].append(arr[11])
#             selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
#             selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
#             selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

#             selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
#             selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
#             selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
#             selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
#             selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
#             selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
#             selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
#             selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
#             selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
#             selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
#             selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
#             selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
#             selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
#             selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
#             selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
            
#             selectedDirs[selectedDir]["mean"]["ring4"].append(arr[30])
#             selectedDirs[selectedDir]["max"]["ring4"].append(arr[31])
#             selectedDirs[selectedDir]["maxmean"]["ring4"].append(arr[32])
#             selectedDirs[selectedDir]["min"]["ring4"].append(arr[33])
#             selectedDirs[selectedDir]["minmean"]["ring4"].append(arr[34])
#         myLog.logging.info(f'Finished tests over datasets')

#     modelName = next(iter(selectedDirs.keys()))
#     saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-tr-{modelName}_MSE.json')
#     myLog.logging.info(f'JSON object saved')

###################    
    selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}}}
#     datasetNameList = ['Data']#[f'{i}SourcesRdm' for i in range(1,21)]
    error, errorField, errorSrc = [], [], []

    for selectedDir in selectedDirs.keys():
        dir = selectedDir
        os.listdir(os.path.join(PATH, "Dict", dir))[0]
        dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
#        print(dict, '\n')
        diffSolve = selectNN(dict)
        try:
            ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict, tag='Best')
        except:
            ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
#         ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
        theModel.eval();
        myLog.logging.info(f'Generating tests using MAE... for model {selectedDir}')
        for (j, ds) in enumerate(datasetNameList):
            myLog.logging.info(f'Dataset: {ds}')
            trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=256, transformation=dict["transformation"], val=False).getDataLoaders()
            arr = errInDS(theModel, testloader, device, transformation=dict["transformation"])
            selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
            selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
            selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
            selectedDirs[selectedDir]["max"]["all"].append(arr[3])
            selectedDirs[selectedDir]["max"]["field"].append(arr[4])
            selectedDirs[selectedDir]["max"]["src"].append(arr[5])
            selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
            selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
            selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
            selectedDirs[selectedDir]["min"]["all"].append(arr[9])
            selectedDirs[selectedDir]["min"]["field"].append(arr[10])
            selectedDirs[selectedDir]["min"]["src"].append(arr[11])
            selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
            selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
            selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

            selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
            selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
            selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
            selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
            selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
            selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
            selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
            selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
            selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
            selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
            selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
            selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
            selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
            selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
            selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
            
            selectedDirs[selectedDir]["mean"]["ring4"].append(arr[30])
            selectedDirs[selectedDir]["max"]["ring4"].append(arr[31])
            selectedDirs[selectedDir]["maxmean"]["ring4"].append(arr[32])
            selectedDirs[selectedDir]["min"]["ring4"].append(arr[33])
            selectedDirs[selectedDir]["minmean"]["ring4"].append(arr[34])           
        myLog.logging.info(f'Finished tests over datasets')

    modelName = next(iter(selectedDirs.keys()))
    saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-{modelName}.json')
    myLog.logging.info(f'JSON object saved')
    
#     selectedDirs = {dir : {"mean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "max" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "maxmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "min" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}, "minmean" : {"all" : [], "src" : [], "field" : [], "ring1" : [], "ring2" : [], "ring3" : [], "ring4" : []}}}
#     datasetNameList = [DATASETNAME]#[f'{i}SourcesRdm' for i in range(1,21)]
#     error, errorField, errorSrc = [], [], []

#     for selectedDir in selectedDirs.keys():
#         dir = selectedDir
#         os.listdir(os.path.join(PATH, "Dict", dir))[0]
#         dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
# #        print(dict, '\n')
#         diffSolve = selectNN(dict)
#         try:
#             ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict, tag='Best')
#         except:
#             ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
# #         ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
#         theModel.eval();
#         myLog.logging.info(f'Generating tests using MAE on training set... for model {selectedDir}')
#         for (j, ds) in enumerate(datasetNameList):
#             myLog.logging.info(f'Dataset: {ds}')
#             trainloader, testloader = generateDatasets(PATH, datasetName=ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=256, transformation=dict["transformation"], val=True).getDataLoaders()
#             arr = errInDS(theModel, trainloader, device, transformation=dict["transformation"])
#             selectedDirs[selectedDir]["mean"]["all"].append(arr[0])
#             selectedDirs[selectedDir]["mean"]["field"].append(arr[1])
#             selectedDirs[selectedDir]["mean"]["src"].append(arr[2])
#             selectedDirs[selectedDir]["max"]["all"].append(arr[3])
#             selectedDirs[selectedDir]["max"]["field"].append(arr[4])
#             selectedDirs[selectedDir]["max"]["src"].append(arr[5])
#             selectedDirs[selectedDir]["maxmean"]["all"].append(arr[6])
#             selectedDirs[selectedDir]["maxmean"]["field"].append(arr[7])
#             selectedDirs[selectedDir]["maxmean"]["src"].append(arr[8])
#             selectedDirs[selectedDir]["min"]["all"].append(arr[9])
#             selectedDirs[selectedDir]["min"]["field"].append(arr[10])
#             selectedDirs[selectedDir]["min"]["src"].append(arr[11])
#             selectedDirs[selectedDir]["minmean"]["all"].append(arr[12])
#             selectedDirs[selectedDir]["minmean"]["field"].append(arr[13])
#             selectedDirs[selectedDir]["minmean"]["src"].append(arr[14])

#             selectedDirs[selectedDir]["mean"]["ring1"].append(arr[15])
#             selectedDirs[selectedDir]["mean"]["ring2"].append(arr[16])
#             selectedDirs[selectedDir]["mean"]["ring3"].append(arr[17])
#             selectedDirs[selectedDir]["max"]["ring1"].append(arr[18])
#             selectedDirs[selectedDir]["max"]["ring2"].append(arr[19])
#             selectedDirs[selectedDir]["max"]["ring3"].append(arr[20])
#             selectedDirs[selectedDir]["maxmean"]["ring1"].append(arr[21])
#             selectedDirs[selectedDir]["maxmean"]["ring2"].append(arr[22])
#             selectedDirs[selectedDir]["maxmean"]["ring3"].append(arr[23])
#             selectedDirs[selectedDir]["min"]["ring1"].append(arr[24])
#             selectedDirs[selectedDir]["min"]["ring2"].append(arr[25])
#             selectedDirs[selectedDir]["min"]["ring3"].append(arr[26])
#             selectedDirs[selectedDir]["minmean"]["ring1"].append(arr[27])
#             selectedDirs[selectedDir]["minmean"]["ring2"].append(arr[28])
#             selectedDirs[selectedDir]["minmean"]["ring3"].append(arr[29])
            
#             selectedDirs[selectedDir]["mean"]["ring4"].append(arr[30])
#             selectedDirs[selectedDir]["max"]["ring4"].append(arr[31])
#             selectedDirs[selectedDir]["maxmean"]["ring4"].append(arr[32])
#             selectedDirs[selectedDir]["min"]["ring4"].append(arr[33])
#             selectedDirs[selectedDir]["minmean"]["ring4"].append(arr[34])        
#         myLog.logging.info(f'Finished tests over datasets')

#     modelName = next(iter(selectedDirs.keys()))
#     saveJSON(selectedDirs, os.path.join(PATH, "AfterPlots", "errors"), f'errorsPerDS-tr-{modelName}.json')
#     myLog.logging.info(f'JSON object saved')
    

    myLog.logging.info(f'Generating Sample')
    dsName = DATASETNAME    #args.dataset #"19SourcesRdm"
    plotName = f'Model-{dir}_DS-{dsName}_sample.png'
    os.listdir(os.path.join(PATH, "Dict", dir))[0]
    dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
    diffSolve = selectNN(dict)
    trainloader, testloader = generateDatasets(PATH, datasetName=dsName ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.0, s=256, transformation=dict["transformation"], val=False).getDataLoaders()
    try:
        ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict, tag='Best')
    except:
        ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
#     ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
    theModel.eval();
    plotSampRelative(theModel, testloader, dict, device, PATH, plotName, maxvalue=0.5, power=2.0)
    myLog.logging.info(f'Sample generated')
##################  
    

#     os.listdir(os.path.join(PATH, "Dict", dir))[0]
#     dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
#     diffSolve = DiffSur().to(device)
#     ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
#     theModel.eval();
#     try:
#         print(dict["transformation"])
#         trainloader, testloader = generateDatasets(PATH, datasetName=dsName, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512, transformation=dict["transformation"]).getDataLoaders()
#     except:
#         trainloader, testloader = generateDatasets(PATH, datasetName=dsName, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.1, s=512).getDataLoaders()
    myLog.logging.info(f'Generating Error grid...')
    xi, yi, zi = predVsTarget(testloader, theModel, device, transformation = dict["transformation"], threshold = 0.0, nbins = 100, BATCH_SIZE = BATCH_SIZE, size = 256, lim = 10)
    dataname = f'Model-{dir}_DS-{dsName}.txt'
    np.savetxt(os.path.join(PATH, "AfterPlots", "Pred", dataname), zi.reshape(100,100).transpose())

    power = 1/8
    plotName = f'Model-{dir}_DS-{dsName}_pow-{power}.png'
    plt.figure(2)
    plt.pcolormesh(xi, yi, np.power(zi.reshape(xi.shape) / zi.reshape(xi.shape).max(),1/8), shading='auto')
    plt.plot([0,1],[0,1], c='r', lw=0.2)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title(f'Model {dir},\nDataset {dsName}')
    plt.colorbar()
    plt.savefig(os.path.join(PATH, "AfterPlots", "Pred", plotName), transparent=False)
    plt.show()
    myLog.logging.info(f'Error grid generated')
    
############
    myLog.logging.info(f'Generating Error Vs Boundary Band')
#     dsName = DATASETNAME   #args.dataset #"19SourcesRdm"
#     trainloader, testloader = generateDatasets(PATH, datasetName=dsName ,batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=0.0, s=256, transformation=dict["transformation"], val=False).getDataLoaders()
#     dir = args.dir
#     os.listdir(os.path.join(PATH, "Dict", dir))[0]
#     dict = inOut().loadDict(os.path.join(PATH, "Dict", dir, os.listdir(os.path.join(PATH, "Dict", dir))[0]))
    theModel = selectNN(dict)
    try:
            ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict, tag='Best')
    except:
        ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
#     ep,err, theModel = inOut().load_model(diffSolve, "Diff", dict)
    theModel.eval();
    dd = errOverLat(theModel, testloader, device, transformation="linear", error_fnc=nn.MSELoss(reduction='none'))
    saveJSON(dd, os.path.join(PATH, "AfterPlots", "errorsVsBoundaryBand"), f'errorVsBB-{dir}.json')
    myLog.logging.info(f'JSON object saved')


#How to run
# python -W ignore tests.py --dir 1DGX