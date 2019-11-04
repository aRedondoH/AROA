import sys
import os
import subprocess
import math
from itertools import product
from csv import writer

def expCombinations():
    # Experiment parameters (Check different configurations)
    varBeta =  [0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39]  # variance of the beta distribution
    K = [1000] # K simulations
    nOri = [600] # number of originals to generate
    nAtt = [20] # number of attacks to generate
    feaToCheck = [0] # number of features (from the originals generated) to change from 1 to 0
    feaToAttack = [370,372,374,376,378,380,400] # number of features to attack per binary to change the features from 0 to 1
    nTriesFindAttack = [5] # number of tries to find the attack from the binary obtained from test set
    numCores = [16] # number of cores to be used in the cluster

    grid = product(varBeta,K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack,numCores)

    for idx,params in enumerate(grid):
        # Output csv
        output = "pathTo/../Output/expCom29.csv"
        if not os.path.isfile(output):
            columns=['expNum','varBeta','K','nOri','nAtt','feaToCheck','feaToAttack','nTriesFindAttack','nbAcu','nbEr','tpNB','tnNB','fpNB','fnNB','fprNB','fnrNB','aroaAcu','aroaEr','tpAroa','tnAroa','fpAroa','fnAroa','fprAROA','fnrAROA','time']
            with open(output, 'a') as f:
                fw = writer(f)
                fw.writerow(columns)
                f.close()
        command = ['qsub', 'exp.sub'] +[str(idx)]+[ str(param) for param in params ] + [output]
        # qsub exp.sub 44 0.5 1e-06 1000 3000 100 100 16 /LUSTRE/users/aredondo/AROA_MC_RealData/Output/exp1.csv 
        print(command)
        output = subprocess.Popen(command,stdout=subprocess.PIPE).stdout.read()
        print("Output: ",output)
    
def expSpecific():
    # Run specific configuration (conf paper: varBeta = 0.25, K = 700, nOri = 300, nAtt = 20, feaToCheck = 0, feaToAttack = 320, numCores = 16)
    varBeta = 0.25 # variance of the beta distribution
    K = 700 # K simulations
    nOri = 300 # number of originals to generate 
    nAtt = 20 # number of attacks to generate
    feaToCheck = 0 # number of features (from the originals generated) to change from 0 to 1
    feaToAttack = 320 # number of features to attack per binary to change the features from 0 to 1
    nTriesFindAttack = 5 # number of tries to find the attack from the binary obtained from test set
    numCores = 16 # number of cores to be used in the cluster

    numberOfExperiments = 1000
    
    for i in range(numberOfExperiments):
        print(i)
        # Output csv
        output = "pathTo/../Output/expSpe_39.csv"
        if not os.path.isfile(output):
            # columns=['expNum','varBeta','K','nOri','nAtt','feaToCheck','feaToAttack','nTriesFindAttack','nbAcu','nbEr','tpNB','tnNB','fpNB','fnNB','fprNB','fnrNB','aroaAcu','aroaEr','tpAroa','tnAroa','fpAroa','fnAroa','fprAROA','fnrAROA','time']
            columns=['expNum','varBeta','K','nOri','nAtt','feaToCheck','feaToAttack','nTriesFindAttack','euNB','euNBMax','euAROA','euAROAmax','time']
            with open(output, 'a') as f:
                fw = writer(f)
                fw.writerow(columns)
                f.close()
        command = ['qsub', 'exp.sub'] +[str(i)] +[str(varBeta),str(K),str(nOri),str(nAtt),str(feaToCheck),str(feaToAttack),str(nTriesFindAttack),str(numCores)] + [output]
        print(command)
        output = subprocess.Popen(command,stdout=subprocess.PIPE).stdout.read()
        print("Output: ",output)

# expCombinations()
expSpecific()
