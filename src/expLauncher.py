import sys
import os
import subprocess
import math
from itertools import product
from csv import writer

def expCombinations():

    # Experiment parameters (Check different configurations)
    varBeta =  [0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39]  
    K = [1000] # K simulations (fix by the moment)
    nOri = [600]
    nAtt = [20]
    feaToCheck = [0]
    feaToAttack = [370,372,374,376,378,380,400]
    nTriesFindAttack = [5] # (fix by the moment)
    numCores = [16]

    grid = product(varBeta,K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack,numCores)

    for idx,params in enumerate(grid):
        # Output csv
        output = "/LUSTRE/users/aredondo/AROA_MC_Binary/Output/expCom29.csv"
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
    varBeta = 0.25 # 
    K = 700 # K simulations
    nOri = 300
    nAtt = 20
    feaToCheck = 0
    feaToAttack = 320
    nTriesFindAttack = 5 # 5 and 9 give good results
    numCores = 16

    numberOfExperiments = 1000
    
    for i in range(numberOfExperiments):
        print(i)
        # Output csv
        output = "/LUSTRE/users/aredondo/AROA_MC_Binary/Output/expSpe_39.csv"
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