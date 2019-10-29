import multiprocessing
import sys
import os
import time
import pandas as pd
import numpy as np
import random as rd
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from csv import writer
from features import *
import itertools


## AROA functions
def getPriors(clf):
    """Obtain priors

    clf -- classifier (NB)
    """
    return np.exp(clf.class_log_prior_)
def getLikelihood(X, clf):
    """Obtain likelihood (xposterior)

    I.e. ``exp(log P(c) + log P(x|c))`` for all rows x of X, as an array-like of
        shape [n_classes, n_samples].

    X -- dataset
    clf -- classifier (NB)

    """
    return np.exp(clf._joint_log_likelihood(X))

# def getOriginalsMC(binary, n, numberOfFeatures,feaToCheck):
#     """Get possible originals 

#     binary -- binary to obtain the possible originals
#     n -- number of possible originals to generate
#     numberOfFeatures -- range of features to create the originals
#     feaToCheck -- number of features to check in the binary
#     """
#     possibleOri = np.empty((0,binary.shape[0]), int)
#     for i in range(n):
#         binCopy = np.copy(binary)
#         indexVector = np.where(binary == 1)[0]
#         if len(indexVector) >= 1:
#             for p in range(feaToCheck):
#                 randFea = rd.choice(indexVector)
#                 binCopy[randFea:randFea+1] = 0    
#         possibleOri = np.append(possibleOri, [binCopy], axis=0)
#     return possibleOri

def getOriginalsMC(binary,n,feaToCheck):
    """Get possible originals 

    binary -- binary to obtain the possible originals
    n -- number of possible originals to generate
    feaToCheck -- number of features to check in the binary
    """
    possibleOri = np.empty((0,binary.shape[0]), int)
    for i in range(n):
        binCopy = np.copy(binary)
        indexVector = np.where(binary == 1)[0]
        if len(indexVector) >= 1:
            randomIndex = np.random.choice(indexVector,feaToCheck,replace=False)
            binCopy[randomIndex]=0
        possibleOri = np.append(possibleOri, [binCopy], axis=0)
    return possibleOri

def adversarialUtilities(yc, y):
    """Alan's random utilities

    yc -- Cleo clasification binary label (Malware=1, Benign=0)
    y -- binary label (Malware=1, Benign=0)
    """
    delta0 = 0
    delta1 = 1  # Good for Alan values should be in range [0-1]
    # Alan is not interested on benign binaries (yc=M/B, y=B)
    if (y == 0):
        utility = delta0
    # Cleo predicts that a binary is malware and it is malware (yc=M, y=M)
    if (yc == 1) and (y == 1):
        utility = delta0
    # Alan fools Cleo and make her predict a malware as benign (yc=B, y=M)
    if (yc == 0) and (y == 1):
        utility = delta1
    return utility

# def getAttacksMC(binary, n,numberOfFeatures,featuresToAttack):
#     """Get possible attacks 

#     binary -- binary to attack
#     n -- number of possible attacks to generate
#     numberOfFeatures -- range of features to generate the attack
#     featuresToAttack -- number of features to attack per binary
#     """
#     possibleAttacks = np.empty((0, binary.shape[0]), int)
#     for i in range(n):
#         binCopy = np.copy(binary)
#         indexVector = np.where(binary == 0)[0]
#         if len(indexVector) >= 1:
#             for p in range(featuresToAttack):
#                 randFea = rd.choice(indexVector)
#                 binCopy[randFea:randFea+1] = 1
#         possibleAttacks = np.append(possibleAttacks,[binCopy],axis=0)
#     return possibleAttacks

def getAttacksMC(binary, n,featuresToAttack):
    """Get possible attacks 

    binary -- binary to attack
    n -- number of possible attacks to generate
    featuresToAttack -- number of features to attack per binary
    """
    possibleAttacks = np.empty((0, binary.shape[0]), int)
    for i in range(n):
        binCopy = np.copy(binary)
        indexVector = np.where(binary == 0)[0]
        if len(indexVector) >= 1:
            randomIndex = np.random.choice(indexVector,featuresToAttack,replace=False)
            binCopy[randomIndex]=1
        possibleAttacks = np.append(possibleAttacks,[binCopy],axis=0)
    return possibleAttacks
    
def getR(possibleAttacks, clf):
    """Get R parameter (mean) for calculating deltas 
    needed to calculate the Alan random probability of attack

    possibleAttacks -- binaries (2 dimension array)
    clf -- NB classifier
    possibleAttacks -- possible attacks that Alan could take
    """
    r = clf.predict_proba(possibleAttacks)[:, 1]
    r[r == 1.0] -= 0.0001
    return r

def getDeltas(meanBeta, varBeta):
    """Get deltas to calculate the beta distribution to obtain 
    the Alan random attack probability

    meanBeta -- mean of the beta distribution
    varBeta -- variance of the beta distribution
    """
    deltas = np.zeros((len(meanBeta), 2))
    for i in range(len(meanBeta)):
        s2 = varBeta * meanBeta[i] * min(meanBeta[i] * (1.0 - meanBeta[i]) / (1.0 + meanBeta[i]),
                                         (1.0 - meanBeta[i])**2 / (2.0 - meanBeta[i]))  # proportion of maximum
        # variance of convex beta
        deltas[i][0] = ((1.0 - meanBeta[i]) / s2 - 1.0 /
                        meanBeta[i]) * meanBeta[i]**2
        deltas[i][1] = deltas[i][0] * (1.0/meanBeta[i] - 1.0)
    return deltas

def randomProb(deltas):
    """Get Alan random probability of attack a malware binary

    deltas -- beta distribution parameters (delta1,delta2)
    """
    return np.random.beta(deltas[:, 0], deltas[:, 1])

# def disProbableAttacks(possibleOriginal, binFromTest, clf, var, K, nAtt,feaToAttack):
#     """AROA simulation of Alan's distribution of probable attacks

#     possibleOriginal -- a possible original obtained selecting from test set
#     binFromTest -- binary from test set
#     clf -- NB classifier
#     d -- Alan's utility when fools Cleo d=[0,1]
#     var -- variance
#     K -- number of simulations to perform
#     nAtt -- number of possible attacks to generate
#     feaToAttack -- number of features to attack
#     """
#     counter = 0
#     found = 0
#     while (found == 0):
#         if counter == 5:
#             indexAttack = 0
#             break
#         # print("NumberOfIterations: ", counter)
#         possibleAttacks = getAttacksMC(possibleOriginal, nAtt,feaToAttack)
#         # print(possibleAttacks)
#         for i in range(possibleAttacks.shape[0]):
#             if np.mean(np.in1d(binFromTest, possibleAttacks[i:i+1])) == 1:
#                 # print("i: ", i)
#                 indexAttack = i
#                 # print(binFromTest)
#                 # print(possibleAttacks[i:i+1])
#                 found = 1
#                 break
#         counter += 1
#     # print("indexFound: ", indexAttack)
#     # Obtain the distribution by simulation
#     # here we will store the number of times each attack is maximum
#     distribution = np.zeros(len(possibleAttacks))
#     # print("Distribution: ")
#     # print(distribution)
#     deltas = getDeltas(getR(possibleAttacks, clf), var)
#     for i in range(K):
#         randProb = randomProb(deltas)
#         # print("randProb: ", randProb)
#         # expUti = randProb * adversarialUtilities(1,1) + (1.0 - randProb) * adversarialUtilities(0,1)
#         expUti = 1.0 - randProb
#         distribution[np.argmax(expUti)] += 1
#         # print("EU in dis[indexAttack]: ", distribution[indexAttack])
#     # print("DisAfter: ")
#     # print(list(distribution))
#     # print("index, dis[index],probAttack ", indexAttack, distribution[indexAttack], distribution[indexAttack]/K)
#     if found == 0:
#         probAttack = 0.0  # not attack found
#     else:
#         # probAttack = sum(distribution[indexAttack])/K # sum to obtain 0.0 in case of distribution[indexAttack]) == []
#         probAttack = distribution[indexAttack]/K
#     # print("ProbAttack: ", probAttack)
#     return probAttack, possibleAttacks[indexAttack]

def disProbableAttacks(possibleOriginal, binFromTest, clf, var, K, nAtt,feaToAttack,nTriesFindAttack):
    """AROA simulation of Alan's distribution of probable attacks

    possibleOriginal -- a possible original obtained selecting from test set
    binFromTest -- binary from test set
    clf -- NB classifier
    d -- Alan's utility when fools Cleo d=[0,1]
    var -- variance
    K -- number of simulations to perform
    nAtt -- number of possible attacks to generate
    feaToAttack -- number of features to attack
    nTriesFindAttack -- number of tries to find the possible attack (similar from comming from test set)
    """
    triesCounter = 0
    found = 0
    while (found == 0):
        if triesCounter == nTriesFindAttack:
            indexAttack = 0
            break
        # print("NumberOfIterations: ", triesCounter)
        possibleAttacks = getAttacksMC(possibleOriginal, nAtt,feaToAttack)
        # print(possibleAttacks)
        for i in range(possibleAttacks.shape[0]):
            if np.mean(np.in1d(binFromTest, possibleAttacks[i:i+1])) == 1:
                # print("i: ", i)
                indexAttack = i
                # print(binFromTest)
                # print(possibleAttacks[i:i+1])
                found = 1
                break
        triesCounter += 1
    # print("indexFound: ", indexAttack)
    # Obtain the distribution by simulation
    # here we will store the number of times each attack is maximum
    distribution = np.zeros(len(possibleAttacks))
    # print("Distribution: ")
    # print(distribution)
    deltas = getDeltas(getR(possibleAttacks, clf), var)
    for i in range(K):
        randProb = randomProb(deltas)
        # print("randProb: ", randProb)
        # expUti = randProb * adversarialUtilities(1,1) + (1.0 - randProb) * adversarialUtilities(0,1)
        expUti = 1.0 - randProb
        distribution[np.argmax(expUti)] += 1
        # print("EU in dis[indexAttack]: ", distribution[indexAttack])
    # print("DisAfter: ")
    # print(list(distribution))
    # print("index, dis[index],probAttack ", indexAttack, distribution[indexAttack], distribution[indexAttack]/K)
    if found == 0:
        probAttack = 0.0  # not attack found
        possibleAttack = binFromTest # the binary from test untainted
    else:
        # probAttack = sum(distribution[indexAttack])/K # sum to obtain 0.0 in case of distribution[indexAttack]) == []
        probAttack = distribution[indexAttack]/K
        possibleAttack = possibleAttacks[indexAttack]
    # print("ProbAttack: ", probAttack)
    return probAttack, possibleAttack


def getAroaLabelFromPosterior(vecPosterior, ut):
    """Obtain which label (benign or malware) has higher probability

    vecPosterior -- double dimension vector with posterior probability of each binary
    ut -- utility matrix
    """
    prodUtWithPosterior = np.dot(ut, vecPosterior.transpose())
    return np.argmax(prodUtWithPosterior, axis=0)

def getExpectedUtility(vecPosterior, ut):
    """Obtain the expected utility for each binary

    vecPosterior -- double dimension vector with posterior probability of each binary
    ut -- utility matrix
    """
    prodUtWithPosterior = np.dot(ut, vecPosterior.transpose())
    
    return prodUtWithPosterior


def getPosterior(binary, clf, var, K, nOri, nAtt,feaToCheck,feaToAttack,nTriesFindAttack):
    """Obtain AROA posterior distribution, a double dimension vector 
    in form v=[[probBenign, probMalware]]

    binary -- binary to obtain its posterior probabilities (it often from test set)
    clf -- NB classifier
    var -- variance
    K -- number of simulations to perform
    nOri -- number of possible originals to generate
    nAtt -- number of possible attacks to generate
    feaToCheck -- number of fea to check when generting originals
    feaToAttack -- number of fea to attack
    nTriesFindAttack -- number of tries to find the possible attack (similar from comming from test set)
    """
    possibleOriginals = getOriginalsMC(binary, nOri,feaToCheck)
    likeliMalware = 0
    for i in range(possibleOriginals.shape[0]):
        # print("possible original: ",i, possibleOriginals[i,:])
        dis, possibleAttack = disProbableAttacks(possibleOriginals[i, :], binary, clf, var, K, nAtt,feaToAttack,nTriesFindAttack)
        # print("ProbAttack: ", dis)
        likeli = getLikelihood(possibleOriginals[[i], :], clf)[0, 1]
        # print("Dis x Likeli: ", dis*likeli)
        # probMalware += dis*likeli
        # product of density functions (can be greater than 1...)
        likeliMalware += dis*likeli
        # print("ProbMal: ", np.round(likeliMalware,2), "ProbBen: ",np.round(getLikelihood([binary], clf)[0, 0],2))
        # count += disProbableAttacks(possibleOriginals[i,:],binary,clf,var, K) * getLikelihood(possibleOriginals[[i],:], clf)[0,1]
    likeliBenign = getLikelihood([binary], clf)[0, 0]
    # posteriorVector = np.array([likeliBenign,likeliMalware])
    # print("PosteriorVector: ", posteriorVector)
    print("FinMal:",np.round(likeliMalware,2),"FinBenign: ", np.round(likeliBenign,2),"\n")
    return np.array([likeliBenign, likeliMalware])

def auxAttacksDist(i, dataset, clf, var, K, nOri, nAtt,feaToCheck,feaToAttack,nTriesFindAttack):
    """Aux function to obtain the attacks distribution in parallel

    i -- binary index of X_train (Poison attacks) or X_test (Evasion attacks)
    dataset -- it can be X_train (Poison attacks) or X_test (Evasion attacks)
    clf -- NB classifier
    var -- variance
    K -- number of simulations to perform
    nOri -- number of possible originals to generate
    nAtt -- number of possible attacks to generate
    feaToCheck -- number of fea to check when generting originals
    feaToAttack -- number of fea to attack
    nTriesFindAttack -- number of tries to find the possible attack (similar from comming from test set)
    """
    return getPosterior(dataset[i, :], clf, var, K, nOri, nAtt,feaToCheck,feaToAttack,nTriesFindAttack)

def getAttacksDistPar(dataset, clf, var, numCores, K, nOri, nAtt,feaToCheck,feaToAttack,nTriesFindAttack):
    """Obtain AROA attack distribution in parallel

    dataset -- it can be X_train (Poison attacks) or X_test (Evasion attacks)
    clf -- NB classifier
    var -- variance
    num -- number of cores to run in parallel
    K -- number of simulations to perform
    nOri -- number of possible originals to generate
    nAtt -- number of possible attacks to generate
    feaToCheck -- number of fea to check when generting originals
    feaToAttack -- number of fea to attack
    nTriesFindAttack -- number of tries to find the possible attack (similar from comming from test set)
    Return -> a double dimension vector with the attack distribution for each binary

    """
    numberOfBinaries = int(dataset.shape[0])
    num_cores = numCores
    posteriorAroa = Parallel(n_jobs=num_cores)(delayed(auxAttacksDist)(i, dataset, clf, var, K, nOri, nAtt,feaToCheck,feaToAttack,nTriesFindAttack) for i in range(numberOfBinaries))

    return np.array(posteriorAroa)
