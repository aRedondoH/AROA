from aroa import *
import sys
from csv import writer

def extractConfMatrix(y_test,y_pred):
    cmNB = confusion_matrix(y_test,y_pred)
    tnNB = cmNB[0][0]
    fnNB = cmNB[1][0]
    tpNB = cmNB[1][1]
    fpNB = cmNB[0][1]
    return tnNB, fnNB, tpNB, fpNB

def getData(numFea):
    benignDf = pd.read_csv("../Data/staDynBenignBinary.csv", index_col=0)
    customCol = top1000columns # [0:numFea]
    customCol.append('label')
    benignDf = benignDf[customCol] # only for testing
    malwareDf = pd.read_csv("../Data/staDynVt3000Binary.csv")
    malwareDf = malwareDf[customCol]
    cutDf = np.random.rand(len(malwareDf)) < ((benignDf.shape[0]*100)/len(malwareDf))/100 # Similar malware files to benign
    malwareCutDf = malwareDf[cutDf]
    
    dataDf = benignDf.append(malwareCutDf, ignore_index=True) # carefull if both do not have the same shape
    X_data, y_data= dataDf.iloc[:,:-1],dataDf.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.4, random_state=2)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_train = pd.to_numeric(y_train) # Convert dtype from object to numeric (int64)
    y_test = y_test.values
    y_test = pd.to_numeric(y_test) # Convert dtype from object to numeric (int64)
    
    clfNB = BernoulliNB()
    clfNB.fit(X_train,y_train) # Fit NB with X_train untainted

    benignDf = pd.read_csv("../Data/staDynBenignBinary.csv", index_col=0)
    benignDf = benignDf[customCol] # only for testing
    obfuMalDf = pd.read_csv("../Data/staDynVt3000BinaryObfuscated.csv",index_col=0)
    obfuMalDf = obfuMalDf[customCol]
    sample_malware_df = obfuMalDf.sample(len(benignDf))
    dataObfuDf = pd.concat([sample_malware_df, benignDf]).sample(frac=1)
    X_dataObfu, y_dataObfu= dataObfuDf.iloc[:,:-1],dataObfuDf.iloc[:,-1]
    X_dataObfu = X_dataObfu.values
    return X_dataObfu, y_dataObfu, clfNB

def getUti(yPred,ytest):
    uTP=1    # utility True Positives 
    uFP=0    # utility False Positives
    uFN=-5    # utility False Negatives
    uTN=1   # utility True Negatives
    utiExp = 0
    for i in range(len(ytest)):
        if yPred[i:i+1][0]==1 and ytest[i:i+1][0] ==1:
            utiExp += uTP
        if yPred[i:i+1][0]==1 and ytest[i:i+1][0] == 0:
            utiExp += uFP
        if yPred[i:i+1][0]==0 and ytest[i:i+1][0] == 1:
            utiExp += uFN
        if yPred[i:i+1][0]==0 and ytest[i:i+1][0] == 0:
            utiExp += uTN
    return utiExp/len(ytest)

def exp(ite,varBeta,K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack,numCores,output):
    uTP=1    # utility True Positives 
    uFP=0    # utility False Positives
    uFN=-5    # utility False Negatives
    uTN=1   # utility True Negatives
    utMat = np.array([[uTP,uFP],[uFN,uTN]]) # Classifier utility matrix

    # expDf = pd.DataFrame(columns=['expNum','varBeta','K','nOri','nAtt','feaToCheck','feaToAttack','nTriesFindAttack','nbAcu','nbEr','tpNB','tnNB','fpNB','fnNB','fprNB','fnrNB','aroaAcu','aroaEr','tpAroa','tnAroa','fpAroa','fnAroa','fprAROA','fnrAROA','time'])
    expDf = pd.DataFrame(columns=['expNum','varBeta','K','nOri','nAtt','feaToCheck','feaToAttack','nTriesFindAttack','euNB','euNBMax','euAROA','euAROAmax','time'])
    

    print("Utility classifier: uTP:",uTP, " uFP: ",uFP," uFN:", uFN," uTN:",uTN)
    print("expNum:",ite,'varBeta:',varBeta,'K:',K,'nOri:',nOri,'nAtt:',nAtt," feaToCheck: ",feaToCheck, " feaToAttack:", feaToAttack," nTriesFindAttack:",nTriesFindAttack,'numCores:', numCores,'output:', output)

    start = time.time() # Start timer
    numFea = 262 # this is manual for the moment (best accuracy in 13 number of features), 262 gets 0.85 acu
    X_test,y_test,clfNB = getData(numFea)

    yPred = clfNB.predict(X_test)
    y_test = y_test.values
    print("yPred:",type(yPred), len(yPred),yPred.shape)
    print("y_test:",type(y_test), len(y_test),y_test.shape)
    NBeu = getUti(yPred,y_test)
    print("NB: ", NBeu)

    posteriorAroa = getAttacksDistPar(X_test, clfNB, varBeta, numCores, K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack)
    # print("posteior: ", posteriorAroa)
    # y_Aroa = getAroaLabelFromPosterior(posteriorAroa, utMat)
    yAROABin = np.argmax(posteriorAroa, axis=1)
    print("yAROABin", yAROABin)
    AROAeu = getUti(yAROABin,y_test)

    
    # y_Aroa = getAroaLabelFromPosterior(posteriorAroa, utMat)
    # aroaAcNB = round(accuracy_score(y_test, y_Aroa),3)
    # aroaErNB = round(mean_squared_error(y_test, y_Aroa),3)
    # tnAroa, fnAroa, tpAroa, fpAroa = extractConfMatrix(y_test,y_Aroa)

    # fprAROA = round(fpAroa/(fpAroa+tnAroa),3)
    # fnrAROA = round(fnAroa/(fnAroa+tpAroa),3)

     #print("AC NB: ", acNB, " err: ", erNB," tp:",tpNB, " tn:", tnNB, " fp:",fpNB," fn:",fnNB, "fpr:",fprNB, " fnr:", fnrNB)
    # print("AC AROA: ", aroaAcNB, " err: ", aroaErNB," tp:",tpAroa, " tn:", tnAroa, " fp:",fpAroa," fn:",fnAroa, "fpr:", fprAROA, "fnr: ", fnrAROA)

    end = time.time() # End timer
    timeLast = (end-start)
    timeLastFor = time.strftime("%H:%M:%S", time.gmtime(timeLast))
    print("Time: ", timeLastFor)

    vec = np.array([ite,varBeta,K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack,NBeu,0,AROAeu,0,timeLastFor])

    # vec = np.array([ite,varBeta,K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack,sumEuNB,maxEuNB,sumEuAROA,maxEuAROA,timeLastFor])

    # vec = np.array([ite,varBeta,K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack,acNB,erNB,tpNB, tnNB,fpNB,fnNB,fprNB,fnrNB,aroaAcNB,aroaErNB,tpAroa, tnAroa,fpAroa,fnAroa,fprAROA,fnrAROA,timeLastFor])

    with open(output, 'a') as f:
        fw = writer(f)
        fw.writerow(vec)
        f.close()

def main():

    # Get parameters ( 1(index) +7(parameters) +1(output) )
    # Index
    ite = int(sys.argv[1])
    # Parameters
    varBeta = float(sys.argv[2])
    K = int(sys.argv[3])
    nOri = int(sys.argv[4])
    nAtt = int(sys.argv[5])
    feaToCheck = int(sys.argv[6])
    feaToAttack = int(sys.argv[7])
    nTriesFindAttack = int(sys.argv[8])
    numCores = int(sys.argv[9])
    # Output
    output = str(sys.argv[10])

    print("ite: ",ite,"varBeta:",varBeta,"K:",K,"nOri:",nOri,"nAtt:",nAtt,"feaToCheck:",feaToCheck,"feaToAttack:",feaToAttack," nTriesFindAttack:",nTriesFindAttack,"numCores:",numCores,"output:",output)

    exp(ite,varBeta,K,nOri,nAtt,feaToCheck,feaToAttack,nTriesFindAttack,numCores,output)
    
main()
