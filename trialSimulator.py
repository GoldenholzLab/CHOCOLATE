import numpy as np
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
from realSim import simulator_base,downsample
from joblib import Parallel, delayed
import scipy.stats as stats

def RR50(PC):
    return 100*np.mean(PC>=50)

def MPC(PC):
    return np.median(PC)

def getPC(trialData,baseline,test):
    B = np.sum(trialData[:,:baseline],axis=1) / (baseline/28)
    T = np.sum(trialData[:,baseline:],axis=1) / (test/28)
    PC = 100*(B-T)/B
    return PC

def runSet(howMany,Lparams,minSz=8,N=200,DRG=.3,PCB=0,baseline=56,test=84,CPUs=9):
    fname='Fig5-RCT.tiff'
    if CPUs==1:
        X = [run1trial(minSz,N,DRG,PCB,baseline,test,Lparams) for _ in trange(howMany)]
    else:
        with Parallel(n_jobs=CPUs, verbose=False) as par:
            X = par(delayed(run1trial)(minSz,N,DRG,PCB,baseline,test,Lparams) for _ in trange(howMany))
    X2 = np.array(X,dtype=float)
    plt.boxplot(X2)
    plt.xticks([1,2,3,4],['RR50_placebo','RR50_drug','MPC_placebo','MPC_drug'])
    plt.savefig(fname,dpi=600)
    plt.show()
    for silly in range(4):
        y = X2[:,silly]
        print(f'X[{silly}]: mean = {np.mean(y):4.4} median = {np.median(y):4.4} std dev={np.std(y):4.4}')
    return X2
    
def run1trial(minSz,N,DRG,PCB,baseline,test,Lparams):
    trialData = makeTrial(minSz,N,DRG,PCB,baseline,test,Lparams)
    PC = getPC(trialData,baseline,test)
    nover2 = int(N/2)
    PC_pcb = PC[:nover2]
    PC_drg = PC[nover2:]
    rr50_pcb = RR50(PC_pcb)
    rr50_drg = RR50(PC_drg)
    mpc_pcb = MPC(PC_pcb)
    mpc_drg = MPC(PC_drg)
    return [rr50_pcb,rr50_drg,mpc_pcb,mpc_drg]

def makeTrial(minSz,N,DRG,PCB,baseline,test,Lparams):
    dur = baseline+test

    trialData = np.zeros((N,dur))
    for pt in range(N):
        temp = makeOnePt(minSz,dur,baseline,Lparams)
        temp = applyDrug(PCB,temp,baseline)
        if pt>=(N/2):
            temp = applyDrug(DRG,temp,baseline)
        trialData[pt,:] = temp
    
    return trialData

def makeTrial_defaults(minSz,N,DRG,PCB,baseline,test):
    dur = baseline+test

    trialData = np.zeros((N,dur))
    for pt in range(N):
        temp = makeOnePt_defaults(minSz,dur,baseline)
        temp = applyDrug(PCB,temp,baseline)
        if pt>=(N/2):
            temp = applyDrug(DRG,temp,baseline)
        trialData[pt,:] = temp
    
    return trialData


def applyDrug(efficacy,x,baseline):
    # INPUTS:
    #  efficacy = fraction of seziures removed
    #  x = diary
    #  baseline = number of samples to consider as baseline samples
    #     that do not get drug applied at all
    
    L = len(x)
    allS = np.sum(x[baseline:])
    deleter = np.random.random(int(allS))<efficacy
    x2 = x.copy()
    counter=0
    for iter in range(baseline,len(x)):
        for sCount in range(int(x[iter])):
            x2[iter] -= deleter[counter]
            counter += 1
    return x2
        
def makeOnePt(minSz,dur,baseline,Lparams):
    sampFREQ = 24
    notDone = True
    while (notDone==True):
        x = simulator_base(sampFREQ,dur,Lparams=Lparams)
        x2 = downsample(x,sampFREQ)
        if sum(x2[0:baseline])>=minSz:
            notDone = False
    return x2

def makeOnePt_defaults(minSz,dur,baseline):
    sampFREQ = 24
    notDone = True
    while (notDone==True):
        x = simulator_base(sampFREQ,dur)
        x2 = downsample(x,sampFREQ)
        if sum(x2[0:baseline])>=minSz:
            notDone = False
    return x2


def calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                   drug_arm_percent_changes):

    num_placebo_arm_responders     = np.sum(placebo_arm_percent_changes > 50)
    num_drug_arm_responders        = np.sum(drug_arm_percent_changes    > 50)
    num_placebo_arm_non_responders = len(placebo_arm_percent_changes) - num_placebo_arm_responders
    num_drug_arm_non_responders    = len(drug_arm_percent_changes)    - num_drug_arm_responders

    table = np.array([[num_placebo_arm_responders, num_placebo_arm_non_responders], [num_drug_arm_responders, num_drug_arm_non_responders]])

    [_, RR50_p_value] = stats.fisher_exact(table)

    return RR50_p_value

def calculate_MPC_p_value(placebo_arm_percent_changes,
                                     drug_arm_percent_changes):

    # Mann_Whitney_U test
    [_, MPC_p_value] = stats.ranksums(placebo_arm_percent_changes, drug_arm_percent_changes)

    return MPC_p_value