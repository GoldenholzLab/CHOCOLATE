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

def runSet(howMany,Lparams,minSz=8,N=200,DRG=.3,PCB=0,baseline=56,test=84,CPUs=9,fname='Fig5-RCT-v2.tiff'):
    ## Compute all the RCTs first
    if CPUs==1:
        X = [run1trial(minSz,N,DRG,PCB,baseline,test,Lparams) for _ in trange(howMany)]
    else:
        with Parallel(n_jobs=CPUs, verbose=False) as par:
            X = par(delayed(run1trial)(minSz,N,DRG,PCB,baseline,test,Lparams) for _ in trange(howMany))
    X2 = np.array(X,dtype=float)

    # Now make a nice picture
    #fname='Fig5-RCT-v2.tiff'    
    decimal_round = 0
    # these are the endpoint response statistics from the meta-analysis of 23 RCTs
    # see Romero et al 2020 for details of that study
    historical_placebo_RR50_mean = 21.1
    historical_placebo_RR50_std = 9.9
    historical_placebo_MPC_mean = 16.7
    historical_placebo_MPC_std = 10.3
    historical_drug_RR50_mean = 43.2
    historical_drug_RR50_std = 13.1
    historical_drug_MPC_mean = 40.9
    historical_drug_MPC_std = 11.0

    # these come from above, X2, freshly simulated
    simulated_placebo_RR50_mean = np.mean(X2[:,0])
    simulated_drug_RR50_mean = np.mean(X2[:,1])
    simulated_placebo_MPC_mean = np.mean(X2[:,2])
    simulated_drug_MPC_mean = np.mean(X2[:,3])
    simulated_placebo_RR50_std = np.std(X2[:,0])
    simulated_drug_RR50_std = np.std(X2[:,1])
    simulated_placebo_MPC_std = np.std(X2[:,2])
    simulated_drug_MPC_std = np.std(X2[:,3])

    y = [1, 2, 3, 4]

    historical_data = [historical_drug_MPC_mean, historical_placebo_MPC_mean,
                       historical_drug_RR50_mean, historical_placebo_RR50_mean]
    
    simulated_data = [simulated_drug_MPC_mean, simulated_placebo_MPC_mean,
                      simulated_drug_RR50_mean, simulated_placebo_RR50_mean]
    
    historical_std_bars = [historical_drug_MPC_std, historical_placebo_MPC_std,
                           historical_drug_RR50_std, historical_placebo_RR50_std]
    
    simulated_std_bars = [simulated_drug_MPC_std, simulated_placebo_MPC_std,
                          simulated_drug_RR50_std, simulated_placebo_RR50_std]
    
    xlabels = ['drug MPC', 'placebo MPC', 'drug RR50', 'placebo RR50']
    ylabel = 'Response percentage'
    #ylabels = ['drug arm, MPC', 'placebo arm, MPC', 'drug arm, RR50', 'placebo arm, RR50']
    #xlabel = 'Response percentage'
    
    title = 'Efficacy endpoints over simulated and historical RCTs'
    
    separation = 0.2
    height_const = 0.4
    heights = height_const*np.ones(4)
    
    [fig, ax] = plt.subplots(figsize = (6,4))
    
    #simulated_rects = ax.barh(np.array(y) + separation, simulated_data, height = heights, xerr=simulated_std_bars, label = 'simulated', color='dimgrey')
    #historical_rects = ax.barh(np.array(y) - separation, historical_data, height = heights, xerr=historical_std_bars, label = 'historical', color='silver')
    simulated_rects = ax.bar(np.array(y) + separation, simulated_data,  width= height_const, yerr=simulated_std_bars, label = 'simulated', color='dimgrey')
    historical_rects = ax.bar(np.array(y) - separation, historical_data, width=height_const, yerr=historical_std_bars, label = 'historical', color='silver')
    
    i = 0
    for rect in simulated_rects:
        text_width = simulated_data[i] + simulated_std_bars[i]
        simumlated_data_string = ('{0:.' + str(decimal_round) + 'f}').format(simulated_data[i])
        simumlated_std_string = ('{0:.' + str(decimal_round) + 'f}').format(simulated_std_bars[i])
        #plt.text(1.05*text_width, rect.get_y() + 0.25*rect.get_height(), simumlated_data_string + ' $\pm$ ' + simumlated_std_string)
        plt.text( rect.get_x() + 0.25*rect.get_width(),1.05*text_width, simumlated_data_string + ' $\pm$ ' + simumlated_std_string)
        
        i += 1
    i = 0
    for rect in historical_rects:
        text_width = historical_data[i] + historical_std_bars[i]
        historical_data_string = ('{0:.' + str(decimal_round) + 'f}').format(historical_data[i])
        historical_std_string = ('{0:.' + str(decimal_round) + 'f}').format(historical_std_bars[i])
        #plt.text(1.05*text_width, rect.get_y() + 0.25*rect.get_height(), str(historical_data_string) + ' $\pm$ ' + historical_std_string)
        plt.text( rect.get_x() + 0.25*rect.get_width(), 1.05*text_width, str(historical_data_string) + ' $\pm$ ' + historical_std_string)
        i += 1
    
    #plt.yticks(y, ylabels, rotation='horizontal')
    plt.xticks(y, xlabels, rotation='horizontal')
    #plt.xlim([0, 100])
    plt.ylim([0,100])
    plt.ylabel(ylabel)
    #plt.xlabel(xlabel)
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    fig.savefig(fname, dpi = 600, bbox_inches = 'tight')
    plt.show()


    
    
    #plt.boxplot(X2)
    #plt.xticks([1,2,3,4],['RR50_placebo','RR50_drug','MPC_placebo','MPC_drug'])
    #plt.savefig(fname,dpi=600)
    #plt.show()
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