#from audioop import mul
#from cmath import phase
#from re import I, search
from cProfile import label
from operator import mod
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import signal
#from tqdm.auto import trange,tqdm
from tqdm.notebook import trange, tqdm
from realSim import simulate_diary,make_key_cycle_params
from realSim import simulator_base,get_mSF,downsample
#import pandas as pd
from scipy import stats
from matplotlib.image import NonUniformImage
import matplotlib.ticker as ticker
import multiprocessing
from functools import partial
from joblib import Parallel, delayed
import warnings
from tempfile import TemporaryFile
    
def draw_chocolate(Lparams):
    fname = 'Fig2-Example-v2.pdf'
    if 0:
        REPS = 100000
        msf = np.array([ get_mSF(-1) for _ in range(REPS) ])
        h,b = np.histogram(msf,bins=1000,range=[0,30],density = True)
        plt.plot(b[:-1],h)
        plt.title('Heterogenity of seizure rates')
        plt.xlabel('Monthly seizure rate')
        plt.ylabel('Relative frequency')
        plt.show()
    #numbdays = 60
    sampRATE = 144
    #SF = 100
    numbdays = 40
    SF = 10
    period = 10
    #clustP = [1, 3, 10, 1, 1]
    clustP = [1, 1, 7, 50, 1]
    do_clusters=False
    while do_clusters==False:
        seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_A,seizure_diary_B = simulator_base(
            sampRATE=sampRATE,number_of_days=numbdays,cyclesTF=True,clustersTF=True,
            maxlimits=True,CP=[[1/period],[1]],defaultSeizureFreq=SF,Lparams=Lparams,returnDetails=True,clusterParams=clustP)
    
    t = np.arange(numbdays*sampRATE) / sampRATE
    #plt.figure()

    M = np.max(rate)
    fig, ax = plt.subplots(4, 1,figsize=(7,5))
    ax[3].eventplot(t[seizure_diary_final>0], color='blue', linelengths = 1,alpha=.5,label='Final seizure diary')     
    ax[3].plot(t,1+t*0,'b--')
    ax[3].set_xlim([0,numbdays])
    #ax[3].axis('off')
    for pos in ['right', 'top', 'left']:
        ax[3].spines[pos].set_visible(False)
    ax[3].legend(loc='lower right')
    ax[3].set_xlabel('Time (days)')
    ax[3].get_yaxis().set_visible(False)
    ax[3].get_xaxis().set_visible(True)
    ax[2].plot(t,1.5+2*(modulated_cluster_rate/M), '.-',alpha=0.5,color = 'g',markersize=3, label = 'k[n] = modulated clustered rate')
    ax[2].eventplot(t[seizure_diary_B>0], color='blue', linelengths = 1,alpha=.5,label='Seizure diary B')     

    ax[2].set_ylim([-3,2])
    
    ax[2].set_xlim([0,numbdays])
    ax[2].axis('off')
    ax[2].legend(loc='lower right')
    #ax[2].ylabel('seizures')
    ax[2].get_yaxis().set_visible(False)
    ax[2].get_xaxis().set_visible(False)
    ax[1].plot(t,1.5+2*(modulated_rate/M),'.-',color='r', alpha=.5,markersize=3,label = 'm[n] = modulated risk rate, m[n]')
    ax[1].eventplot(t[seizure_diary_A>0], color='blue', linelengths = 1,alpha=.5,label='Seizure diary A')     

    ax[1].set_ylim([-2,2])
    ax[1].set_xlim([0,numbdays])
    ax[1].axis('off')
    ax[1].legend(loc='lower right')
    #ax[1].ylabel('seizures')
    ax[1].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    
    
    ax[0].plot(t,.1+2*rate/M,'-',color='k',markersize=2, label = 'r[n] = base risk rate')
    ax[0].scatter(t,np.zeros(len(t)),c=cycles+.5,edgecolors='none',label = 'c[n] = cycles')
    ax[0].set_xlim([0,numbdays])
    
    for pos in ['right', 'top', 'bottom', 'left']:
        ax[0].spines[pos].set_visible(False)
    #ax[0].axis('off')
    ax[0].legend(loc='upper right')
    ax[0].set_xticks(np.arange(min(t), max(t)+1, 10))
    ax[0].get_yaxis().set_visible(False)
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_ylim([-.1,1.1])
    #ax[0].xlabel('day')
    #fig.supxlabel('Time (day)')
    
    plt.savefig(fname,dpi=600)
    plt.show()




def try_out_L():
    Lparams=[1,7.4548,3.663,0.40024,3.0366,7.7389]
    SF = np.logspace(-3,1,100)
    disp = Lparams[1]*np.log10(SF) + Lparams[2]*(SF**Lparams[3]) + Lparams[4]*SF + Lparams[5]
    disp = np.maximum(Lparams[0],disp)
    plt.plot(np.log10(SF),disp,'x')
    plt.show()




def make_example(sampFREQ=1,howmanydays=300,byHowMuch=30,Lparams=[1.0051,7.4548,3.663,0.40024,3.0366,7.7389]):
    howmany = sampFREQ*howmanydays
    CP = make_key_cycle_params()
    x = simulate_diary(sampFREQ,howmany,params=[-1, True, 0.5, 1, 5, .5],returnDetails=False,cycleParams=CP,Lparams=Lparams)
    t = np.arange(len(x))/sampFREQ
    plt.plot(t,x,'x-')
    plt.show()
    xD = downsample(x,byHowMuch)
    tD = np.arange(len(xD)) / (sampFREQ/byHowMuch)
    plt.plot(tD[1]+tD,xD,'x-')
    plt.show()
    
def oneTool(bestM,bestS,sampRATE,howmany,cyclesTF,clustersTF,maxlimitsTF,Lparams):
    x = simulator_base(sampRATE,howmany,cyclesTF=cyclesTF,clustersTF=clustersTF,
        maxlimits=maxlimitsTF,defaultSeizureFreq=-1,bestM=bestM,bestS=bestS,
        Lparams=Lparams)
    sf,ssf = get_sz_freq2(x,sampRATE)
    return [sf,ssf]

    
def toolTest(bestM,bestS,REPS,sampRATE,howmany,clustersTF,cyclesTF,maxlimitsTF,Lparams):
    doFAST=True
    if doFAST==False:
        #sf = np.zeros(REPS)
        #ssf = np.zeros(REPS)
        #for iter in range(REPS):
        #   mSF = get_mSF(-1,bestM,bestS)
        #    x = simulator_base(sampRATE,howmany,cyclesTF,clustersTF,maxlimitsTF,defaultSeizureFreq=mSF,Lparams=Lparams)
        #    sf[iter],ssf[iter] = get_sz_freq(x,1/sampRATE)
        resultXs = [ oneTool(bestM,bestS,sampRATE,howmany,cyclesTF,clustersTF,maxlimitsTF,Lparams) for _ in range(REPS)]
    else:
        with Parallel(n_jobs=9, verbose=False) as par:
            resultXs = par(delayed(oneTool)(bestM,bestS,sampRATE,howmany,cyclesTF,clustersTF,maxlimitsTF,Lparams) for _ in range(REPS))
        
    bigtemp=np.array(resultXs,dtype=float)
    #bigtemp=np.reshape(np.hstack(resultXs),(REPS,2))

    sf = bigtemp[:,0].copy()
    ssf = bigtemp[:,1].copy()
    mask1 = np.isfinite(sf) & ~np.isnan(sf) & (sf>0) & (ssf>0)
    X = np.log10(sf[mask1])
    Y = np.log10(ssf[mask1])
    reg = LinearRegression().fit(np.reshape(X,(-1,1)),Y)
    slopeD= reg.coef_[0]
    #print(f'rate={sampRATE} clust={clustersTF} cycles={cyclesTF} max={maxlimitsTF} median={np.median(sf):.2} median44={np.median(sf[sf>4]):.2} slope={slopeD:.2}')
    printstr= str(f'rate={sampRATE} clust={clustersTF} cycles={cyclesTF} max={maxlimitsTF} median={np.median(sf):.2} median44={np.median(sf[sf>4]):.2} slope={slopeD:.2}')
    return np.array([np.median(sf),np.median(sf[sf>4]), slopeD]),printstr


def paramSearchGD_toolTest(clustersTF=True,sampRATE=24):
    #fullP= np.array([4.0351,15.289,9.3601,5.6703,-0.18075,1.15,6.0664,4.4518])     
    #fullP= np.array([3.7531,14.966,9.124,4.9893,-0.19354,1.4388,5.6662,4.9068])
    #fullP= np.array([3.4991,15.275,8.686,5.108,-0.13603,1.4767,6.2651,5.2385])
    #fullP= np.array([3.4991,15.275,8.686,5.108,-0.13603,1.4767,6.2651,5.2385])

    #fullP= np.array([3.4991,15.275,8.686,5.108,-0.13603,1.4767,6.2651,5.2385])
    #fullP= np.array([2.9311,14.705,9.1555,4.7926,-0.069285,1.0076,5.8966,5.1051])
    #fullP= np.array([2.9848,14.862,9.2483,4.8177,-0.047201,1.0389,5.9038,5.147])
    #fullP= np.array([2.5464,14.963,9.2485,4.7591,-0.054609,1.0597,5.9878,5.4573])
    #fullP= np.array([1.9715,15.311,9.1955,4.9103,-0.064478,0.40741,6.1654,5.5564])
    #fullP= np.array([2.0354,15.307,9.0359,4.8749,-0.093473,0.68912,6.086,5.7717])
    ### best for 24 kind
    #fullP= np.array([2.2249,15.54,8.8178,4.7,-0.065587,0.28393,5.9785,6.0694])
    ### best for 1 sample per day kind
    #fullP = np.array([1.9672,15.616,8.9173,3.8982,-0.26137,0.83451,5.7446,6.1316])
    #fullP= np.array([2.358,15.084,9.6779,4.5411,-0.28041,0.93061,5.671,5.5096])
    #fullP= np.array([1.8502,15.995,9.7759,4.5473,-0.28714,0.7735,5.759,6.0557])
    #fullP= np.array([2.1447,16.309,9.7793,4.6561,-0.28565,0.68398,6.0393,5.8578])
    #fullP= np.array([2.2805,16.364,9.8807,4.6349,-0.26809,0.87806,5.8539,5.7616])
    
    #fullP= np.array([2.4481,16.819,9.8818,5.1849,-0.31477,1.985,6.3103,6.1432])
    #fullP= np.array([2.2804,17.309,10.271,5.3229,-0.29711,1.3145,6.331,6.4449])
    
    #fullP= np.array([1.8818,17.143,9.8771,5.2799,-0.2976,1.1716,6.2377,6.3548])
    #fullP= np.array([1.7579,17.147,9.9741,5.236,-0.299,1.0926,6.2818,6.377])
    #fullP= np.array([1.6216,17.262,10.342,5.5982,-0.30353,1.2773,6.3355,6.3966])
    fullP= np.array([1.5475,17.645,10.617,5.9917,-0.3085,1.5371,6.1709,6.1455])

    print(fullP)
    #totalP = len(fullP)
    howMANYiters=1000
    bestP = fullP.copy()
    bestCost = np.inf
    for iter in trange(howMANYiters):
        fullP = bestP.copy()
        bestP1,bestCost1,costers1 = sub_paramSearch(clustersTF,sampRATE,fullP)
        if bestCost1 < bestCost:
            bestCost = bestCost1
            bestP = bestP1.copy()
            print(f'iter = {iter} bestCost = {bestCost:.5} thisSum = {costers1[0]:.3} {costers1[1]:.3} {costers1[2]:.3}')
            print(f'fullP= np.array([{bestP[0]:.5},{bestP[1]:.5},{bestP[2]:.5},{bestP[3]:.5},{bestP[4]:.5},{bestP[5]:.5},{bestP[6]:.5},{bestP[7]:.5}])')
            

def sub_paramSearch(clustersTF,sampRATE,fullP):
    bestM= 1.2267388224600906
    bestS = 1.1457004817186776
    REPS=50000

    totalP = len(fullP)
    ITERS = 10
    targets = np.array([2.7,8.5,.75])
    weights = np.array([1,1,1000])
    bestCost = np.inf
    etta = .1

    etta = 0.01
    bestCost = np.inf
    oldCost=0
    fullPold = fullP
    bestP = fullP
    takeRand = False
    startSIZE = 0.1
    maxSIZE = 0.3
    searchSize = startSIZE
    costersB = np.ones(3)

    if clustersTF == False:
        targets += [.3,.3,.05]
    for iter in range(ITERS):
        if takeRand==False:
            fullPold = fullP.copy()
        else:
            # smaller steps
            fullP += np.random.randn(totalP)*searchSize
            searchSize += .1
            if searchSize>maxSIZE:
                searchSize = startSIZE
        # some params must be >0
        fullP[0] = np.maximum(fullP[0],0.01)
        #fullP[6] = np.maximum(fullP[6],0.01)
        #fullP[7] = np.maximum(fullP[7],0.01)
        
        #costers,printstr = toolTest(fullP[6],fullP[7],REPS,sampRATE,howmany=sampRATE*28*10,clustersTF=clustersTF,cyclesTF=True,maxlimitsTF=True,Lparams=fullP[0:6])
        costers,printstr = toolTest(bestM,bestS,REPS,sampRATE,howmany=sampRATE*30*10,clustersTF=clustersTF,cyclesTF=True,maxlimitsTF=True,Lparams=fullP[0:6])

        thisCost = np.sum(np.multiply((costers-targets)**2,weights)) 
        if thisCost<bestCost:
            # this is better
            nabla = (fullP-fullPold) / (thisCost-oldCost)
            costersB = costers.copy()
            bestCost = thisCost
            bestP = fullP.copy()
            oldCost = thisCost
            takeRand = False
            fullP = fullP-etta*nabla
            searchSize = startSIZE
            
        else:
            # we failed. back up.
            fullP = bestP.copy()
            oldCost = bestCost
            takeRand = True
    
    return bestP,bestCost,costersB


def test_p_value(x,periods,sampFREQ,w=12):
    widths = w*sampFREQ*periods / (2*np.pi)
    xm = (x - np.mean(x) ) / np.std(x)
    cwtm = signal.cwt(xm, signal.morlet2, widths, w=w)
    szdays = cwtm[:,x>0]
    myC = szdays/np.absolute(szdays)
    theFs = np.mean(myC,axis=1)
    PLV = np.absolute(theFs)

def make_L_plot(sf,ssf,noplot=False):
    # make an L relationship plot
    # INPUTS
    # sf - mean seizure frequency (vector of values, 1 per patient)
    # ssf - std deviation of seizure frequency (vector of values, 1 per patient)
    #  noplot (optional) if True, dont plot

    mask1 = np.isfinite(sf) & ~np.isnan(sf) & (sf>0) & (ssf>0)
    X = np.log10(sf[mask1])
    Y = np.log10(ssf[mask1])
    #plt.plot(X,Y,'.',label='one diary',linewidth=0.5)
    #plt.hist2d(X,Y,bins=100,density=True)
    if noplot==False:
        plt.plot(X,Y,'.',alpha=0.1,label='one diary',linewidth=0.5,color='black')
    #plt.colorbar()
    #mask2 = np.isfinite(X) & np.isfinite(Y)
    #x2= np.reshape(X[mask2],(-1,1))
    #y2 = Y[mask2]
    #reg = LinearRegression().fit(x2, y2)
    reg = LinearRegression().fit(np.reshape(X,(-1,1)),Y)
    slopeD= reg.coef_[0]

    #xspan = np.array([np.min(x2),np.max(x2)])
    xspan = np.array([np.min(X),np.max(X)])
    yspan = reg.predict(np.reshape(xspan,(-1,1)))

    #xspan = np.array([np.min(xspan),np.max(xspan)])
    #yspan = 0.75*xspan
    if noplot==False:
        plt.plot(xspan,yspan,'r-',label=f'fit line slope={slopeD:0.2}',linewidth=2)
        plt.legend()
        plt.title('L relationship')
        plt.ylabel('Log10 std.dev. sz/month')
        plt.xlabel('Log10 mean sz/month')
    return slopeD

def make_freq_plot(sf,noplot=False):
    # make a frequency distribution plot
    # INPUT
    # sf - vector of mean seizure frequency, one per patient
    # noplot (optional) = if True, do not make a plot

    P,bins = np.histogram(sf,bins=1000,density=True)
    medsf =np.median(sf)
    medsf4 = np.median(sf[sf>4])
    if noplot==False:
        plt.plot(bins[:-1],100*P,'.-',color='blue',)
        plt.plot(medsf*np.array([1,1]),np.array([0,20]),color='red')
        plt.plot(medsf4*np.array([1,1]),np.array([0,15]),color='black')
        plt.text(medsf-1,21,f'Median = {medsf:.2}',color='red')
        plt.text(medsf4-1,16,f'Median (sz/m>4) = {medsf4:.2}',color='black')
        plt.xlim(0,20)
        #plt.ylim(0,100)
        plt.title('Distribution of seizure rates')
        #, cluster_prob=' + str(cluster_prob) + ' clust_decay=' + str(cluster_decay))
        plt.xlabel('Seizures / 30 days')
        plt.ylabel('% population')
        #frame1 = plt.gca()
        #frame1.axes.get_xaxis().set_visible(False)
        #plt.show()
    return medsf,medsf4

def get_sz_freq2(diary,sampFREQ):
    # take a monthly look at a diary and get summary stats on that
    # INPUT
    #  diary - vector of seizure diaries
    #  sampFREQ - number of samples per day
    # OUTPUT
    #  msf = mean of seizure frequency
    #  ssf = std of seizure frequency
    downby = int(sampFREQ*30)
    sf = downsample(diary,downby)

    msf = np.mean(sf)
    ssf = np.std(sf)
    return msf,ssf

def get_sz_freq(diary,dayspersample):
    # take a monthly look at a diary and get summary stats on that
    # INPUT
    #  diary - vector of seizure diaries
    #  dayspersample - number of days per sample (1/samplerate)
    # OUTPUT
    #  msf = mean of seizure frequency
    #  ssf = std of seizure frequency
    
    one_month = int(30/dayspersample)
    
    dC = np.cumsum(diary)
    L = len(diary)
    count=0
    lastCount=0
    monthsL = int(np.ceil(L/one_month))
    sf = np.zeros(monthsL)
    for i in range(monthsL):
        count = np.amin([count+one_month,L-1])
        newSZ = dC[count]-lastCount
        sf[i] = newSZ
        lastCount=dC[count]
    
    msf = np.mean(sf)
    ssf = np.std(sf)
    return msf,ssf

def get_PLV(x,periods,sampFREQ,w=6.0):
    # given signal X, and periods, and w, do complex morlet wavelet analysis
    # then keep wavelets for times when signal was nonzeros and compute phase locking value
    # for those times only
    widths = w*sampFREQ*periods / (2*np.pi)
    xm = (x - np.mean(x) ) / np.std(x)
    cwtm = signal.cwt(xm, signal.morlet2, widths, w=w)
    szdays = cwtm[:,x>0]
    myC = szdays/np.absolute(szdays)
    theFs = np.mean(myC,axis=1)
    PLV = np.absolute(theFs)
    return PLV

def compute_simple_PLV(x,periods,sampFREQ):
    # uses no wavelets here
    t = np.arange(len(x)) / sampFREQ
    PLVs = np.zeros(len(periods))
    for i in range(len(periods)):
        thisFreq = 1/periods[i]
        phaselist = np.mod(t[x>0]*2*np.pi*thisFreq,2*np.pi)
        zm = np.mean(np.exp(1j*phaselist))
        PLVs[i] = np.absolute(zm)

    return PLVs
        

def make_basic_plots_3(REPS,Lparams, sampRATE, fname):
    cyclesTF=True
    clustersTF=True
    maxlimitsTF=True

    MINs = 10**(-1.5)   # this is a very low sz freq
    
    howmany = int(sampRATE*30*48)
    bestM=1.2267388224600906
    bestS = 1.1457004817186776
    #sf = np.zeros(REPS)
    #ssf = np.zeros(REPS)
    #print('Getting started....')
    #for iter in trange(REPS):
    #    mSF = get_mSF(-1)
    #    x = simulator_base(sampRATE,howmany,cyclesTF,clustersTF,maxlimitsTF,defaultSeizureFreq=mSF,Lparams=Lparams)
    #    sf[iter],ssf[iter] = get_sz_freq(x,1/sampRATE)
    doFAST=True
    if doFAST==False:
        resultXs = [ oneTool(bestM,bestS,sampRATE,howmany,cyclesTF,clustersTF,maxlimitsTF,Lparams) for _ in trange(REPS)]
    else:
        n_jobs = 9
        with Parallel(n_jobs=n_jobs, verbose=False) as par:
            resultXs = par(delayed(oneTool)(bestM,bestS,sampRATE,howmany,cyclesTF,clustersTF,maxlimitsTF,Lparams) for _ in trange(REPS))
    bigB2=np.array(resultXs,dtype=float)
    sf = bigB2[:,0].copy()
    ssf = bigB2[:,1].copy()
    
    outfile = TemporaryFile()
    np.savez(outfile, x=sf, y=ssf)
    if 0:
        outfile = TemporaryFile()
        _ = outfile.seek(0)
        npzfile = np.load(outfile)
        sf = npzfile['x']
        ssf = npzfile['y']
        
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    make_freq_plot(sf)
    plt.subplot(1,2,2)
    plt.tight_layout(pad=3.0)
    make_L_plot(sf[sf>MINs],ssf[sf>MINs])
    plt.savefig(fname,dpi=600)
    plt.show()

    
def permute_intervals(x):
    # given series x, keep the values of x, but permute based on the intervals between nonzero entries
    # first zero pad the first item here. This is used by Leguia et al 2021
    x = np.concatenate(([0],x))
    if np.sum(x)==0:
        return x
    else:
        W = np.where(x>0)
        inds = W[0]
        smallerx = x[inds]
        theIntervals = np.concatenate(([inds[0]], np.diff(inds)))
        permIntervals = np.random.permutation(theIntervals)
        sumPermIntervals = np.cumsum(permIntervals)
        new_x = np.zeros(len(x))
        new_x[sumPermIntervals] = np.random.permutation(smallerx)
        return new_x[1:]




def tryCombo5(repeats, Lparams,fname):
    w=12
    simpleTF=True
    sampFREQ = 24
    periods = np.concatenate([np.linspace(.25,1.25,5),np.arange(2,11),np.arange(30,6*30,30),np.arange(30*9,30*18,30*3)])
    plen = len(periods)
    number_of_days = 365*5
    howmany = sampFREQ*365*5
    #fullP= np.array([0.30223,0.9999,0.015366,0.33203,0.0098772,0.49784,0.23713,0.26503,0.0,0.0])
    maxProbs = 5

    PLVs = np.zeros((repeats,plen))
    perms = 1000
    permuteTest = np.zeros((repeats,plen))
    #probs = fullP[0:maxProbs]
    #   theAmps = fullP[maxProbs:]

    for iter in trange(repeats,desc='patient'):
        x= simulator_base(sampFREQ,number_of_days,Lparams=Lparams)
        #CP = make_key_cycle_params(probs,theAmps)
        #x = simulate_diary(sampFREQ,howmany,params=[-1, True, 0.5, 1, 5, .5],returnDetails=False,cycleParams=CP,Lparams=Lparams)
        
        if simpleTF==False:
            PLVs[iter,:]= get_PLV(x,periods,sampFREQ,w)
        else:
            PLVs[iter,:]= compute_simple_PLV(x,periods,sampFREQ)

        PLVs0 = np.zeros((plen,perms))
        for perm_iter in range(perms):
            px = permute_intervals(x)
            if simpleTF==False:
                PLVs0[:,perm_iter] = get_PLV(px,periods,sampFREQ,w)
            else:
                PLVs0[:,perm_iter] = compute_simple_PLV(px,periods,sampFREQ)
    
        for thisP_i in range(plen):
            permuteTest[iter,thisP_i] = stats.percentileofscore(PLVs0[thisP_i,:],PLVs[iter,thisP_i])
    
    drawC(permuteTest,PLVs,repeats,periods,fname)

    return periods,PLVs,permuteTest

def drawC(permuteTest,PLVs,repeats,periods,fname):
    lp = np.log10(periods)
    cutoff = 95
    threshP = (permuteTest>cutoff)*PLVs
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

    cmaps = 'Purples'
    im = NonUniformImage(ax, interpolation='nearest', extent=(np.min(lp),np.max(lp), 0,repeats),origin='lower',cmap=cmaps)
    im.set_data(lp, np.arange(repeats), threshP)
    ax.add_image(im)
    ax.set_xlim(np.min(lp),np.max(lp))
    ax.set_ylim(0,repeats)

    ax2=ax.twinx()
    ax2.plot(lp,(np.nanmean(PLVs,axis=0)),'b-')
    ax2.plot(lp,(np.nanmean(PLVs,axis=0)+np.nanstd(PLVs,axis=0)),'r--')
    ax2.plot(lp,(np.nanmean(PLVs,axis=0)-np.nanstd(PLVs,axis=0)),'r--')
    plt.title('Significant (p<.05) PLV values')
    ax.set_xlabel('Cycle period')
    ax.set_ylabel('Patient number')
    ax2.set_ylabel('Average (across patients) PLV')
    plt.xticks(np.log10([0.5,1,7,30,30*2,30*6,365]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('PLV')

    plt.savefig(fname,dpi=600)
    plt.show()
    return

def update_ticks(x, pos):
        if x == np.log10(.5):
            return '12hr'
        elif x==np.log10(1):
            return '24hr'
        elif x==np.log10(7):
            return '7d'
        elif x==np.log10(30):
            return '1m'
        elif x==np.log10(30*2):
            return '2m'
        elif x==np.log10(30*6):
            return '6m'
        elif x==np.log10(365):
            return '1y'
        else:
            return x


def FULLparamSearchGD():
    like1 = .9999
    idealSum = np.array([.28,.4,.12,.105,.34,.34,.17])
    totalP = 10
    maxProbs=5
    # initialize
    #fullP = np.random.random(totalP)
    # normalize second half
    #fullP[maxProbs:] = fullP[maxProbs:] / np.sum(fullP[maxProbs:])
    #fullP= np.array([0.38931,0.89391,0.12497,0.23017,0.4564,0.78038,0.15704,0.0010266,0.0,0.061551])
    #fullP= np.array([0.83362,0.79416,0.56836,0.23729,0.26629,0.0,0.31105,0.0,0.68895,0.0])
    #fullP= np.array([0.89688,0.88456,0.51745,0.17232,0.23745,0.0,0.42248,0.0,0.57752,0.0])
    #fullP= np.array([0.59966,0.87316,0.048516,0.22119,0.012463,0.0,0.32039,0.38344,0.020885,0.27529])
    #fullP= np.array([0.81959,0.79416,0.56836,0.25798,0.26629,0.029656,0.3074,0.0,0.66294,0.0])
    #fullP= np.array([0.81959,0.79416,0.56836,0.25798,0.26629,0.029656,0.3074,0.0,0.66294,0.0])
    #fullP= np.array([0.85849,0.9999,0.67928,0.0,0.0,0.0,0.40414,0.01059,0.76676,0.0])
    #fullP= np.array([0.84722,0.9999,0.67597,0.0,0.0,0.0,0.46351,0.026971,0.56809,0.0])
    #fullP= np.array([0.9514,0.94392,0.72187,0.0,0.0,0.028001,0.43724,0.0,0.53476,0.0])
    #fullP= np.array([0.80349,0.9884,0.88959,0.073063,0.026148,0.0055795,0.43999,0.0,0.4456,0.10882])
    #fullP= np.array([0.96379,0.85497,0.0,0.84652,0.073115,0.047319,0.67974,0.18516,0.0,0.087773])
    #fullP= np.array([0.97112,0.87365,0.0,0.87217,0.099968,0.026141,0.66818,0.20251,0.0,0.10317])
    #fullP= np.array([0.14983,0.9999,0.23524,0.0,0.0,0.94012,0.053785,0.0,0.0060961,0.0])
    fullP= np.array([0.15044,0.9999,0.24119,0.0,0.011649,0.92924,0.050595,0.0,0.017547,0.0026228])

    howMANY = 10000
    etta = 0.05
    bestCost = np.inf
    oldCost=0
    fullPold = fullP
    bestP = fullP
    takeRand = True
    startSIZE = 0.01
    maxSIZE = 0.3
    searchSize = startSIZE
    for iter in trange(howMANY,desc='param search'):
        if takeRand==True:
            fullPold = fullP
            if np.mod(iter,20)==1:
                # total scramble occasionally
                fullP = np.random.random(totalP)
            else:
                # smaller steps
                fullP += np.random.randn(totalP)*searchSize
                searchSize += .01
                if searchSize>maxSIZE:
                    searchSize = startSIZE
            for i in range(totalP):
                if i<maxProbs:
                    fullP[i] = np.min([fullP[i],like1])
                fullP[i] = np.max([fullP[i],0])
            fullP[maxProbs:] = fullP[maxProbs:] / np.sum(fullP[maxProbs:])

        PLVs,thisSum = tryToFindALLParams(fullP[0:maxProbs],fullP[maxProbs:])
        thisCost = np.sum((thisSum-idealSum)**2)

        if thisCost<bestCost:
            # this is better
            nabla = (fullP-fullPold) / (thisCost-oldCost)
            bestCost = thisCost
            bestP = fullP
            print(f'iter = {iter} bestCost = {bestCost:.5} thisSum = {thisSum[0]:.3} {thisSum[1]:.3} {thisSum[2]:.3} {thisSum[3]:.3} {thisSum[4]:.3} {thisSum[5]:.3} {thisSum[6]:.3}')
            print(f'fullP= np.array([{fullP[0]:.5},{fullP[1]:.5},{fullP[2]:.5},{fullP[3]:.5},{fullP[4]:.5},{fullP[5]:.5},{fullP[6]:.5},{fullP[7]:.5},{fullP[8]:.5},{fullP[9]:.5}])')
            oldCost = thisCost
            takeRand = False
            fullP = fullP-etta*nabla
            searchSize = startSIZE
            
        else:
            # we failed. back up.
            fullP = bestP
            oldCost = bestCost
            takeRand = True
        
        

def tryToFindALLParams(probs,theAmps):
    repeats = 1000
    sampFREQ = 24
    w = 12

    periods = np.array([0.5,1,7,30])
    plen = len(periods)

    periodsB1 = np.array([1])
    periodsMulti = np.arange(4,46)
    periodsB3 = np.array([365])
    plenB = 3
    howmany = sampFREQ*365*5
    catParams = np.concatenate([probs,theAmps])
    PLVs = np.zeros((repeats,plen+plenB))
    perms = 1000
    #permuteTest = np.zeros((repeats,plen))
    for iter in range(repeats):
        CP = make_key_cycle_params(probs,theAmps)
        x = simulate_diary(sampFREQ,howmany,params=[-1, True, 0.5, 1, 5, .5],returnDetails=False,cycleParams=CP)
        PLVs[iter,0:4]= compute_simple_PLV(x,periods,sampFREQ)
        # Baud numbers now
        PLVs[iter,4] = compute_simple_PLV(x,periodsB1,sampFREQ)
        #not doing wavelets
        #PLVs[iter,4] = get_PLV(x,periodsB1,sampFREQ,w)
        x2 = convertToDays(x,sampFREQ)
        PLVs[iter,5] = np.max(compute_simple_PLV(x2,periodsMulti,sampFREQ))
        #not doing wavelets
        #PLVs[iter,5] = np.mean(get_PLV(x2,periodsMulti,1,w))
        # get a moving average with window size=500
        moving_window=500
        cx2 = np.cumsum(x2)
        mave = np.convolve(cx2, np.ones(moving_window)/moving_window, mode='same')
        x3 = cx2 - mave
        #not doing wavelets
        #PLVs[iter,6] = get_PLV(x3,periodsB3,1,w)
        PLVs[iter,6] = compute_simple_PLV(x3,periodsB3,1)

    return PLVs,np.nanmean(PLVs,axis=0)

def convertToDays(x,sampFREQ):
    # given a sampFREQ integer>1, we can downsample to days
    L = len(x)
    L2 = int(np.ceil(L/sampFREQ))        # how many days to make
    ind = 0
    x2 = np.zeros(L2)
    for i in range(L2):
        endCut = ind+sampFREQ
        x2[i] = np.sum(x[ind:endCut])
        ind=endCut
    return x2

def clusterTest1(repeats=1000,Lparams=[1.0051,7.4548,3.663,0.40024,3.0366,7.7389]):
    sampFREQ=1
    howmany = 365*20*sampFREQ
    MAXbin=20
    hista = np.zeros((repeats,MAXbin-2))
    B = np.arange(MAXbin)
    for iter in trange(repeats,desc='clustTest1'):
        
        CP = make_key_cycle_params()
        x= simulate_diary(sampFREQ,howmany,params=[-1, True,  0.5, 1, 5, .5],returnDetails=False,cycleParams=CP,maxLimits=True,Lparams=Lparams)  
        hist1,bin_edges = np.histogram(x[x>0],bins = B[1:],density=True)
       
        hista[iter,:] = hist1
    
    plt.bar(bin_edges[:-1],100*np.nanmedian(hista,axis=0))
    plt.title(f'Median seizures per day across {repeats} patients')
    plt.ylabel('Percentage of all seizures in 20 years')
    plt.xlabel('Number of seizures')
    plt.show()
    plt.bar([1,2],[100*np.nanmedian(hista[:,0],axis=0),100*np.sum(np.nanmedian(hista[:,1:],axis=0))])
    plt.title(f'Median across {repeats} patients')
    plt.xticks([1,2],['Isolated','Cluster'])
    plt.ylabel('Percentage of all seizures in 20 years')
    plt.show()
    #return bin_edges,hista

def clusterTest1v2(repeats,Lparams,clustP):
    sampFREQ=24
    
    number_of_days = 365*20
    MAXbin=20
    B = np.arange(MAXbin)
    histA = np.zeros((2,repeats,MAXbin-2))
    for i,clustersTF in enumerate([True,False]):
        for iter in trange(repeats,desc='clustTest1'):
            x = simulator_base(sampFREQ,number_of_days,cyclesTF=True,clustersTF=clustersTF,
                clusterParams=clustP,Lparams=Lparams)
            hist1,bin_edges = np.histogram(x[x>0],bins = B[1:])
            histA[i,iter,:] = hist1

    yA = np.nanmedian(histA[0,:,:],axis=0)
    yA /= (np.sum(yA) / 100)
    yB = np.nanmedian(histA[1,:,:],axis=0)
    yB /= (np.sum(yB) / 100)

    X_axis = bin_edges[:-1]
    shifter = 0.2
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.bar(X_axis - shifter, yA, 0.3, label = 'With')
    plt.bar(X_axis + shifter, yB, 0.3, label = 'Without')
    plt.xlim([0,6])
    plt.legend()
    plt.title(f'Median seizures per day across {repeats} patients')
    plt.ylabel('# of all seizures in 20 years')
    plt.xlabel('Number of seizures')
    plt.subplot(1,2,2)
    plt.bar(X_axis[yA>0] - shifter, np.log10(yA[yA>0]), 0.3, label = 'With')
    plt.bar(X_axis[yB>0] + shifter, np.log10(yB[yB>0]), 0.3, label = 'Without')
#    plt.title(f'Median seizures per day across {repeats} patients')
    plt.xlim([0,6])
    plt.ylabel('log10 # of all seizures in 20 years')
    plt.xlabel('Number of seizures')

    plt.show()

    shifter = 0.15
    X_axis = np.array([1,2])
    A = np.array([np.nanmedian(histA[0,:,0],axis=0),np.sum(np.nanmedian(histA[0,:,1:],axis=0))],dtype=object)
    A /= (np.sum(A)/100)
    B = np.array([np.nanmedian(histA[1,:,0],axis=0),np.sum(np.nanmedian(histA[1,:,1:],axis=0))],dtype=object)
    B /= (np.sum(B)/100)
    plt.bar(X_axis-shifter,A,.3,label='With')
    plt.bar(X_axis+shifter,B,.3,label='Without')
    plt.legend()
    plt.title(f'Median across {repeats} patients')
    plt.xticks([1,2],['Isolated','Cluster'])
    plt.ylabel('Percentage of all seizures in 20 years')
    plt.show()
    print('With')
    print(A)
    print('Without')
    print(B)

def clusterTest2(repeats=2000,cycleTF=True,starting_mSF=-1):
    sampFREQ=1
    howmany = 365*1*sampFREQ
    xyears = 20
    MAXbin=40
    B = np.arange(MAXbin)
    med_cratio = np.zeros(repeats)
    for iter in trange(repeats):
        cratio = np.zeros(xyears)
        for yr in range(xyears):
            if yr==0:
                if cycleTF==True:
                    CP = make_key_cycle_params()
                else:
                    CP = []
                x,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters = simulate_diary(sampFREQ,howmany,params=[starting_mSF, True, 0.5, 1, 5, .5],returnDetails=True,cycleParams=CP)
            x,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters = simulate_diary(sampFREQ,howmany,params=[mSF, True, 0.5, 1, 5, .5],returnDetails=True,cycleParams=CP)
        
            hist1 = np.sum(x==1)
            histC = np.sum(x>1)
            if hist1==0:
                cratio[yr] = np.nan
            else:
                cratio[yr] = histC / hist1
        med_cratio[iter] = np.nanmedian(cratio)
    
    plt.hist(med_cratio,bins=100)
    plt.show()
    print(f'Less or equal to 1:{np.sum(med_cratio<=1)/repeats}')
    print(f'Less than 1:{np.sum(med_cratio<1)/repeats}')


def defOverallcratios(repeats,Lparams,clusterParams,theCUTOFF):
    clustPno = [0, 0.5, 5, 0.5, 0.5]
    cr = 100*clusterTest4all(repeats=repeats,cyclesTF=True,starting_mSF=-1,sampFREQ=24,clusterParams=clusterParams,Lparams=Lparams)
    crNo  =  100*clusterTest4all(repeats=repeats,cyclesTF=True,starting_mSF=-1,sampFREQ=24,clusterParams=clustPno,Lparams=Lparams) 
    plt.hist(cr,alpha=0.5,color='r',label='With')
    plt.hist(crNo,alpha=0.5,color='b',label='Without')
    plt.legend()
    plt.ylabel('Number of patients')
    plt.xlabel('Median cratio (%) over 20 years')
    plt.title(f'{repeats} patients simulating 20 years of cratios')
    plt.show()
    bars2 = [100*np.sum(cr>theCUTOFF)/repeats,100*np.sum(crNo>theCUTOFF)/repeats]
    plt.bar(['with','without'],bars2)
    plt.ylabel('Percentave of clustering patients')
    plt.title('Seemingly clustering patients')
    plt.show()
    print(bars2)

def drawCratios(repeats,Lparams,clusterParams):
    X=np.linspace(.5,10,20)
    clustPno = [0, 0.5, 5, 0.5, 0.5]
    cr = np.array([ clusterTest4(repeats=repeats,cyclesTF=True,starting_mSF=rate,sampFREQ=24,clusterParams=clusterParams,Lparams=Lparams) for _,rate in enumerate(tqdm(X))],dtype=object)
    crNo  = np.array([ clusterTest4(repeats=repeats,cyclesTF=True,starting_mSF=rate,sampFREQ=24,clusterParams=clustPno,Lparams=Lparams) for _,rate in enumerate(tqdm(X))],dtype=object)

    plt.plot(X,cr,label='with')
    plt.plot(X,crNo,label='without')
    plt.legend()
    plt.ylabel('Ratio of clusters to (clusters+isolated)')
    plt.xlabel('Monthly seizure rate')
    plt.show()

def clusterTest4all(repeats,cyclesTF,starting_mSF,sampFREQ,clusterParams,Lparams):
    number_of_days = 365*1
    xyears = 20
    doFAST = True
    if doFAST==True:
        n_jobs = 10
        with Parallel(n_jobs=n_jobs, verbose=False) as par:
             med_cratio = np.array(par(delayed(clustTestSub4)(cyclesTF, starting_mSF, sampFREQ, Lparams, 
                number_of_days, xyears, clusterParams) for _ in range(repeats)),dtype=object)
    else:
        med_cratio = np.array([clustTestSub4(cyclesTF, starting_mSF, sampFREQ, Lparams, 
            number_of_days, xyears, clusterParams) for _ in range(repeats)],dtype=object)
    return med_cratio

def clusterTest4(repeats,cyclesTF,starting_mSF,sampFREQ,clusterParams,Lparams):
    med_cratio = clusterTest4all(repeats,cyclesTF,starting_mSF,sampFREQ,clusterParams,Lparams)
    return np.median(med_cratio)

def clustTestSub4(cyclesTF, starting_mSF, sampFREQ, Lparams, number_of_days, xyears, clusterParams):
    cratio = np.zeros(xyears)
    
    ## USING THE 8 HOURS cluster def
    S = int(sampFREQ / 3)
    
    for yr in range(xyears):
        seizure_diary_final = simulator_base(sampFREQ,number_of_days,cyclesTF=cyclesTF,clustersTF=True,
            defaultSeizureFreq=starting_mSF,Lparams=Lparams,clusterParams=clusterParams)
        x2 = np.convolve(np.ones(S),seizure_diary_final,mode='valid')
        x3 = np.concatenate([[0],x2])
        dx2 = np.diff(x3)
        dx2back1 = np.concatenate([[0],dx2[0:-1]])
        newClustCount = np.sum((dx2>0) & (x2>1) & (dx2back1<=0))
        newEvent = np.sum((dx2>0) & (x2>0))
            
        if newEvent==0:
            cratio[yr] = np.nan
        else:
            cratio[yr] = newClustCount / newEvent

    return np.nanmedian(cratio)

def show_histo(theSize,sampRATE=24,repeats = 10000,lab='hour',Lparams=[1.0051,7.4548,3.663,0.40024,3.0366,7.7389]):
    
    number_of_days = 365
    
    B = np.linspace(0,theSize,theSize+1)
    H = np.zeros(theSize)
    for K in trange(repeats):
        x = simulator_base(sampRATE,number_of_days,cyclesTF=True,clustersTF=True,Lparams=Lparams)
        h,b = np.histogram(x,bins=B,density=True)
        H += h

    H = H / np.max(H)
    bins = B[:-1]
    Y = np.log10(H)
    M = np.min(Y[H>0])
    B2 = bins[H==0]
    plt.plot(bins,Y,'b.',label='seizures')
    plt.plot(B2,M+(B2*0),'r.','no seizures')
    plt.legend()
    plt.ylabel('log10 density')
    plt.xlabel('number of seizures per ' + lab)
    plt.show()

def optimize_mSF():
    x=np.array([1.0267388224600906,1.1457004817186776])
    DNAsize = 2
    STEPS = 100
    zots = 100
    randConst = 0.1
    bestScoreEver = np.Inf
    bestx = x
    score = np.zeros(zots)
    x2 = np.matlib.repmat(x,zots,1)
    for stepcount in trange(STEPS):
        
        for zotcount in range(zots):
            score[zotcount] = get_msf_score(x)
        inds = np.argsort(score)
        x2 = x2[inds,:]
        scoreP = 1 - np.cumsum((score[inds] / np.sum(score)))
        bestScoreNow = score[inds[0]]
        if bestScoreNow<bestScoreEver:
            bestScoreEver = bestScoreNow
            bestx = x2[0,:]
            print(f'Best score ever = {bestScoreEver:.3}')
            print(f'x=np.array([{bestx[0]},{bestx[1]})')

        # do genetic evolution now
        xnew = np.zeros(zots,DNAsize)
        for zotcount in range(zots):
            for argcount in range(DNAsize):
                r = np.random.random()
                whichZot = np.argmin(np.abs(r-scoreP))
                xnew[zotcount,argcount] = x2[whichZot,argcount]
        xmew += (np.random.randn(zots,DNAsize)*randConst)
        x2 = xnew


def get_msf_score(x):
    REPS = 1000
    m = np.zeros(REPS)
    for iter in range(REPS):
        m[i] = get_mSF(-1,x[0],x[1])
    hist,bins = np.histogram(m,100,density=True)
    score = 10*(np.median(m)-2.7)**2 + (bins[np.argmax(hist)]-2.0)**2 + (np.median(m[m>4])-8.0)**2
