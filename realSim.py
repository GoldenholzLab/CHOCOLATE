import numpy as np


def simulator_base(sampRATE,number_of_days,cyclesTF=True,clustersTF=True, maxlimits=True, defaultSeizureFreq = -1,
    Lparams=[1.5475,17.645,10.617,5.9917,-0.3085,1.5371],CP=[],returnDetails=False,clusterParams=[.5, 1, 7, 1, 1],
    bestM=1.2267388224600906,bestS = 1.1457004817186776):
    # wrapper function for simulate_dairy
    # given input params, will simulate 1 diary
    # INPUTS:
    #  sampRATE = samples per 24hr day (ie 1 = 1 per day, 24 = 1 per hr, etc)
    #  number of days = number of DAYS of samples
    #  cyclesTF (optional, default true) use cycles? if so, use standard published type
    #  clustersTF (optional, default true) use clusters? if so, use standard tyoe
    #  maxlimits (optional, default true) use max limit on sz rate?
    #  defaultSeizureFreq (optional, default=-1 ie pick one randomly) 
    #  Lparams = paramters for the L relationship setup
    #  CP = (default =[], generate standard cycles), these are cycle params, which could be custom if desired
    #  returnDetails = (default False) - if set to true, the output will be a big tuple with all the detailed variables (see simulate_diary)
    #  clusterParams = [ cluster_prob, clustTIME, refractoryClustTIME, clustMax,each_cluster_prob] 
    #  bestM = for seizure frequency heteregenity calculation bestM
    #  bestS = for seizure frequency heterogenity calculation bestS
    # OUTPUTS:
    #  diary = vector of diary counts

    # how many samples are desired?
    number_of_samples = number_of_days*sampRATE

    # seizure frequency parameters
    SFparams = [defaultSeizureFreq, bestM, bestS]

    # cluster related defaults
    if clustersTF==True:
        cluster_prob = clusterParams[0]          # the fraction of patients who have clusters added
    else:
        cluster_prob = 0
    clusterParams = [cyclesTF,cluster_prob, clusterParams[1], clusterParams[2], clusterParams[3], clusterParams[4]]

    return simulate_diary(sampRATE=sampRATE,number_of_samples=number_of_samples,
        SFparams=SFparams,returnDetails=returnDetails,cycleParams=CP,
        maxLimits=maxlimits,Lparams=Lparams,clusterParams=clusterParams)

def simulate_diary(sampRATE,number_of_samples,SFparams,returnDetails,cycleParams,maxLimits,Lparams,clusterParams):
    # given input params, will simulate 1 diary
    # INPUTS:
    #  sampRATE = samples per 24hr day (ie 1 = 1 per day, 24 = 1 per hr, etc). Max is 144.
    #  number_of_samples = number of diary samples
    # SFparams = [requestedSF,bestM,bestS]
    #   requested_msf - if -1, that means compute a new one. If >=0, use this as monthly seizure rate
    # returnDetails = returns the details of variables used to build diary
    # cycleParams = (optional) list of lists. list[0] is frequencies. list[1] is relative amplitudes. If cycleParams = []
    #       then no cycles will be used. Default value
    # maxLimits = (optional, default True) true or false impose a limit on the maximum number of
    #       seizures allowed from this system. This will prevent patients from having >1 seizure
    #       in 10 minutes, which anyway would be effectively status epilepticus and therefore not
    #       reasonably counted as >1 seizure.
    # Lparams = the parameters needed for L relationship.
    # clusterParams = 
    #       [cyclesTF, cluster_prob, clustTIME, refractoryClustTIME, clustMax,each_cluster_prob]

    # OUTPUTS:
    #  seizure_diary_final = vector of diary counts
    #   if returnDetails=True... then additional outputs are given too..
    # seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_B
    
    if sampRATE>144:
        raise NameError('sampRATE is too big! Max is 144, ie every 10 minutes.')

    # do you even want cycles?
    cyclesTF = clusterParams[0]
    # chance of having a patient with clusters (0 to 1)
    cluster_prob = clusterParams[1]
    # time (in days) that a cluster can contine
    clustTIME = clusterParams[2]
    # time (in days) after a cluster that clusters will be less likely
    refractoryClustTIME = clusterParams[3]
    # maximal multiplier for clustering
    clustMAX = clusterParams[4]
    # chance of each seizure becoming a cluster
    each_cluster_prob=clusterParams[5]

    # get heterogenity of SF
    mSF = get_mSF( requested_msf=SFparams[0],bestM=SFparams[1],bestS=SFparams[2] )

    # generate the basic rate (think of this as basic seizure susceptibility)
    # first convert the monthly rate into a daily rate
    SF = (mSF / 30)
    # now compute the overdispersion parameter
    #overdispersion_coef = 10
    #overdispersion_pow = -0.5
    #overdispersion =  overdispersion_coef*SF**overdispersion_pow
    #0.82176,7.4326,3.6196,0.19086,2.7791,6.5902
    #tempdisp = Lparams[1]*np.log10(SF) + Lparams[2]*(SF**Lparams[3]) + Lparams[4]*SF + Lparams[5]
    tempdisp = Lparams[1]*np.log10(np.max([0.001,SF + Lparams[2]])) + Lparams[3]*np.log10(np.max([0.001,SF + Lparams[4]]))  + Lparams[5]
    
    overdispersion = np.maximum(Lparams[0],tempdisp)
    # using a a gamma function, get that rate
    rate = np.random.gamma(1/(sampRATE*overdispersion), SF*overdispersion, number_of_samples)

    # Prepare cycles
    # if there is nothing in CP, then generate stanard cycle parameters
    doSomeCycles=False
    if cyclesTF==True:
        if len(cycleParams)==0:
            cycleParams = make_key_cycle_params()
        # it is possible that no cycles are selected. In that case, don't run
        if len(cycleParams)>0:
            theFreqs = cycleParams[0]
            theAmps = cycleParams[1]
            cycles = make_cycles(number_of_samples,sampRATE,theFreqs,theAmps)
            doSomeCycles=True
    if doSomeCycles==False:
        cycles = np.zeros(number_of_samples)
        theFreqs = []
        theAmps = []

    # modulate the rate based on cycles
    modulated_rate = rate * (cycles + 1)

    # input the modulated rate into a poisson process
    # (note, by construction, this amounts to a negati(ve binomial model)
    seizure_diary_A = np.random.poisson(modulated_rate)

    do_clusters,modulated_cluster_rate,seizure_diary_B = run_cluster(cluster_prob,modulated_rate,
        sampRATE,clustTIME,refractoryClustTIME,clustMAX,number_of_samples,seizure_diary_A,each_cluster_prob)
    
    seizure_diary_final = apply_maxlimits(sampRATE, number_of_samples, maxLimits, seizure_diary_B)

    # if details were EXPLICITLY requested, then and only then return them
    if returnDetails==False:
        return seizure_diary_final
    else:
        return seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_A,seizure_diary_B


def apply_maxlimits(sampRATE, number_of_samples, maxLimits, seizure_diary_B):
    #INPUTS:
    # sampeRATE - number of samples per 24 hours
    # number_of_samples - how many samples at that rate will be obtained
    # maxLimits - True or False. If False, do nothing to the data
    # seizure_diary_B - the proposed diary to modify
    #
    #OUTPUTS:
    # seizure_diary_final - after applying maxlimits when requested
    
    if maxLimits==True:
        # impose a maximum limit rule: maximum seizure count = 1 seizure in 10 minutes.
        # if greater than that, you are overcounting something that should really be status.
        maxS =  144 / sampRATE  # this is the largest seizure count allowed
        M = np.concatenate([[seizure_diary_B],[(np.zeros(number_of_samples) + maxS)]])
        seizure_diary_final = np.min(M,axis=0)
    else:
        seizure_diary_final = seizure_diary_B.copy()
    return seizure_diary_final
    

def run_cluster(cluster_prob,modulated_rate,sampRATE,clustTIME,refractoryClustTIME,clustMAX,number_of_samples,seizure_diary_A,each_cluster_prob):
    # Modify the seizure_diary based on clusters
    # INPUTS:
    #   cluster_prob - probability between 0..1 that this patient has extra clusters
    #   modulated_rate - the rate modified by cycles
    #   sampRATE - number of samples per 24hrs
    #   clustTIME - the number of days cluster risk is higher
    #   refractoryClustTIME - the number of days cluster risk is lower after a cluster
    #   clustMAX - the amplitude of the cluster (between 0..1)
    #   number_of_samples - number of samples in the entire diary
    #   seizure_diary - diary without clustering
    #   each_cluster_prob - probability of THIS seizure starting a cluster
    # OUTPUTS:
    #   do_clusters - true or false, this patient has clusters added
    #   modulated_cluster_rate - the modulated_rate modified by cluster risk
    #   seizure_diary - new seizure diary with clusters implemented

    seizure_diary_B = seizure_diary_A.copy()
    # Determine if clusters are needed
    do_clusters =  (np.random.random() < cluster_prob)

    #copy the normal modulated rate. If do_clusters is false, this is trivial.
    modulated_cluster_rate = modulated_rate.copy()
    if do_clusters:
        # generate a balanced sequence clustSEQ that is positive during clustTIME, negative
        # during refractoryClustTIME, and has a max amplitude modulated by clustMAX [0,1]
        # such that the sum of clustSEQ = 0, and max(abs(clustSEQ))=1.
        clustSEQ,maxClustSteps = make_clustSEQ(sampRATE,clustTIME,refractoryClustTIME,clustMAX)
        # quick error check here: if clustSEQ <-1, then rate can become negative.
        clustSEQ[clustSEQ<-1] = 0 

        ranChances = np.random.random(number_of_samples) < each_cluster_prob
        clusterCounter = maxClustSteps
        mask = np.zeros(number_of_samples)
        clusterTerm = np.zeros(number_of_samples)
        for i in range(number_of_samples):
            if (seizure_diary_A[i]>0) and (clusterCounter==maxClustSteps) and (ranChances[i]==True):
                # this only happens if a seizure happened, and we are not in a cluster already, AND we are randomly allowed to start
                clusterCounter = 0
            if clusterCounter<maxClustSteps:
                clusterTerm[i] = clustSEQ[clusterCounter]
                clusterCounter+=1
                # if the modulated rate is 0 here, there is no point in adding this
                # to the mask
                #mask[i] = 1 * (modulated_rate[i]>0)
                mask[i] = 1
                
        # do a batch modification of diaries based on cluster term
        modulated_cluster_rate[mask==1] = modulated_rate[mask==1] * (1 + clusterTerm[mask==1])
        seizure_diary_B[mask==1] = np.random.poisson(modulated_cluster_rate[mask==1])
       
    return do_clusters,modulated_cluster_rate,seizure_diary_B

def get_mSF(requested_msf,bestM=1.2267388224600906,bestS = 1.1457004817186776):
    # these two parameters were found using a stochastic optimization procedure that minimized this score:
    #score = 10*(np.median(mSF)-2.7)**2 + (bins[np.argmax(hist)]-2.0)**2 + (np.median(mSF[mSF>4])-8.0)**2
    # see Ferastraoaru et al 2018 for large population data.
    # INPUTS: requested_msf (default=-1, optional) - if -1 given, generate a random number
    #                otherwise simply return the requested_msf
    #         bestM,bestS - parameters for lognormal (OPTIONAL!)
    #      (  bestM=1.2267388224600906,bestS = 1.1457004817186776  )
    # OUTPUTS: mSF - the monthly seizure frequency, randomly chosen to be realistic
    
    if requested_msf==-1:
        mSF = np.random.lognormal(bestM,bestS,size=1)
    else:
        mSF = requested_msf

    return mSF


def make_cycles(number_of_samples,sampFreq,theFreqs,theAmps):
    # given number of samples, how many samples in 1 day, an array of cycle 
    # frequencies (cycles per day) and relative amplitudes for each
    # a combined cycle signal will be generated that represents the sum of some sin waves
    # of prespecified frequency and amplitude, random phase, and normalized such that the max
    # of the entire signal is 1.
    # INPUTS
    #  number_of_samples - total number of samples requested
    #  sampFreq - number of samples per 24hr day
    #  theFreqs = array of frequencies (cycles per 24 hr day)
    #  theAmps = array of amplitudes that corresponds to theFreqs
    # OUTPUTS
    #  normedCycles - array of normalized sum of cycles
    cycles = np.zeros(number_of_samples)
    t = np.linspace(start=0,stop=(number_of_samples/sampFreq),num=number_of_samples)

    for c, this_freq in enumerate(theFreqs):
        # only add the cycle of we have at least Nyquest sampling
        if sampFreq > (2*this_freq):
            this_phase = np.random.random() * np.pi * 2
            this_amp = theAmps[c]
            this_cycle = this_amp * np.sin(t*this_freq*2*np.pi + this_phase)
            cycles  += this_cycle
    
    maxCycle = np.max(np.abs(cycles))
    if maxCycle==0:
        maxCycle = 1
    normedCycles = cycles/maxCycle
    return normedCycles

def make_key_cycle_params(plist=[0.15044,0.9999,0.24119,0.0,0.011649],
        alist=[0.92924,0.050595,0.0,0.017547,0.0026228]):
    # the following code uses a series of paramters generated via an optimization algorithm
    # it will generate a list of arrays which are used to generate cycles
    # sometimes it outputs empty list, meaning make no cycles.
    # INPUTS
    #  plist
    #  alist
    # OUTPUTS
    #  CP - list of []
    c= 0
    freqs = np.array([])
    amps = np.array([])
    if np.random.random()<plist[0]:
        thisPeriod = 0.5
        freqs = np.concatenate([freqs,[1./thisPeriod]])
        amps = np.concatenate([amps,[alist[0]]])
        c+=1
    if np.random.random()<plist[1]:
        thisPeriod = 1
        freqs = np.concatenate([freqs,[1./thisPeriod]])
        amps = np.concatenate([amps,[alist[1]]])
        c+=1
    if np.random.random()<plist[2]:
        thisPeriod = 7
        freqs = np.concatenate([freqs,[1./thisPeriod]])
        amps = np.concatenate([amps,[alist[2]]])
        c+=1
    if np.random.random()<plist[3]:
        thisPeriod = np.random.randint(4,46)  
        freqs = np.concatenate([freqs,[1./thisPeriod]])
        amps = np.concatenate([amps,[alist[3]]])
        c+=1
    if np.random.random()<plist[4]:
        thisPeriod = np.random.randint(90,451)  
        freqs = np.concatenate([freqs,[1./thisPeriod]])
        amps = np.concatenate([amps,[alist[4]]])
        c+=1
    
    if c==0:
        CP=[]    
    else:
        CP = [freqs,amps]
    return CP

def make_clustSEQ(sampRATE,clustTIME,refractoryClustTIME,clustMAX):
    # generate a balanced sequence clustSEQ that is positive during clustTIME, negative
    # during refractoryClustTIME, and has a max amplitude modulated by clustMAX [0,1]
    # such that the sum of clustSEQ = 0, and max(abs(clustSEQ))=1.
    # INPUTS:
    #   sampRATE = samples per 24hr day
    #   clustTIME = number of days for a cluster
    #   refractoryClustTIME = number of days AFTER a cluster when seizures are LESS likely
    #   clustMAX = an amplitude for the severity of the cluster, between 0 and 1
    # OUTPUTS:
    #   clustSEQ = cluster sequence, used to modulate seizure risk
    #   maxClusterSteps = length of clustSEQ

    # error checking to ensure 0..1
    #clustMAX = np.max([np.min([clustMAX,1]),0]) 
    clustMAX = np.max([0,clustMAX])

    # how many samples UP = cluster likely
    downTIME = int(refractoryClustTIME*sampRATE)
    # how many samples DOWN = cluster UNlikely
    upTIME = int(np.max([1,np.floor(clustTIME*sampRATE)]))
    maxClustSteps = downTIME+upTIME

    # build a zero mean signal, that has balanced up and down segments
    clustSEQ = np.zeros(maxClustSteps)/maxClustSteps
    clustSEQ[0:upTIME] += clustMAX
    clustSEQ[upTIME:] -= clustMAX * upTIME/downTIME
    #clustSEQ[0:upTIME] += (clustMAX/upTIME)
    #clustSEQ[upTIME:] -= (clustMAX/downTIME)
        
    return clustSEQ,maxClustSteps


def downsample(x,byHowmuch):
    # input: 
    #    x = diary
    #    byHowMuch = integeter by how much to downsample
    # outputs
    #   x3 = the new diary, downsampled.
    #
    # If I sample 24 samples per day, and downsample by 24 then I get
    # daily samples as the output, for instance.
    #
    L = len(x)
    x2 = np.reshape(x,(int(L/byHowmuch),byHowmuch))
    x3 = np.sum(x2,axis=1)
    return x3