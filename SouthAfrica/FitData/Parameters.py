"""Extract modelling parameters for SISD model of an infection outbreak.

Model assumes that people susceptible to be infected (S) are infected (I) at a
rate of beta (% of people per day), that infected people who recover at a rate
rho (recoveries per day), will be again susceptible to be infected, and that
infected people can die (D) at a rate delta (deaths per day), hence (S -> I -> S
or S -> I -> D). Let N be the total population which can be infected and let Ns
and Ni be the number of people susceptible and infected, respectively, then the
flow from S to I is beta*Ni*Ns = beta*Ni*(N - Ni) and the flow from I back to S
is rho*Ni, while the flow from I to D is delta*Ni.
"""


################################################################################
# Only execute the code if the script is ran and not if it is imported         #
################################################################################
if __name__ == "__main__":
    import numpy as np                    # Numerical Python package
    import matplotlib.pyplot as plt       # Graphing package
    import pickle                         # Object file saving package
    from scipy.optimize import curve_fit  # Curve fit method
    
    ############################################################################
    # Calculate the slope and intercept of a straight line through data points #
    ############################################################################
    def SlopeIntercept(x, y):
        N = len(y)
        sx = np.sum(x)
        sy = np.sum(y)
        sx2 = np.sum(x**2)
        sxy = np.sum(x*y)
        fac = N*sx2 - sx**2
        m = (N*sxy - sx*sy)/fac
        c = (sx2*sy - sxy*sx)/fac
        return m, c
    
    def StraightLine(x, m, c):
        return m*x + c
    
    ############################################################################
    # Data                                                                     #
    ############################################################################
    data = np.loadtxt("Data.txt", usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))
    
    N = 5.9e7                               # Population
    ai = data[:,1] - data[:,5] - data[:,7]  # Active infections
    sai = ai*(N - ai)                       # Susceptibles times active infections
    
    # Important events
    days = [10, 13, 18, 22, 35, 40, 53, 57]
    comments = ['A: State of disaster', 'B: Schools and universities close',
                'C: lvl5 lockdown announced', 'D: lvl5 lockdown started',
                'E: lvl5 lockdown extension announced',
                'F: Mines started with 50% productivity', 'G: Freedom day',
                'H: Workers day / lvl4 lockdown started']
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    infect = np.zeros_like(days)
    for i in range(len(infect)):
        infect[i] = data[days[i],1]
    inf1 = data[36,1]
    inf2 = data[39,1]
    
    # Rework recoveries to per day
    infecRecov = np.zeros(len(data[:,6]))
    j = len(data[:,6])
    tmp = data[-1,6]
    for i in range(j-1, 14, -1):
        if (data[i,6] != 0.0):
            infecRecov[i+1:j] = tmp/(j - i - 1.0)
            j = i + 1
            tmp = data[i,6]
    infecRecov[i:j] = tmp/(j - i)
    
    # Positive tests vs time
    posTest = (data[:,2]/data[:,4])*100.0
    s = 0.0
    s2 = 0.0
    n = 0.0
    for i in range(0, 23):  # Remove nan or inf values when calculating average
        if np.isfinite(posTest[i]):
            s += posTest[i]
            s2 += posTest[i]**2
            n += 1.0
    avgPosTestI = s/n
    # Uncertainty on average is standard deviation divided by the square-root of
    # the number of data points used to calculate the average
    stdPosTestI = np.sqrt((s2/n - avgPosTestI**2)/n)
    s = 0.0
    s2 = 0.0
    n = 0.0
    for i in range(23, 57):
        if np.isfinite(posTest[i]):
            s += posTest[i]
            s2 += posTest[i]**2
            n += 1.0
    avgPosTest5 = s/n
    stdPosTest5 = np.sqrt((s2/n - avgPosTest5**2)/n)
    s = 0.0
    s2 = 0.0
    n = 0.0
    for i in range(57, len(posTest)):
        if np.isfinite(posTest[i]):
            s += posTest[i]
            s2 += posTest[i]**2
            n += 1.0
    avgPosTest4 = s/n
    stdPosTest4 = np.sqrt((s2/n - avgPosTest4**2)/n)
    #print [avgPosTestI, stdPosTestI], [avgPosTest5, stdPosTest5], [avgPosTest4, stdPosTest4]
    
    # Percentage increase vs time
    prcntIn = (data[1:,2]/(data[:-1,1] - data[:-1,5] - data[:-1,7]))*100.0
    avgIncrsI = np.average(prcntIn[0:22])
    stdIncrsI = np.std(prcntIn[0:22])/np.sqrt(len(prcntIn[0:22]))
    avgIncrs5 = np.average(prcntIn[22:56])
    stdIncrs5 = np.std(prcntIn[22:56])/np.sqrt(len(prcntIn[22:56]))
    avgIncrs4 = np.average(prcntIn[56:])
    stdIncrs4 = np.std(prcntIn[56:])/np.sqrt(len(prcntIn[56:]))
    #print [avgIncrsI, stdIncrsI], [avgIncrs5, stdIncrs5], [avgIncrs4, stdIncrs4]
    
    # 3 day running average of infections, recoveries, deaths, and positive tests
    avg3infct = np.zeros_like(data[:,2])
    avg3recov = np.zeros_like(avg3infct)
    avg3death = np.zeros_like(avg3infct)
    avg3ptest = np.zeros_like(avg3infct)
    avg3incrs = np.zeros_like(avg3infct)
    for i in range(1, len(avg3infct)-1):
        avg3infct[i] = np.average(data[i-1:i+2,2])
        avg3recov[i] = np.average(infecRecov[i-1:i+2])
        avg3death[i] = np.average(data[i-1:i+2,8])
        avg3ptest[i] = np.average(posTest[i-1:i+2])
        if (i < len(prcntIn)-1):
            avg3incrs[i] = np.average(prcntIn[i-1:i+2])
    avg3incrs[-2] = np.average(prcntIn[-3:])
    
    # Slope and intercept of infection rate vs active infections times susceptibles
    # (First calculate with formula to have initial values and then with fitting
    #  function to have uncertainties; uncertainty on data point is estimated to
    #  be the square-root of the values)
    b1, c1 = SlopeIntercept(sai[:23], data[:23,2])
    fit, cov = curve_fit(StraightLine, sai[:23], data[:23,2], p0=[1.1*b1, 1.1*c1])
    b1 = fit[0]
    c1 = fit[1]
    cov = np.sqrt(np.diag(cov))
    eb1 = cov[0]
    ec1 = cov[1]
    #print fit, cov, (cov/np.abs(fit))*100.0
    b2, c2 = SlopeIntercept(sai[23:57], data[23:57,2])
    fit, cov = curve_fit(StraightLine, sai[23:57], data[23:57,2], p0=[1.1*b2, 1.1*c2])
    b2 = fit[0]
    c2 = fit[1]
    cov = np.sqrt(np.diag(cov))
    eb2 = cov[0]
    ec2 = cov[1]
    #print fit, cov, (cov/np.abs(fit))*100.0
    b3, c3 = SlopeIntercept(sai[57:], data[57:,2])
    fit, cov = curve_fit(StraightLine, sai[57:], data[57:,2], p0=[1.1*b3, 1.1*c3])
    b3 = fit[0]
    c3 = fit[1]
    cov = np.sqrt(np.diag(cov))
    eb3 = cov[0]
    ec3 = cov[1]
    #print fit, cov, (cov/np.abs(fit))*100.0
    
    # Slope and intercept of recovery rate vs active infections
    r, cr = SlopeIntercept(ai[15:], infecRecov[15:])
    fit, cov = curve_fit(StraightLine, ai[15:], infecRecov[15:], p0=[1.1*r, 1.1*cr])
    r = fit[0]
    cr = fit[1]
    cov = np.sqrt(np.diag(cov))
    er = cov[0]
    ecr = cov[1]
    #print fit, cov, (cov/np.abs(fit))*100.0
    
    # Slope and intercept of death rate vs active infections
    d, cd = SlopeIntercept(ai[21:], data[21:,8])
    fit, cov = curve_fit(StraightLine, ai[21:], data[21:,8], p0=[1.1*d, 1.1*cd])
    d = fit[0]
    cd = fit[1]
    cov = np.sqrt(np.diag(cov))
    ed = cov[0]
    ecd = cov[1]
    #print fit, cov, (cov/np.abs(fit))*100.0
    
    ############################################################################
    # Parameters                                                               #
    ############################################################################
    # Time dependence of infection rate
    def Beta(t, betai, lckdwn5, beta5, lckdwn4, beta4, trans):
        initial = betai*0.5*(1.0 + np.tanh((lckdwn5 - t)/trans))
        lvl5 = beta5*0.5*(np.tanh((t - lckdwn5)/trans) - np.tanh((t - lckdwn4)/trans))
        lvl4 = beta4*0.5*(1.0 + np.tanh((t - lckdwn4)/trans))
        return initial + lvl5 + lvl4
    
    def ModelBeta(t, betai, lckdwn5, beta5, lckdwn4, beta4, trans):
        initial = betai*0.5*(1.0 + np.tanh((lckdwn5 - t)/trans))
        badstat = beta5*0.5*(np.tanh((t - lckdwn5)/trans) - np.tanh((t - lckdwn4)/trans))
        betstat = beta4*0.5*(1.0 + np.tanh((t - lckdwn4)/trans))
        time = [4.0, 13.0, 22.0, 36.0, 43.0, 58.0, 77.0,]
        mods = [0.5,  1.0, 1.15, 0.35,  0.5, 0.65, 0.85, 0.8]
        mod = mods[0]*0.5*(1.0 + np.tanh((time[0] - t)/trans))
        for i in range(1, len(time)):
            mod += mods[i]*0.5*(np.tanh((t - time[i-1])/trans) - np.tanh((t - time[i])/trans))
        mod += mods[-1]*0.5*(1.0 + np.tanh((t - time[-1])/trans))
        return mod*(initial + badstat + betstat)
    
    def InfectionRate(t, x, betai, ini, lckdwn5, beta5, in5, lckdwn4, beta4, in4, trans):
        initial = (betai*x + ini)*0.5*(1.0 + np.tanh((lckdwn5 - t)/trans))
        lvl5 = (beta5*x + in5)*0.5*(np.tanh((t - lckdwn5)/trans) - np.tanh((t - lckdwn4)/trans))
        lvl4 = (beta4*x + in4)*0.5*(1.0 + np.tanh((t - lckdwn4)/trans))
        return initial + lvl5 + lvl4
    
    # Time dependence of recovery rate
    def Rho(t, rho, delay, trans):
        return rho*0.5*(1.0 + np.tanh((t - delay)/trans))
    
    def ModelRho(t, rho, delay, trans):
        time = [34.0, 69.0]
        mods = [0.05, 0.6, 1.0]
        mod = mods[0]*0.5*(1.0 + np.tanh((time[0] - t)/trans))
        for i in range(1, len(time)):
            mod += mods[i]*0.5*(np.tanh((t - time[i-1])/trans) - np.tanh((t - time[i])/trans))
        mod += mods[-1]*0.5*(1.0 + np.tanh((t - time[-1])/trans))
        return mod*rho*0.5*(1.0 + np.tanh((t - delay + 5.0)/trans))
    
    def RecoveryRate(t, x, rho, offset, delay, trans):
        return (rho*x + offset)*0.5*(1.0 + np.tanh((t - delay - 0.5)/trans))
    
    # Time dependence of death rate
    def ModelDelta(t, delta, delay, trans):
        time = [40.0, 53.0, 80.0]
        mods = [0.4,   0.8,  0.7, 0.8]
        mod = mods[0]*0.5*(1.0 + np.tanh((time[0] - t)/trans))
        for i in range(1, len(time)):
            mod += mods[i]*0.5*(np.tanh((t - time[i-1])/trans) - np.tanh((t - time[i])/trans))
        mod += mods[-1]*0.5*(1.0 + np.tanh((t - time[-1])/trans))
        return mod*delta*0.5*(1.0 + np.tanh((t - delay - 0.5)/trans))
    
    lckdwn5 = 22.5
    lckdwn4 = 57.0
    sicktime = 19.0
    firstdeath = 22.0
    transition = 0.5
    t = np.arange(data[0,0], data[-1,0], 0.1)
    betat = Beta(t, b1, lckdwn5, b2, lckdwn4, b3, transition)
    betatu = Beta(t, b1+eb1, lckdwn5, b2+eb2, lckdwn4, b3+eb3, transition)
    betatl = Beta(t, b1-eb1, lckdwn5, b2-eb2, lckdwn4, b3-eb3, transition)
    modbt = ModelBeta(t, b1, lckdwn5, b2, lckdwn4, b3, transition)
    rhot = Rho(t, r, sicktime, transition)
    rhotu = Rho(t, r+er, sicktime, transition)
    rhotl = Rho(t, r-er, sicktime, transition)
    modrt = ModelRho(t, r, sicktime, transition)
    deltat = Rho(t, d, firstdeath, transition)
    deltatu = Rho(t, d+ed, firstdeath, transition)
    deltatl = Rho(t, d-ed, firstdeath, transition)
    moddt = ModelDelta(t, d, firstdeath, transition)
    
    # Modelled data
    nwnfct = InfectionRate(data[:,0], sai, b1, c1, lckdwn5, b2, c2, lckdwn4, b3, c3, transition)
    recov = RecoveryRate(data[:,0], ai, r, cr, sicktime, transition)
    deaths = RecoveryRate(data[:,0], ai, d, cd, firstdeath, transition)
    
    # Safe parameters for use in SISD modelling
    InfectRate = {'betas':(b1, b2, b3), 'times':(lckdwn5, lckdwn4), 'trans':transition}
    RecoverRate = {'rho':r, 'delay':sicktime, 'trans':transition}
    DeathRate = {'delta':d, 'delay':firstdeath, 'trans':transition}
    with open('FitParameters.pckl', 'wb') as fh:
        pickle.dump((InfectRate, RecoverRate, DeathRate), fh)
    
    ############################################################################
    # Plots                                                                    #
    ############################################################################
    # Infection, recovery, and death rate vs time
    fig1 = plt.figure()
    
    maks = np.max(data[:,2])
    subplot1a = fig1.add_subplot(311)
    subplot1a.plot(data[:,0], data[:,2], color='r', markeredgecolor='r', marker='o', linestyle=':')
    subplot1a.plot(data[1:-1,0], avg3infct[1:-1], color='r')
    for i in range(len(days)):
        subplot1a.vlines(days[i], 0.0, maks, color='grey', linestyle='--', label=comments[i])
    subplot1a.axvspan(36.0, 39.0, color='grey', alpha=0.3, label='Easter')
    subplot1a.set_title('South African COVID-19 Data', fontsize=18)
    subplot1a.set_xlim(data[0,0], data[-1,0])
    subplot1a.set_xticklabels([''])
    subplot1a.set_ylabel('New infections per day', fontsize=16)
    subplot1a.set_ylim(0.0, maks)
    subplot1a.tick_params(labelsize=16, top=False, right=True, direction='inout')
    
    maks = np.max(data[:,6])
    subplot1b = fig1.add_subplot(312)
    subplot1b.plot(data[:,0], data[:,6], color='lightgreen', markeredgecolor='lightgreen', marker='^', linestyle='--', label='Data')
    subplot1b.plot(data[:,0], infecRecov, color='g', markeredgecolor='g', marker='o', linestyle=':', label='Reworked data')
    subplot1b.plot(data[1:-1,0], avg3recov[1:-1], color='g', label='3 day running average')
    for i in range(len(days)):
        subplot1b.vlines(days[i], 0.0, maks, color='grey', linestyle='--')
        subplot1b.text(days[i], maks, abc[i], fontsize=16)
    subplot1b.axvspan(36.0, 39.0, color='grey', alpha=0.3, label='Easter')
    subplot1b.set_xlim(data[0,0], data[-1,0])
    subplot1b.set_xticklabels([''])
    subplot1b.set_ylabel('Recoveries per day', fontsize=16)
    subplot1b.set_ylim(0, maks)
    subplot1b.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.5)
    
    maks = np.max(data[:,8])
    subplot1c = fig1.add_subplot(313)
    subplot1c.plot(data[:,0], data[:,8], color='k', markeredgecolor='k', marker='o', linestyle=':')
    subplot1c.plot(data[1:-1,0], avg3death[1:-1], color='k')
    for i in range(len(days)):
        subplot1c.vlines(days[i], 0.0, maks, color='grey', linestyle='--', label=comments[i])
    subplot1c.axvspan(36.0, 39.0, color='grey', alpha=0.3)
    subplot1c.set_xlabel('Days since first infection on 5 March', fontsize=16)
    subplot1c.set_xlim(data[0,0], data[-1,0])
    subplot1c.set_ylabel('Deaths per day', fontsize=16)
    subplot1c.set_ylim(0.0, maks)
    subplot1c.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.5)
    
    # Infection rate vs active infections times susceptibles
    fig2 = plt.figure()
    subplot2 = fig2.add_subplot(111)
    subplot2.plot(sai[:23], data[:23,2], color='r', markeredgecolor='r', marker='o', linestyle='', label='Data: 5 - 27 March (before lvl5 lockdown)')
    subplot2.plot(sai[23:57], data[23:57,2], color='orange', markeredgecolor='orange', marker='o', linestyle='', label='Data: 28 March - 30 April (lvl5 lockdown)')
    subplot2.plot(sai[57:], data[57:,2], color='c', markeredgecolor='c', marker='o', linestyle='', label='Data: 29 April - 31 May (lvl4 lockdown)')
    subplot2.plot(sai, nwnfct, color='orange', label=r'$(I S - %6.2f / %6.3g) \beta_i (t) + (I S - %6.2f / %6.3g) \beta_5 (t) + (I S - %6.2f / %6.3g) \beta_4 (t)$'%(abs(c1), b1, abs(c2), b2, abs(c3), b3))
    subplot2.set_title('South African COVID-19 Data', fontsize=18)
    subplot2.set_xlabel(r'Active infections times susceptibles [$IS = I(N-I)$]', fontsize=16)
    subplot2.set_xlim(0.0, np.max(sai))
    subplot2.set_ylabel(r'New infections per day [$\Delta I_{\mathrm{new}} / \Delta t$]', fontsize=16)
    subplot2.set_ylim(0.0, np.max(data[:,2]))
    subplot2.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=16, framealpha=0.5)
    
    # Recovery and death rate vs active infections
    fig3 = plt.figure()
    
    subplot3a = fig3.add_subplot(211)
    subplot3a.plot(ai, data[:,6], color='lightgreen', markeredgecolor='lightgreen', marker='s', linestyle='', label='Data')
    subplot3a.plot(ai, infecRecov, color='g', markeredgecolor='g', marker='o', linestyle='', label='Reworked data')
    subplot3a.plot(ai, recov, color='c', label=r'$(I - %6.2f / %6.3f) \rho (t)$'%(abs(cr), r))
    subplot3a.set_title('South African COVID-19 Data', fontsize=18)
    subplot3a.set_xlim(0, ai[-1])
    subplot3a.set_xticklabels([''])
    subplot3a.set_ylabel(r'Recoveries per day [$\Delta I_{\mathrm{recover}} / \Delta t$]', fontsize=16)
    subplot3a.set_ylim(0, np.max(data[:,6]))
    subplot3a.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=16, framealpha=0.5)
    
    subplot3b = fig3.add_subplot(212)
    subplot3b.plot(ai, data[:,8], color='k', markeredgecolor='k', marker='o', linestyle='', label='Data')
    subplot3b.plot(ai, deaths, color='k', alpha=0.6, label=r'$(I - %6.3f / %6.3g) \delta (t)$'%(abs(cd), d))
    subplot3b.set_xlabel(r'Active infections [$I$]', fontsize=16)
    subplot3b.set_xlim(0.0, ai[-1])
    subplot3b.set_ylabel(r'Deaths per day [$\Delta I_{\mathrm{death}} / \Delta t$]', fontsize=16)
    subplot3b.set_ylim(0.0, np.max(data[:,8]))
    subplot3b.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=16, framealpha=0.5)
    
    # Positive testing rate & percentage increase vs time
    fig4 = plt.figure()
    
    maks = 20.0
    subplot4a = fig4.add_subplot(211)
    subplot4a.plot(data[:,0], posTest, color='m', markeredgecolor='m', marker='o', linestyle='', label='Data')
    subplot4a.plot(data[1:-1,0], avg3ptest[1:-1], color='m', label='3 day running average')
    subplot4a.hlines(avgPosTestI, 0.0, 22.0, color='r', label=r'$%4.1f \pm %3.1f$'%(avgPosTestI, stdPosTestI))
    subplot4a.fill_between(data[0:23,0], avgPosTestI-stdPosTestI, avgPosTestI+stdPosTestI, color='r', alpha=0.2)
    subplot4a.hlines(avgPosTest5, 22.0, 57.0, color='g', label=r'$%4.1f \pm %3.1f$'%(avgPosTest5, stdPosTest5))
    subplot4a.fill_between(data[22:58,0], avgPosTest5-stdPosTest5, avgPosTest5+stdPosTest5, color='g', alpha=0.2)
    subplot4a.hlines(avgPosTest4, 57.0, data[-1,0], color='b', label=r'$%4.1f \pm %3.1f$'%(avgPosTest4, stdPosTest4))
    subplot4a.fill_between(data[57:,0], avgPosTest4-stdPosTest4, avgPosTest4+stdPosTest4, color='b', alpha=0.2)
    for i in range(len(days)):
        subplot4a.vlines(days[i], 0.0, maks, color='grey', linestyle='--')
    subplot4a.axvspan(36.0, 39.0, color='grey', alpha=0.3, label='Easter')
    subplot4a.set_title('South African COVID-19 Data', fontsize=18)
    subplot4a.set_xlim(data[0,0], data[-1,0])
    subplot4a.set_xticklabels([''])
    subplot4a.set_ylabel('Percentage positive tests', fontsize=16)
    subplot4a.set_ylim(0.0, maks)
    subplot4a.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.5)
    
    maks = 140.0
    subplot4b = fig4.add_subplot(212)
    subplot4b.plot(data[1:,0], prcntIn, color='m', markeredgecolor='m', marker='o', linestyle='')
    subplot4b.plot(data[1:-1,0], avg3incrs[1:-1], color='m')
    subplot4b.hlines(avgIncrsI, 0.0, 22.0, color='r', label=r'$%4.1f \pm %3.1f$'%(avgIncrsI, stdIncrsI))
    subplot4b.fill_between(data[0:23,0], avgIncrsI-stdPosTestI, avgIncrsI+stdPosTestI, color='r', alpha=0.2)
    subplot4b.hlines(avgIncrs5, 22.0, 57.0, color='g', label=r'$%4.1f \pm %3.1f$'%(avgIncrs5, stdIncrs5))
    subplot4b.fill_between(data[22:58,0], avgIncrs5-stdIncrs5, avgIncrs5+stdIncrs5, color='g', alpha=0.2)
    subplot4b.hlines(avgIncrs4, 57.0, data[-1,0], color='b', label=r'$%4.1f \pm %3.1f$'%(avgIncrs4, stdIncrs4))
    subplot4b.fill_between(data[57:,0], avgIncrs4-stdIncrs4, avgIncrs4+stdIncrs4, color='b', alpha=0.2)
    for i in range(len(days)):
        subplot4b.vlines(days[i], 0.0, maks, color='grey', linestyle='--', label=comments[i])
        subplot4b.text(days[i], maks, abc[i], fontsize=16)
    subplot4b.axvspan(36.0, 39.0, color='grey', alpha=0.3)
    subplot4b.set_xlabel('Days since first infection on 5 March', fontsize=16)
    subplot4b.set_xlim(data[0,0], data[-1,0])
    subplot4b.set_ylabel('Percentage increase in infections', fontsize=16)
    subplot4b.set_ylim(0.0, maks)
    subplot4b.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.5)
    
    # Time dependence of parameters
    mins = 0.5e-9
    maks = 4.5e-9
    fig5 = plt.figure()
    subplot5 = fig5.add_subplot(111)
    yax = subplot5.twinx()
    yax.spines['right'].set_position(('axes', 1.0))
    yax.set_frame_on(True)
    yax.patch.set_visible(False)
    for sp in yax.spines.values():
        sp.set_visible(False)
    yax.spines['right'].set_visible(True)
    lines = []
    line, = subplot5.plot(t, betat, color='r', label=r'$\beta (t) = \beta_i (t) + \beta_5 (t) + \beta_4 (t)$')
    lines += [line]
    subplot5.fill_between(t, betatl, betatu, color='r', alpha=0.2)
    line, = subplot5.plot(0.0, 0.0, color='r', label=r'$\beta_i (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{%6.1f - t}{%6.1f} \right) \right]$'%(b1, lckdwn5, transition))
    lines += [line]
    line, = subplot5.plot(0.0, 0.0, color='r', label=r'$\beta_5 (t) = \frac{%6.3g}{2} \left[ \tanh \left( \frac{t - %6.1f}{%6.1f} \right) - \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(b2, lckdwn5, transition, lckdwn4, transition))
    lines += [line]
    line, = subplot5.plot(0.0, 0.0, color='r', label=r'$\beta_4 (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(b3, lckdwn4, transition))
    lines += [line]
    line, = subplot5.plot(t, modbt, color='orange', label=r'$\beta (t)$ for SISD model to fit data closely')
    lines += [line]
    line, = yax.plot(t, rhot, color='g', label=r'$\rho (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(r, sicktime, transition))
    lines += [line]
    yax.fill_between(t, rhotl, rhotu, color='g', alpha=0.2)
    line, = yax.plot(t, modrt, color='c', label=r'$\rho (t)$ for SISD model to fit data closely')
    lines += [line]
    line, = yax.plot(t, deltat, color='k', alpha=0.6, label=r'$\delta (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(d, firstdeath, transition))
    lines += [line]
    yax.fill_between(t, deltatl, deltatu, color='k', alpha=0.2)
    line, = yax.plot(t, moddt, color='grey', label=r'$\delta (t)$ for SISD model to fit data closely')
    lines += [line]
    for i in range(len(days)):
        subplot5.vlines(days[i], mins, maks, color='grey', linestyle='--', label=comments[i])
        subplot5.text(days[i], maks, abc[i], fontsize=16)
    subplot5.axvspan(36.0, 39.0, color='grey', alpha=0.3, label='Easter')
    subplot5.set_xlabel(r'Days since first infection on 5 March [$t$]', fontsize=16)
    subplot5.set_xlim(t[0], t[-1])
    subplot5.set_ylabel(r'$\beta (t)$', fontsize=16)
    subplot5.set_ylim(mins, maks)
    yax.set_ylabel(r'$\rho (t)$ or $\delta (t)$', fontsize=16)
    subplot5.tick_params(labelsize=16, top=True, right=False, direction='inout')
    yax.tick_params(labelsize=16, top=True, right=True, direction='inout')
    subplot5.legend(lines, [l.get_label() for l in lines], loc=3, fontsize=16, framealpha=0.5)
    
    plt.show()
