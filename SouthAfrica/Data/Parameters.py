"""Extract modelling parameters for SIS model of an infection outbreak.

Model assumes that people susceptible to be infected (S) are infected (I) at a
rate of beta (% of people per day) and that infected people who recover at a
rate rho (recoveries per day), will be again susceptible to be infected, hence
(S --> I --> S). Let N be the total population which can be infected and let Ns
and Ni be the number of people susceptible and infected, respectively, then the
flow from S to I is beta*Ni*Ns = beta*Ni*(N - Ni), while the flow from I back to
S is rho*Ni.
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
    N = 0.0
    for i in range(0, 23):  # Remove nan or inf values when calculating average
        if np.isfinite(posTest[i]):
            s += posTest[i]
            s2 += posTest[i]**2
            N += 1.0
    avgPosTestI = s/N
    # Uncertainty on average is standard deviation divided by the square-root of
    # the number of data points used to calculate the average
    stdPosTestI = np.sqrt((s2/N - avgPosTestI**2)/N)
    s = 0.0
    s2 = 0.0
    N = 0.0
    for i in range(23, 57):
        if np.isfinite(posTest[i]):
            s += posTest[i]
            s2 += posTest[i]**2
            N += 1.0
    avgPosTest5 = s/N
    stdPosTest5 = np.sqrt((s2/N - avgPosTest5**2)/N)
    s = 0.0
    s2 = 0.0
    N = 0.0
    for i in range(57, len(posTest)):
        if np.isfinite(posTest[i]):
            s += posTest[i]
            s2 += posTest[i]**2
            N += 1.0
    avgPosTest4 = s/N
    stdPosTest4 = np.sqrt((s2/N - avgPosTest4**2)/N)
    print [avgPosTestI, stdPosTestI], [avgPosTest5, stdPosTest5], [avgPosTest4, stdPosTest4]
    
    # 3 day running average of infections, recoveries, deaths, and positive tests
    avg3infct = np.zeros_like(data[:,2])
    avg3recov = np.zeros_like(avg3infct)
    avg3death = np.zeros_like(avg3infct)
    avg3ptest = np.zeros_like(avg3infct)
    for i in range(1, len(avg3infct)-1):
        avg3infct[i] = (data[i-1,2] + data[i,2] + data[i+1,2])/3.0
        avg3recov[i] = (infecRecov[i-1] + infecRecov[i] + infecRecov[i+1])/3.0
        avg3death[i] = (data[i-1,8] + data[i,8] + data[i+1,8])/3.0
        avg3ptest[i] = (posTest[i-1] + posTest[i] + posTest[i+1])/3.0
    
    # Slope and intercept of infection rate vs active infections times susceptibles
    # (First calculate with formula to have initial values and then with fitting
    #  function to have uncertainties)
    b1, c1 = SlopeIntercept(sai[:23], data[:23,2])
    #print b1, c1
    fit, cov = curve_fit(StraightLine, sai[:23], data[:23,2], p0=[1.1*b1, 1.1*c1])
    b1 = fit[0]
    c1 = fit[1]
    cov = np.sqrt(np.diag(cov))
    eb1 = cov[0]
    ec1 = cov[1]
    print fit, cov, (cov/np.abs(fit))*100.0
    b2, c2 = SlopeIntercept(sai[23:57], data[23:57,2])
    #print b2, c2
    fit, cov = curve_fit(StraightLine, sai[23:57], data[23:57,2], p0=[1.1*b2, 1.1*c2])
    b2 = fit[0]
    c2 = fit[1]
    cov = np.sqrt(np.diag(cov))
    eb2 = cov[0]
    ec2 = cov[1]
    print fit, cov, (cov/np.abs(fit))*100.0
    b3, c3 = SlopeIntercept(sai[57:], data[57:,2])
    #print b3, c3
    fit, cov = curve_fit(StraightLine, sai[57:], data[57:,2], sigma=np.sqrt(data[57:,2]), p0=[1.1*b3, 1.1*c3])
    b3 = fit[0]
    c3 = fit[1]
    cov = np.sqrt(np.diag(cov))
    eb3 = cov[0]
    ec3 = cov[1]
    print fit, cov, (cov/np.abs(fit))*100.0
    
    # Slope and intercept of recovery rate vs active infections
    r, cr = SlopeIntercept(ai[15:], infecRecov[15:])
    #print r, cr
    fit, cov = curve_fit(StraightLine, ai[15:], infecRecov[15:], p0=[1.1*r, 1.1*cr])
    r = fit[0]
    cr = fit[1]
    cov = np.sqrt(np.diag(cov))
    er = cov[0]
    ecr = cov[1]
    print fit, cov, (cov/np.abs(fit))*100.0
    
    # Slope and intercept of death rate vs active infections
    d, cd = SlopeIntercept(ai[21:], data[21:,8])
    #print d, cd
    fit, cov = curve_fit(StraightLine, ai[21:], data[21:,8], p0=[1.1*d, 1.1*cd])
    d = fit[0]
    cd = fit[1]
    cov = np.sqrt(np.diag(cov))
    ed = cov[0]
    ecd = cov[1]
    print fit, cov, (cov/np.abs(fit))*100.0
    
    ############################################################################
    # Parameters                                                               #
    ############################################################################
    # Time dependence of infection rate
    def Beta(t, betai, lckdwn5, beta5, lckdwn4, beta4, trans):
        initial = betai*0.5*(1.0 + np.tanh((lckdwn5 - t)/trans))
        lvl5 = beta5*0.5*(np.tanh((t - lckdwn5)/trans) - np.tanh((t - lckdwn4)/trans))
        lvl4 = beta4*0.5*(1.0 + np.tanh((t - lckdwn4)/trans))
        return initial + lvl5 + lvl4
    
    def InfectionRate(t, x, betai, ini, lckdwn5, beta5, in5, lckdwn4, beta4, in4, trans):
        initial = (betai*x + ini)*0.5*(1.0 + np.tanh((lckdwn5 - t)/trans))
        lvl5 = (beta5*x + in5)*0.5*(np.tanh((t - lckdwn5)/trans) - np.tanh((t - lckdwn4)/trans))
        lvl4 = (beta4*x + in4)*0.5*(1.0 + np.tanh((t - lckdwn4)/trans))
        return initial + lvl5 + lvl4
    
    # Time dependence of recovery rate
    def Rho(t, rho, delay, trans):
        return rho*0.5*(1.0 + np.tanh((t - delay)/trans))
    
    def RecoveryRate(t, x, rho, offset, delay, trans):
        return (rho*x + offset)*0.5*(1.0 + np.tanh((t - delay - 0.5)/trans))
    
    lckdwn5 = 22.5
    lckdwn4 = 57.0
    sicktime = 19.0
    firstdeath = 22.0
    transition = 0.5
    t = np.arange(data[0,0], data[-1,0], 0.1)
    betat = Beta(t, b1, lckdwn5, b2, lckdwn4, b3, transition)
    betatu = Beta(t, b1+eb1, lckdwn5, b2+eb2, lckdwn4, b3+eb3, transition)
    betatl = Beta(t, b1-eb1, lckdwn5, b2-eb2, lckdwn4, b3-eb3, transition)
    rhot = Rho(t, r, sicktime, transition)
    rhotu = Rho(t, r+er, sicktime, transition)
    rhotl = Rho(t, r-er, sicktime, transition)
    deltat = Rho(t, d, firstdeath, transition)
    deltatu = Rho(t, d+ed, firstdeath, transition)
    deltatl = Rho(t, d-ed, firstdeath, transition)
    
    # Modelled data
    nwnfct = InfectionRate(data[:,0], sai, b1, c1, lckdwn5, b2, c2, lckdwn4, b3, c3, transition)
    recov = RecoveryRate(data[:,0], ai, r, cr, sicktime, transition)
    deaths = RecoveryRate(data[:,0], ai, d, cd, firstdeath, transition)
    
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
    #subplot1a.set_ylabel(r'New infections per day [$\Delta I_{\mathrm{new}} / \Delta t$]', fontsize=16)
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
    #subplot1b.set_ylabel(r'Recoveries per day [$\Delta I_{\mathrm{recover}} / \Delta t$]', fontsize=16)
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
    #subplot1c.set_ylabel(r'Deaths per day [$\Delta I_{\mathrm{death}} / \Delta t$]', fontsize=16)
    subplot1c.set_ylim(0.0, maks)
    subplot1c.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.5)
    
    # Infection rate vs active infections times susceptibles
    fig2 = plt.figure()
    subplot2 = fig2.add_subplot(111)
    subplot2.plot(sai[:23], data[:23,2], color='r', markeredgecolor='r', marker='o', linestyle='', label='Data: 5 - 27 March (before lvl5 lockdown)')
    subplot2.plot(sai[23:57], data[23:57,2], color='orange', markeredgecolor='orange', marker='o', linestyle='', label='Data: 28 March - 30 April (lvl5 lockdown)')
    subplot2.plot(sai[57:], data[57:,2], color='c', markeredgecolor='c', marker='o', linestyle='', label='Data: 29 April - 17 May (lvl4 lockdown)')
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
    
    # Correlations between tests and infections?
    fig4 = plt.figure()
    subplot4 = fig4.add_subplot(111)
    subplot4.plot(data[:23,4], data[:23,2], color='r', markeredgecolor='r', marker='o', linestyle='', label='Data: 5 - 27 March (before lvl5 lockdown)')
    subplot4.plot(data[23:38,4], data[23:38,2], color='orange', markeredgecolor='orange', marker='o', linestyle='', label='Data: 28 March - 11 April (few tests)')
    subplot4.plot(data[38:55,4], data[38:55,2], color='b', markeredgecolor='b', marker='o', linestyle='', label='Data: 12 - 28 April (majority of lvl5 lockdown)')
    subplot4.plot(data[55:,4], data[55:,2], color='g', markeredgecolor='g', marker='o', linestyle='', label='Data: 29 April - 17 May (increased testing)')
    subplot4.set_title('South African COVID-19 Data', fontsize=18)
    subplot4.set_xlabel('Tests per day', fontsize=16)
    subplot4.set_xlim(0.0, np.max(data[:,4]))
    subplot4.set_ylabel('New infections per day', fontsize=16)
    subplot4.set_ylim(0.0, np.max(data[:,2]))
    #subplot4.set_yscale('log')
    subplot4.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=16, framealpha=0.5)
    
    # Positive testing rate vs time
    maks = 20.0
    fig5 = plt.figure()
    subplot5 = fig5.add_subplot(111)
    subplot5.plot(data[:,0], posTest, color='m', markeredgecolor='m', marker='o', linestyle='', label='Data')
    subplot5.plot(data[1:-1,0], avg3ptest[1:-1], color='m', label='3 day running average')
    subplot5.hlines(avgPosTestI, 0.0, 22.0, color='r', label=r'$%4.1f \pm %3.1f$'%(avgPosTestI, stdPosTestI))
    subplot5.fill_between(data[0:23,0], avgPosTestI-stdPosTestI, avgPosTestI+stdPosTestI, color='r', alpha=0.2)
    subplot5.hlines(avgPosTest5, 22.0, 57.0, color='g', label=r'$%4.1f \pm %3.1f$'%(avgPosTest5, stdPosTest5))
    subplot5.fill_between(data[22:58,0], avgPosTest5-stdPosTest5, avgPosTest5+stdPosTest5, color='g', alpha=0.2)
    subplot5.hlines(avgPosTest4, 57.0, data[-1,0], color='b', label=r'$%4.1f \pm %3.1f$'%(avgPosTest4, stdPosTest4))
    subplot5.fill_between(data[57:,0], avgPosTest4-stdPosTest4, avgPosTest4+stdPosTest4, color='b', alpha=0.2)
    for i in range(len(days)):
        subplot5.vlines(days[i], 0.0, maks, color='grey', linestyle='--', label=comments[i])
        subplot5.text(days[i], maks, abc[i], fontsize=16)
    subplot5.axvspan(36.0, 39.0, color='grey', alpha=0.3, label='Easter')
    #subplot5.set_title('South African COVID-19 Data', fontsize=18)
    subplot5.set_xlabel('Days since first infection on 5 March', fontsize=16)
    subplot5.set_xlim(data[0,0], data[-1,0])
    subplot5.set_ylabel('Percentage positive tests', fontsize=16)
    subplot5.set_ylim(0.0, maks)
    subplot5.tick_params(labelsize=16, top=False, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.5)
    
    # Time dependence of parameters
    mins = 0.5e-9
    maks = 4.0e-9
    fig6 = plt.figure()
    subplot6 = fig6.add_subplot(111)
    yax = subplot6.twinx()
    yax.spines['right'].set_position(('axes', 1.0))
    yax.set_frame_on(True)
    yax.patch.set_visible(False)
    for sp in yax.spines.values():
        sp.set_visible(False)
    yax.spines['right'].set_visible(True)
    lines = []
    line, = subplot6.plot(t, betat, color='r', label=r'$\beta (t) = \beta_i (t) + \beta_5 (t) + \beta_4 (t)$')
    lines += [line]
    subplot6.fill_between(t, betatl, betatu, color='r', alpha=0.2)
    line, = subplot6.plot(0.0, 0.0, color='r', label=r'$\beta_i (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{%6.1f - t}{%6.1f} \right) \right]$'%(b1, lckdwn5, transition))
    lines += [line]
    line, = subplot6.plot(0.0, 0.0, color='r', label=r'$\beta_5 (t) = \frac{%6.3g}{2} \left[ \tanh \left( \frac{t - %6.1f}{%6.1f} \right) - \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(b2, lckdwn5, transition, lckdwn4, transition))
    lines += [line]
    line, = subplot6.plot(0.0, 0.0, color='r', label=r'$\beta_4 (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(b3, lckdwn4, transition))
    lines += [line]
    line, = yax.plot(t, rhot, color='g', label=r'$\rho (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(r, sicktime, transition))
    lines += [line]
    yax.fill_between(t, rhotl, rhotu, color='g', alpha=0.2)
    line, = yax.plot(t, deltat, color='k', alpha=0.6, label=r'$\delta (t) = \frac{%6.3g}{2} \left[1 + \tanh \left( \frac{t - %6.1f}{%6.1f} \right) \right]$'%(d, firstdeath, transition))
    lines += [line]
    yax.fill_between(t, deltatl, deltatu, color='k', alpha=0.2)
    for i in range(len(days)):
        subplot6.vlines(days[i], mins, maks, color='grey', linestyle='--', label=comments[i])
        subplot6.text(days[i], maks, abc[i], fontsize=16)
    subplot6.axvspan(36.0, 39.0, color='grey', alpha=0.3, label='Easter')
    #subplot6.set_title('South African COVID-19 Data', fontsize=18)
    subplot6.set_xlabel(r'Days since first infection on 5 March [$t$]', fontsize=16)
    subplot6.set_xlim(t[0], t[-1])
    subplot6.set_ylabel(r'$\beta (t)$', fontsize=16)
    subplot6.set_ylim(mins, maks)
    yax.set_ylabel(r'$\rho (t)$ or $\delta (t)$', fontsize=16)
    subplot6.tick_params(labelsize=16, top=True, right=False, direction='inout')
    yax.tick_params(labelsize=16, top=True, right=True, direction='inout')
    subplot6.legend(lines, [l.get_label() for l in lines], loc=3, fontsize=16, framealpha=0.5)
    
    plt.show()
