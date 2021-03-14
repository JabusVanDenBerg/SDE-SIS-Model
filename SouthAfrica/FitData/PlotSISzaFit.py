"""Plot modelling results for SIS model of an infection outbreak with varying coefficients.

Model assumes that people susceptible to be infected (S) are infected (I) at a
rate of beta (% of people per day) and that infected people who recover at a
rate rho (recoveries per day), will be again susceptible to be infected, hence
(S --> I --> S). Let N be the total population which can be infected and let Ns
and Ni be the number of people susceptible and infected, respectively, then the
flow from S to I is beta*Ni*Ns = beta*Ni*(N - Ni), while the flow from I back to
S is rho*Ni.
"""


import numpy as np                         # Numerical Python package
import matplotlib.pyplot as plt            # Graphing package
from matplotlib import colors, ticker, cm  # Colour maps and bar for contour plots


################################################################################
# Plotting Function                                                            #
################################################################################
def Plot(result):
    """Function to plot the results.
    
    Plot(results)
        Plot the results.
    
    Arguments:
        result : A dictionary with the following keys:
                 'Nt'    : The number of time bins.
                 't'     : The mid points of the time bins.
                 'tbins' : The boundaries of the time bins. It is assumed that
                           the first value is the lower value of the first bin.
                 'Ni'    : The number of infection bins.
                 'I'     : The mid points of the infection bins.
                 'Ibins' : The boundaries of the infection bins. It is assumed
                           that the first value is the lower value of the first
                           bin.
                 'avgI'  : The average number of infections as a function of
                           time.
                 'stdI'  : The standard deviation of the number of infections
                           as a function of time.
                 'f'     : The probability distribution function of infections
                           as a function of time and the number of infections.
    """
    ############################################################################
    # Raw data and three day running average                                   #
    ############################################################################
    fig1 = plt.figure()
    
    maks = 3.0e4
    subplot1a = fig1.add_subplot(211)
    subplot1a.plot(result['dataT'], result['dataIt'], color='r', markeredgecolor='r', marker='o', linestyle='')
    subplot1a.plot(result['dataT'], result['dataIa'], color='m', markeredgecolor='m', marker='s', linestyle='')
    subplot1a.plot(result['dataT'], result['dataRt'], color='b', markeredgecolor='b', marker='^', linestyle='')
    subplot1a.plot(result['dataT'], result['dataDt'], color='k', markeredgecolor='k', marker='+', linestyle='')
    subplot1a.plot(result['dataT'][1:-1], results['3draIt'][1:-1], color='r', alpha=0.7)
    subplot1a.plot(result['dataT'][1:-1], results['3draIa'][1:-1], color='m', alpha=0.7)
    subplot1a.plot(result['dataT'][1:-1], results['3draRt'][1:-1], color='b', alpha=0.7)
    subplot1a.plot(result['dataT'][1:-1], results['3draDt'][1:-1], color='k', alpha=0.7)
    for i in range(len(result['days'])):
        subplot1a.axvline(result['days'][i], color='grey', linestyle='--', label=result['comments'][i])
    subplot1a.axvspan(36.0, 39.0, color='grey', alpha=0.3, label='Easter')
    subplot1a.set_title('South African COVID-19 infections, recoveries, & deaths', fontsize=18)
    subplot1a.set_xlim(result['dataT'][0], result['dataT'][-1])
    subplot1a.set_xticklabels([''])
    subplot1a.set_ylabel('Number of people', fontsize=16)
    subplot1a.set_ylim(0.0, maks)
    subplot1a.tick_params(labelsize=16, top=True, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.3)
    
    subplot1b = fig1.add_subplot(212)
    subplot1b.plot(result['dataT'], result['dataIt'], color='r', markeredgecolor='r', marker='o', linestyle='', label='Data: Infections (Total)')
    subplot1b.plot(result['dataT'], result['dataIa'], color='m', markeredgecolor='m', marker='s', linestyle='', label='Data: Infections (Active)')
    subplot1b.plot(result['dataT'], result['dataRt'], color='b', markeredgecolor='b', marker='^', linestyle='', label='Data: Recoveries (Total)')
    subplot1b.plot(result['dataT'], result['dataDt'], color='k', markeredgecolor='k', marker='+', linestyle='', label='Data: Deaths (Total)')
    subplot1b.plot(result['dataT'][1:-1], results['3draIt'][1:-1], color='r', alpha=0.7, label='3 day running average: Infections (Total)')
    subplot1b.plot(result['dataT'][1:-1], results['3draIa'][1:-1], color='m', alpha=0.7, label='3 day running average: Infections (Active)')
    subplot1b.plot(result['dataT'][1:-1], results['3draRt'][1:-1], color='b', alpha=0.7, label='3 day running average: Recoveries (Total)')
    subplot1b.plot(result['dataT'][1:-1], results['3draDt'][1:-1], color='k', alpha=0.7, label='3 day running average: Deaths (Total)')
    for i in range(len(result['days'])):
        subplot1b.axvline(result['days'][i], color='grey', linestyle='--')
        subplot1b.text(result['days'][i], maks, abc[i], fontsize=16)
    subplot1b.axvspan(36.0, 39.0, color='grey', alpha=0.3)
    subplot1b.set_xlabel('Days since first infection on 5 March', fontsize=16)
    subplot1b.set_xlim(result['dataT'][0], result['dataT'][-1])
    subplot1b.set_ylabel('Number of people', fontsize=16)
    subplot1b.set_ylim(1.0, maks)
    subplot1b.set_yscale('log')
    subplot1b.tick_params(labelsize=16, top=True, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.3)
    
    ############################################################################
    # Model results compared to three day running average of data              #
    ############################################################################
    y, x = np.meshgrid(result['I'], result['t'])
    clrmap = plt.get_cmap('Greens')
    level = ticker.MaxNLocator(nbins=100).tick_values(-10.0, 0.0)
    normal = colors.BoundaryNorm(level, ncolors=clrmap.N, clip=True)
    f = np.log10(result['f'])
    
    fig2 = plt.figure()
    
    subplot2a = fig2.add_subplot(211)
    subplot2a.contourf(x, y, f, levels=level, cmap=clrmap, norm=normal)
    subplot2a.plot(result['dataT'][1:-1], result['3draIt'][1:-1], color='r', markeredgecolor='r', marker='o', linestyle='', label='3d avg: Infections (Total)')
    subplot2a.plot(result['dataT'][1:-1], result['3draIa'][1:-1], color='m', markeredgecolor='m', marker='s', linestyle='', label='3d avg: Infections (Active)')
    subplot2a.plot(result['dataT'][1:-1], result['3draRt'][1:-1], color='b', markeredgecolor='b', marker='^', linestyle='', label='3d avg: Recoveries (Total)')
    subplot2a.plot(result['dataT'][1:-1], result['3draDt'][1:-1], color='k', markeredgecolor='k', marker='+', linestyle='', label='3d avg: Deaths (Total)')
    subplot2a.plot(result['t'], result['avgIt'], color='r', label='Average total infections')
    subplot2a.fill_between(result['t'], result['avgIt']-result['errIt'], result['avgIt']+result['errIt'], color='r', alpha=0.2)
    subplot2a.plot(result['t'], result['avgI'], color='m', label='Average active infections')
    subplot2a.fill_between(result['t'], result['avgI']-result['errI'], result['avgI']+result['errI'], color='m', alpha=0.2)
    subplot2a.plot(result['t'], result['probI'], color='c', label='Most probable active infections')
    subplot2a.fill_between(result['t'], result['probI']-result['errProbI'], result['probI']+result['errProbI'], color='c', alpha=0.4)
    subplot2a.plot(result['t'], result['avgR'], color='b', label='Average recoveries')
    subplot2a.fill_between(result['t'], result['avgR']-result['errR'], result['avgR']+result['errR'], color='b', alpha=0.2)
    subplot2a.plot(result['t'], result['avgD'], color='k', label='Average deaths')
    subplot2a.fill_between(result['t'], result['avgD']-result['errD'], result['avgD']+result['errD'], color='k', alpha=0.2)
    for i in range(len(result['days'])):
        subplot2a.axvline(result['days'][i], color='grey', linestyle='--')
    subplot2a.axvspan(36.0, 39.0, color='grey', alpha=0.3)
    subplot2a.set_title('Modelled South African COVID-19 infections, recoveries, & deaths', fontsize=18)
    subplot2a.set_xlim(result['t'][0], result['t'][-2])
    subplot2a.set_xticklabels([''])
    subplot2a.set_ylabel('Number of people', fontsize=16)
    subplot2a.set_ylim(0.0, maks)
    subplot2a.tick_params(labelsize=16, top=True, right=True, direction='inout')
    plt.legend(loc=2, fontsize=14, framealpha=0.3)
    
    subplot2b = fig2.add_subplot(212)
    cf = subplot2b.contourf(x, y, f, levels=level, cmap=clrmap, norm=normal)
    subplot2b.plot(result['dataT'][1:-1], result['3draIt'][1:-1], color='r', markeredgecolor='r', marker='o', linestyle='')
    subplot2b.plot(result['dataT'][1:-1], result['3draIa'][1:-1], color='m', markeredgecolor='m', marker='s', linestyle='')
    subplot2b.plot(result['dataT'][1:-1], result['3draRt'][1:-1], color='b', markeredgecolor='b', marker='^', linestyle='')
    subplot2b.plot(result['dataT'][1:-1], result['3draDt'][1:-1], color='k', markeredgecolor='k', marker='+', linestyle='')
    subplot2b.plot(result['t'], result['avgIt'], color='r')
    subplot2b.fill_between(result['t'], result['avgIt']-result['errIt'], result['avgIt']+result['errIt'], color='r', alpha=0.2)
    subplot2b.plot(result['t'], result['avgI'], color='m')
    subplot2b.fill_between(result['t'], result['avgI']-result['errI'], result['avgI']+result['errI'], color='m', alpha=0.2)
    subplot2b.plot(result['t'], result['probI'], color='c')
    subplot2b.fill_between(result['t'], result['probI']-result['errProbI'], result['probI']+result['errProbI'], color='c', alpha=0.4)
    subplot2b.plot(result['t'], result['avgR'], color='b')
    subplot2b.fill_between(result['t'], result['avgR']-result['errR'], result['avgR']+result['errR'], color='b', alpha=0.2)
    subplot2b.plot(result['t'], result['avgD'], color='k')
    subplot2b.fill_between(result['t'], result['avgD']-result['errD'], result['avgD']+result['errD'], color='k', alpha=0.2)
    for i in range(len(result['days'])):
        subplot2b.axvline(result['days'][i], color='grey', linestyle='--')
        subplot2b.text(result['days'][i], 1.0e5, abc[i], fontsize=16)
    subplot2b.axvspan(36.0, 39.0, color='grey', alpha=0.3)
    subplot2b.set_xlabel('Days since first infection on 5 March', fontsize=16)
    subplot2b.set_xlim(result['t'][0], result['t'][-2])
    subplot2b.set_ylabel('Number of people', fontsize=16)
    subplot2b.set_ylim(1.0, 1.0e5)
    subplot2b.set_yscale('log')
    subplot2b.tick_params(labelsize=16, top=True, right=True, direction='inout')
    
    cbar = fig2.colorbar(cf, ax=(subplot2a, subplot2b), fraction=0.05)
    cbar.set_label('log(Probability) [not normalised]', fontsize=16)
    
    ############################################################################
    # Factor of unknown cases                                                  #
    ############################################################################
    maks = 2.0
    fig3 = plt.figure()
    subplot3 = fig3.add_subplot(111)
    subplot3.plot(result['t'][1:-1], result['probRatio'], color='g', label=r'$I_{\rm peak} / \bar{I}$ $=$ $%4.2f$'%(np.average(result['probRatio'][10:])))
    subplot3.plot(result['t'][1:-1], result['uncrRatio'], color='b', label=r'$(\bar{I} + \delta \bar{I}) / \bar{I}$ $=$ $%4.2f$'%(np.average(result['uncrRatio'][10:])))
    subplot3.plot(result['t'][1:-1], result['stdvRatio'], color='r', label=r'$(\bar{I} + \sigma_I) / \bar{I}$ $=$ $%4.2f$'%(np.average(result['stdvRatio'][10:])))
    for i in range(len(result['days'])):
        subplot3.axvline(result['days'][i], color='grey', linestyle='--')
        subplot3.text(result['days'][i], maks, abc[i], fontsize=16)
    subplot3.axvspan(36.0, 39.0, color='grey', alpha=0.3)
    subplot3.set_xlabel('Days since first infection on 5 March', fontsize=16)
    subplot3.set_xlim(result['t'][0], result['t'][-2])
    subplot3.set_ylabel('Estimated factor of unknown infection', fontsize=16)
    subplot3.set_ylim(1.0, maks)
    subplot3.tick_params(labelsize=16, top=True, right=True, direction='inout')
    plt.legend(loc=0, fontsize=16, framealpha=0.3)
    
    plt.show()


################################################################################
# Only execute the code if the script is ran and not if it is imported         #
################################################################################
if __name__ == "__main__":
    import argparse  # Command line argument parser package
    import pickle    # Object file saving package
    
    ############################################################################
    # Give the program the ability to accept command line arguments            #
    ############################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Result file name.')
    args = parser.parse_args()
    
    ############################################################################
    # Read results from file                                                   #
    ############################################################################
    print("Reading results ...")
    with open(args.filename, 'rb') as fh:
        results = pickle.load(fh)
    
    data = np.loadtxt("Data.txt", usecols=(1, 2, 6, 8))
    results['dataT'] = data[:,0]
    results['dataIt'] = data[:,1]
    results['dataIa'] = data[:,1] - data[:,2] - data[:,3]
    results['dataRt'] = data[:,2]
    results['dataDt'] = data[:,3]
    
    avg3infct = np.zeros_like(data[:,0])
    avg3recov = np.zeros_like(avg3infct)
    avg3death = np.zeros_like(avg3infct)
    for i in range(1, len(avg3infct)-1):
        avg3infct[i] = np.average(data[i-1:i+2,1])
        avg3recov[i] = np.average(data[i-1:i+2,2])
        avg3death[i] = np.average(data[i-1:i+2,3])
    results['3draIt'] = avg3infct
    results['3draIa'] = avg3infct - avg3recov - avg3death
    results['3draRt'] = avg3recov
    results['3draDt'] = avg3death
    
    days = [10, 13, 18, 22, 35, 40, 53, 56]
    comments = ['A: State of disaster', 'B: Schools and universities close',
                'C: lvl5 lockdown announced', 'D: lvl5 lockdown started',
                'E: lvl5 lockdown extension announced',
                'F: Mines started with 50% productivity', 'G: Freedom day',
                'H: Workers day / lvl4 started']
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    results['days'] = days
    results['comments'] = comments
    results['abc'] = abc
    
    
    ############################################################################
    # Plot the data                                                            #
    ############################################################################
    print("Plotting results ...")
    Plot(results)
