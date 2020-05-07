"""Plot modelling results for SIS model of an infection outbreak.

Model assumes that people susceptible to be infected (S) are infected (I) at a
rate of beta (% of people per day) and that infected people who recover at a
rate rho (recoveries per day), will be again susceptible to be infected, hence
(S --> I --> S). Let N be the total population which can be infected and let Ns
and Ni be the number of people susceptible and infected, respectively, then the
flow from S to I is beta*Ni*Ns = beta*Ni*(N - Ni), while the flow from I back to
S is rho*Ni.
"""


import numpy as np                     # Numerical Python package
import matplotlib.pyplot as plt        # Graphing package
from matplotlib import colors, ticker  # Colour maps and bar for contour plots


def Plot(result):
    """Function to plot the results.
    
    Plot(results)
        Plot the results.
    
    Arguments:
        result : A dictionary with the following keys:
                 't'     : The mid points of the time bins.
                 'I'     : The mid points of the infection bins.
                 'avgI'  : The average number of infections as a function of
                           time.
                 'stdI'  : The standard deviation of the number of infections
                           as a function of time.
                 'f'     : The probability distribution function of infections
                           as a function of time and the number of infections.
                 'beta'  : The infection rate.
                 'rho'   : The recovery rate.
    """
    maks = result['N'] - result['rho']/result['beta']
    y, x = np.meshgrid(result['I'], result['t'])
    clrmap = plt.get_cmap('Greens')
    level = ticker.MaxNLocator(nbins=100).tick_values(-10.0, 0.0)
    normal = colors.BoundaryNorm(level, ncolors=clrmap.N, clip=True)
    f = np.log10(result['f'])

    fig = plt.figure()
    
    subplot1 = fig.add_subplot(211)
    subplot1.contourf(x, y, f, levels=level, cmap=clrmap, norm=normal)
    subplot1.plot(result['t'], result['avgI'], color='r', label='Average')
    subplot1.fill_between(result['t'], result['avgI']-0.5*result['stdI'], result['avgI']+0.5*result['stdI'], color='r', alpha=0.2)
    subplot1.plot(result['t'], result['probI'], color='orange', label='Most probable')
    subplot1.fill_between(result['t'], result['pIl'], result['pIu'], color='orange', alpha=0.2)
    subplot1.axhline(maks, color='grey', label='Maximum number of infection: '+str(int(maks)))
    subplot1.set_xlim(result['t'][0], result['t'][-2])
    subplot1.set_xticklabels([''])
    subplot1.set_ylabel('Number of active infections', fontsize=16)
    subplot1.set_ylim(1.0, 10.0**np.ceil(np.log10(maks)))
    subplot1.tick_params(labelsize=16, top=True, right=True, direction='inout')
    plt.legend(loc=0, fontsize=14, framealpha=0.3)
    
    subplot2 = fig.add_subplot(212)
    cf = subplot2.contourf(x, y, f, levels=level, cmap=clrmap, norm=normal)
    subplot2.plot(result['t'], result['avgI'], color='r')
    subplot2.fill_between(result['t'], result['avgI']-0.5*result['stdI'], result['avgI']+0.5*result['stdI'], color='r', alpha=0.2)
    subplot2.plot(result['t'], result['probI'], color='orange')
    subplot2.fill_between(result['t'], result['pIl'], result['pIu'], color='orange', alpha=0.2)
    subplot2.axhline(maks, color='grey')
    subplot2.set_xlabel('Time since first infection [days]', fontsize=16)
    subplot2.set_xlim(result['t'][0], result['t'][-2])
    subplot2.set_ylabel('Number of active infections', fontsize=16)
    subplot2.set_ylim(1.0, 10.0**np.ceil(np.log10(maks)))
    subplot2.set_yscale('log')
    subplot2.tick_params(labelsize=16, top=True, right=True, direction='inout')
    
    cbar = fig.colorbar(cf, ax=(subplot1, subplot2), fraction=0.05)
    cbar.set_label('log(Probability) [not normalised]', fontsize=16)

    plt.show()


################################################################################
# Only execute the code if the script is ran and not if it is imported         #
################################################################################
if __name__ == "__main__":
    import argparse  # Command line argument parser package
    import pickle    # Object file saving package
    import matplotlib.pyplot as plt  # Graphing package
    
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
    
    ############################################################################
    # Plot the data                                                            #
    ############################################################################
    print("Plotting results ...")
    Plot(results)
