"""SISD model for modelling an infection outbreak with varying coefficients.

Model assumes that people susceptible to be infected (S) are infected (I) at a
rate of beta (% of people per day), that infected people who recover at a rate
rho (recoveries per day), will be again susceptible to be infected, and that
infected people can die (D) at a rate delta (deaths per day), hence (S -> I -> S
or S -> I -> D). Let N be the total population which can be infected and let Ns
and Ni be the number of people susceptible and infected, respectively, then the
flow from S to I is beta*Ni*Ns = beta*Ni*(N - Ni) and the flow from I back to S
is rho*Ni, while the flow from I to D is delta*Ni.
    
The stochastic differential equation (SDE) governing the number of infected (X),
is dX(t) = a(X,t)*dt + b(X,t)*dW, where a(x,t) = beta(t)*x*(N - x) - (rho(t) +
delta(t))*x and b(x,t) = sqrt(beta(t)*x*(N - x) + rho(t)*x).
"""


import numpy as np    # Numerical Python package
import random as rndm # Random number generator package


def InfectionRate(t, x, betas, times, trans):
    # Infection rate
    beta = betas[0]*0.5*(1.0 + np.tanh((times[0] - t)/trans))
    for i in range(1, len(times)):
        beta += betas[i]*0.5*(np.tanh((t - times[i-1])/trans) - np.tanh((t - times[i])/trans))
    beta += betas[-1]*0.5*(1.0 + np.tanh((t - times[-1])/trans))
    # Modifications
    time = [4.0, 13.0, 22.0, 36.0, 43.0, 58.0, 77.0,]
    mods = [0.5,  1.0, 1.15, 0.35,  0.5, 0.65, 0.85, 0.8]
    mod = mods[0]*0.5*(1.0 + np.tanh((time[0] - t)/trans))
    for i in range(1, len(time)):
        mod += mods[i]*0.5*(np.tanh((t - time[i-1])/trans) - np.tanh((t - time[i])/trans))
    mod += mods[-1]*0.5*(1.0 + np.tanh((t - time[-1])/trans))
    return mod*beta*x
    
def RecoveryRate(t, x, rho, delay, trans):
    r = rho*0.5*(1.0 + np.tanh((t - delay + 5.0)/trans))
    # Modifications
    time = [34.0, 69.0]
    mods = [0.05, 0.6, 1.0]
    mod = mods[0]*0.5*(1.0 + np.tanh((time[0] - t)/trans))
    for i in range(1, len(time)):
        mod += mods[i]*0.5*(np.tanh((t - time[i-1])/trans) - np.tanh((t - time[i])/trans))
    mod += mods[-1]*0.5*(1.0 + np.tanh((t - time[-1])/trans))
    return mod*r*x

def DeathRate(t, x, delta, delay, trans):
    d = delta*0.5*(1.0 + np.tanh((t - delay - 0.5)/trans))
    # Modifications
    time = [40.0, 53.0, 80.0]
    mods = [0.4,   0.8,  0.7, 0.8]
    mod = mods[0]*0.5*(1.0 + np.tanh((time[0] - t)/trans))
    for i in range(1, len(time)):
        mod += mods[i]*0.5*(np.tanh((t - time[i-1])/trans) - np.tanh((t - time[i])/trans))
    mod += mods[-1]*0.5*(1.0 + np.tanh((t - time[-1])/trans))
    return mod*d*x

def Coefficients(t, x, args):
    """Function to calculate the drift and diffusion coefficients.
    
    Coefficients(t, x, args)
        Calculate the drift and diffusion coefficients for the SDE.
    
    Arguments:
        t    : The current time.
        x    : The number of infected.
        args : Additional arguments needed by the function. The order of the
               arguments are:
               N     : The total population.
               beta  : A dictionary with the parameters used in calculating the
                       infection rate.
               rho   : A dictionary with the parameters used in calculating the
                       recovery rate.
               delta : A dictionary with the parameters used in calculating the
                       death rate.
               dt    : The simulation time step.
    
    Returns:
        A tuple with the drift and diffusion coefficients and the time step.
    """
    dIn = InfectionRate(t, x*(args[0] - x), **args[1])
    dIr = RecoveryRate(t, x, **args[2])
    dId = DeathRate(t, x, **args[3])
    a = dIn - dIr - dId
    b = np.sqrt(dIn + dIr)
    return (a, b, args[-1])


def Boundary(t, x, args):
    """Function to apply boundary conditions.
    
    Boundary(t, x, args)
        Apply boundary conditions to the SDE.
    
    Arguments:
        t    : The current time.
        x    : The number of infected.
        args : Additional arguments needed by the function. The order of the
               arguments are:
               N : The total population.
               T : The maximum simulation time.
    
    Returns:
        A tuple with the status (0 if the calculations must be continued and 1
        if the calculation must be stopped) and the updated (according to the
        boundary conditions if necessary) number of infected.
    """
    # Infected cannot be negative or more than the total population
    if (x < 0.0):
        x = -x
    elif (x > args[0]):
        x = args[0]
    # Temporal boundary
    if (t < args[1]):
        return (0, x)
    else:
        return (1, x)


def Bin(results, k, t, x, status, args):
    """Function to bin results.
    
    Bin(results, k, t, x, status, args)
        Bin SDE results.
        NOTE : This function must be called a final time with status equal to 2
               and k equal to the total number of solutions at the end to
               finalise the calculations.
    
    Arguments:
        results : Dictionary used to save results in.
        k       : Solution number or the total number of solutions.
        t       : The current time.
        x       : The number of infected.
        status  : The current status of the solution (0: Calculation continues,
                  1: Calculation done, 2: Results can be finalized)
        args    : Additional arguments needed by the function. The order of the
                  arguments are:
                  N     : The total population.
                  T     : The maximum simulation time.
                  dt    : The time step.
                  rho   : A dictionary with the parameters used in calculating
                          the recovery rate.
                  delta : A dictionary with the parameters used in calculating
                          the death rate.
    
    Returns:
        The updated results dictionary.
    """
    def BinIndex(linlog, value, mid, bins, width, N):
        """Function to find the bin index of a value which must be binned
        
        BinIndex(linlog, value, mid, bins, width, N)
            Find the bin index of a value which must be binned.
        
        Arguments:
            linlog : A string to indicate whether the bins are linearly ('lin')
                     or logarithmically ('log') spaced.
            value  : The value which must be binned.
            mid    : The mid points of the bins.
            bins   : The bounds of the bins. It is assumed that the first values
                     is the lower bound of the first bin.
            width  : The bin width. If logarithmic bins are used, then this must
                     be the constant logarithmic bin width.
            N      : The number of bins.
        
        Returns:
            The index of the bin or -1 if the value does not fall into a bin.
        """
        if (linlog == 'lin'): # Initial guess
            indx = int(abs(value - mid[0])/width)
        elif (linlog == 'log'):
            if (value == 0.0):
                indx = int(np.log10(abs(value - mid[0]))/width)
            else:
                indx = int(abs(np.log10(value) - np.log10(mid[0]))/width)
        while (True): # Find bin
            # Bin not found
            if ((indx < 0) or (indx > N-1)):
                return -1
            # Test current bin
            if ((value > bins[indx]) and (value <= bins[indx+1])):
                return indx
            # Adjust index to search neighbouring bins
            elif (value < bins[indx]):
                indx -= 1
            elif (value >= bins[indx+1]):
                indx += 1
    
    # Initial set up
    if (len(results) == 0):
        results['dT'] = 0.5                                      # Time bin width
                                                                 # Time bin mid points
        results['t'] = np.arange(0.0, args[1]+results['dT'], results['dT'])
        results['Nt'] = len(results['t'])                        # Number of time bins
        results['tbins'] = np.zeros(results['Nt'] + 1)           # Time bins
        results['tbins'][:-1] = results['t'] - 0.5*results['dT']
        results['tbins'][-1] = results['t'][-1] + 0.5*results['dT']
        results['avgI'] = np.zeros(results['Nt'])                # Average active infections vs time
        results['stdI'] = np.zeros(results['Nt'])                # Standard deviation of active infections
        results['probI'] = np.zeros(results['Nt'])               # Most probable active infections vs time
        results['avgR'] = np.zeros(results['Nt'])                # Average recoveries vs time
        results['errR'] = np.zeros(results['Nt'])                # Uncertainties of recoveries
        results['avgD'] = np.zeros(results['Nt'])                # Average deaths vs time
        results['errD'] = np.zeros(results['Nt'])                # Uncertainties of recoveries
        results['Ibins'] = np.logspace(0.0, 5.0, num=250)        # Infection bins
                                                                 # Logarithm of infection bin width
        results['dI'] = np.log10(results['Ibins'][1]) - np.log10(results['Ibins'][0])
        results['Ni'] = len(results['Ibins']) - 1                # Number of infection bins
        results['I'] = np.zeros(results['Ni'])                   # Infection bin mid points
        for i in range(results['Ni']):
            results['I'][i] = 0.5*(np.log10(results['Ibins'][i+1]) + np.log10(results['Ibins'][i]))
        results['I'] = 10**results['I']
        results['f'] = np.zeros((results['Nt'], results['Ni']))  # Distribution
    # Bin result
    if ((status == 0) or (status == 1)):
        n = BinIndex('lin', t, results['t'], results['tbins'], results['dT'], results['Nt'])
        if (n >= 0):
            results['avgI'][n] += x
            results['stdI'][n] += x**2
            i = BinIndex('log', x, results['I'], results['Ibins'], results['dI'], results['Ni'])
            if (i >= 0):
                results['f'][n,i] += 1.0
    # Do final calculations
    elif (status == 2):
        k = float(k)
        # First and last bins are only half bins
        results['avgI'][0] *= 2.0
        results['avgI'][-1] *= 2.0
        results['stdI'][0] *= 2.0
        results['stdI'][-1] *= 2.0
        results['f'][0,:] *= 2.0
        results['f'][-1,:] *= 2.0
        # Average infections as function of time
        norm = args[2]/(results['dT']*k)
        results['avgI'] *= norm
        # Standard deviations of infections
        results['stdI'] *= norm
        results['stdI'] = np.sqrt(results['stdI'] - results['avgI']**2)
        results['errI'] = results['stdI']*np.sqrt(args[2]/results['dT'])
        # Recoveries: integral off recovery rate calculated from average active infections
        avg = RecoveryRate(results['t'], results['avgI'], **args[3])
        err = RecoveryRate(results['t'], results['errI'], **args[3])
        for n in range(1, results['Nt']):
            results['avgR'][n] = results['avgR'][n-1] + avg[n]*results['dT']
            results['errR'][n] = np.sum(err[:n+1]**2)
        results['errR'] = np.sqrt(results['errR'])*results['dT']
        # Deaths: integral off death rate calculated from average active infections
        avg = DeathRate(results['t'], results['avgI'], **args[4])
        err = DeathRate(results['t'], results['errI'], **args[4])
        for n in range(1, results['Nt']):
            results['avgD'][n] = results['avgD'][n-1] + avg[n]*results['dT']
            results['errD'][n] = np.sum(err[:n+1]**2)
        results['errD'] = np.sqrt(results['errD'])*results['dT']
        # Total number of infections: average infections plus recoveries and deaths
        results['avgIt'] = results['avgI'] + results['avgR']
        results['errIt'] = np.sqrt(results['errI']**2 + results['errR']**2 + results['errD']**2)
        # Smooth histogram with a 5 point average over infections
        f = np.zeros_like(results['f'])
        for n in range(results['Nt']):
            f[n,0] = (results['f'][n,0] + results['f'][n,1] + results['f'][n,2])/3.0
            f[n,1] = (results['f'][n,0] + results['f'][n,1] + results['f'][n,2] + results['f'][n,3])/4.0
            for i in range(2, results['Ni']-2):
                f[n,i] = (results['f'][n,i-2] + results['f'][n,i-1] + results['f'][n,i] + results['f'][n,i+1] + results['f'][n,i+2])/5.0
            f[n,-2] = (results['f'][n,-1] + results['f'][n,-2] + results['f'][n,-3] + results['f'][n,-4])/4.0
            f[n,-1] = (results['f'][n,-1] + results['f'][n,-2] + results['f'][n,-3])/3.0
        results['f'] = f
        # Smooth histogram with a 3 point average over time
        for i in range(0, results['Ni']):
            f[0,i] = (results['f'][0,i] + 0.5*results['f'][1,i])/1.5
            for n in range(1, results['Nt']-1):
                f[n,i] = 0.5*(0.5*results['f'][n-1,i] + results['f'][n,i] + 0.5*results['f'][n+1,i])
            f[-1,i] = (results['f'][-1,i] + 0.5*results['f'][-2,i])/1.5
        results['f'] = f
        # Calculate most probable infection bin
        for n in range(results['Nt']):
            maks = np.max(results['f'][n,:])
            for i in range(results['Ni']):
                if (maks == results['f'][n,i]):
                    results['probI'][n] = results['I'][i]
        # Smooth most probable infection (3 day running average)
        prop = np.zeros_like(results['probI'])
        uncr = np.zeros_like(prop)
        prop[0] = np.average(results['probI'][:4])
        uncr[0] = np.std(results['probI'][:4])
        prop[1] = np.average(results['probI'][:5])
        uncr[1] = np.std(results['probI'][:5])
        prop[2] = np.average(results['probI'][:6])
        uncr[2] = np.std(results['probI'][:6])
        for i in range(3, len(prop)-3):
            prop[i] = np.average(results['probI'][i-3:i+4])
            uncr[i] = np.std(results['probI'][i-3:i+4])
        prop[-3] = np.average(results['probI'][-6:])
        uncr[-3] = np.std(results['probI'][-6:])
        prop[-2] = np.average(results['probI'][-5:])
        uncr[-2] = np.std(results['probI'][-5:])
        prop[-1] = np.average(results['probI'][-4:])
        uncr[-1] = np.std(results['probI'][-4:])
        results['probI'] = prop
        results['errProbI'] = uncr
        # Normalise histogram
        results['f'] *= norm
        # for i in range(results['Ni']):
            # results['f'][:,i] /= np.log10(results['Ibins'][i+1]) - np.log10(results['Ibins'][i])
        # Estimate factor of unknown infections
        results['probRatio'] = results['probI'][1:-1]/results['avgI'][1:-1]
        results['uncrRatio'] = (results['avgI'][1:-1] + results['errI'][1:-1])/results['avgI'][1:-1]
        results['stdvRatio'] = (results['avgI'][1:-1] + results['stdI'][1:-1])/results['avgI'][1:-1]
    return results


def EulerMaruyama(Np, t0, x0, coefArgs, boundArgs, binArgs):
    """Function to solve an SDE with the Euler-Maruyama scheme.
    
    EulerMaruyama(Np, t0, x0, coefArgs, boundArgs, binArgs)
        Solve an SDE with the Euler-Maruyama scheme. If a(x,t) and b(x,t) is the
        drift and diffusion coefficient, respectively, dependent on time t and
        the variable x, then the SDE is solved iteratively,
        X(t + dt) = X(t) + a(X(t),t)*dt + b(X(t),t)*R(t)*dt^0.5 ,
        from the initial state X(t0) = x0, where R(t) is a random number choosen
        at each step such that it is Normal/Gaussian distribution with zero mean
        and unit variance.
        NOTE : Three functions must be supplied: Coefficients(t, x, args),
               Boundary(t, x, args), and Bin(result, k, t, x, status, args)
               to calculate drfit and diffusion coefficients, apply boundary
               conditions, and to bin results, respectively. The arguments of
               the functions are (t: the current time, x: the current state,
               result: a dictionary for the saved results, k: the solution
               number, status: the current status of the calculations, and args:
               additional arguments needed by the function). The Coefficients
               function must return a tuple with the drift and diffusion
               coefficients and the time step. The Boundary function must return
               a tuple with the status (0 if the calculations must be continued
               and 1 if the calculation must be stopped) and the updated
               (according to the boundary conditions if necessary) state. The
               Bin function must return the updated results dictionary. The Bin
               function will be called a last time with k=Np and status=2 at the
               end of all calculations to finalize the results.
    
    Arguments:
        Np        : Number of times the SDE must be solved.
        t0        : Initial time.
        x0        : Initial state at time t0.
        coefArgs  : Arguments needed by the Coefficients function.
        boundArgs : Arguments needed by the Boundary function.
        binArgs   : Arguments needed by the Bin function.
    
    Returns:
        A dictionary with the results saved and calculated by the Bin function.
    """
    rndm.seed(50001204) # Initialize pseudo-random number generator
    result = {}         # Dictionary for results
    # For each SDE solution
    for k in range(Np):
        t = t0[k]
        X = x0[k]
        # Initial step
        a, b, dt = Coefficients(t, X, coefArgs)        # Calculate coefficients
        X += a*dt + b*np.sqrt(dt)*rndm.gauss(0.0, 1.0) # Evolve solution
        t += dt
        status, X = Boundary(t, X, boundArgs)          # Apply boundary conditions
        result = Bin(result, k, t, X, status, binArgs) # Bin results
        # Continue to evolve solution
        while (status == 0):
            a, b, dt = Coefficients(t, X, coefArgs)
            X += a*dt + b*np.sqrt(dt)*rndm.gauss(0.0, 1.0)
            t += dt
            status, X = Boundary(t, X, boundArgs)
            result = Bin(result, k, t, X, status, binArgs)
    # Final call to finalize binning
    print("\tFinalizing Results ...")
    return Bin(result, Np, t0, x0, 2, binArgs)


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
    parser.add_argument('Np', type=int, help='Number of SDE solutions.')
    parser.add_argument('N', type=float, help='Population size.')
    parser.add_argument('T', type=float, help='Duration of solution (in days).')
    parser.add_argument('dt', type=float, help='Time step (in days).')
    parser.add_argument('parameters', type=str, help='File name with parameters.')
    args = parser.parse_args()
    
    ############################################################################
    # Do the calculations                                                      #
    ############################################################################
    t0 = 10.0*np.random.rand(args.Np) # Injection of new cases
    x0 = t0*4.5 + 1.0
    with open(args.parameters, 'rb') as fh:
        beta, rho, delta = pickle.load(fh)
    coefArgs = [args.N, beta, rho, delta, args.dt]
    boundArgs = [args.N, args.T]
    binArgs = [args.N, args.T, args.dt, rho, delta]
    print("Doing calculations ...")
    result = EulerMaruyama(args.Np, t0, x0, coefArgs, boundArgs, binArgs)
    # Add for future calculations
    result['N'] = args.N
    result['T'] = args.T
    
    ############################################################################
    # Save results to file                                                     #
    ############################################################################
    print("Saving results ...")
    with open('SISzaFit.pckl', 'wb') as fh:
        pickle.dump(result, fh)
