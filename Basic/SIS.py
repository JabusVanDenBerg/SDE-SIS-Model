"""SIS model for modelling an infection outbreak.

Model assumes that people susceptible to be infected (S) are infected (I) at a
rate of beta (% of people per day) and that infected people who recover at a
rate rho (recoveries per day), will be again susceptible to be infected, hence
(S --> I --> S). Let N be the total population which can be infected and let Ns
and Ni be the number of people susceptible and infected, respectively, then the
flow from S to I is beta*Ni*Ns = beta*Ni*(N - Ni), while the flow from I back to
S is rho*Ni.
    
The stochastic differential equation (SDE) governing the number of infected (X),
is dX(t) = a(X)*dt + b(X)*dW, where a(x) = beta*x*(N - x) - rho*x and b(x) =
sqrt(beta*x*(N - x) + rho*x).
"""


import numpy as np     # Numerical Python package
import random as rndm  # Random number generator package


def Coefficients(t, x, args):
    """Function to calculate the drift and diffusion coefficients.
    
    Coefficients(t, x, args)
        Calculate the drift and diffusion coefficients for the SDE.
    
    Arguments:
        t    : The current time.
        x    : The number of infected.
        args : Additional arguments needed by the function. The order of the
               arguments are:
               N    : The total population.
               beta : The infection rate.
               rho  : The recovery rate.
               dt   : The simulation time step.
    
    Returns:
        A tuple with the drift and diffusion coefficients and the time step.
    """
    a = args[1]*x*(args[0] - x) - args[2]*x
    b = np.sqrt(args[1]*x*(args[0] - x) + args[2]*x)
    return (a, b, args[3])


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
                  N  : The total population.
                  T  : The maximum simulation time.
                  dt : The time step.
    
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
            mid    : An array with the mid points of the bins.
            bins   : An array with the bounds of the bins. It is assumed that
                     the first value is the lower bound of the first bin.
            width  : The bin width. If logarithmic bins are used, then this must
                     be the constant logarithmic bin width.
            N      : The number of bins.
        
        Returns:
            The index of the bin or -1 if the value does not fall into a bin.
        """
        if (linlog == 'lin'):  # Initial guess
            indx = int(abs(value - mid[0])/width)
        elif (linlog == 'log'):
            if (value == 0.0):
                indx = int(np.log10(abs(value - mid[0]))/width)
            else:
                indx = int(abs(np.log10(value) - np.log10(mid[0]))/width)
        while (True):  # Find bin
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
        results['dT'] = 1.0                                      # Time bin width
                                                                 # Time bin mid points
        results['t'] = np.arange(0.0, args[1]+results['dT'], results['dT'])
        results['Nt'] = len(results['t'])                        # Number of time bins
        results['tbins'] = np.zeros(results['Nt'] + 1)           # Time bins
        results['tbins'][:-1] = results['t'] - 0.5*results['dT']
        results['tbins'][-1] = results['t'][-1] + 0.5*results['dT']
        results['avgI'] = np.zeros(results['Nt'])                # Average active infections vs time
        results['stdI'] = np.zeros(results['Nt'])                # Standard deviation of active infections
        results['probI'] = np.zeros(results['Nt'])               # Most probable active infections vs time
        results['pIu'] = np.zeros(results['Nt'])
        results['pIl'] = np.zeros(results['Nt'])
                                                                 # Infection bins
        results['Ibins'] = np.logspace(0.0, np.log10(args[0]), num=400)
                                                                 # Logarithm of infection bin width
        results['dI'] = np.log10(results['Ibins'][1]) - np.log10(results['Ibins'][0])
        results['Ni'] = len(results['Ibins']) - 1                # Number of infection bins
        results['I'] = np.zeros(results['Ni'])                   # Infection bin mid points
        for i in range(results['Ni']):
            results['I'][i] = 0.5*(np.log10(results['Ibins'][i+1]) + np.log10(results['Ibins'][i]))
        results['I'] = 10**results['I']
        results['f'] = np.zeros((results['Nt'], results['Ni']))  # Distribution
    # Bin result
    if (status == 0):
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
        # Calculate most probable infection bin
        for n in range(results['Nt']):
            maks = np.max(results['f'][n,:])
            for i in range(results['Ni']):
                if (maks == results['f'][n,i]):
                    results['probI'][n] = results['I'][i]
                    results['pIu'][n] = results['Ibins'][i+1]
                    results['pIl'][n] = results['Ibins'][i]
        # Normalise histogram
        results['f'] *= norm
        #for i in range(results['Ni']):
            #results['f'][:,i] /= np.log10(results['Ibins'][i+1]) - np.log10(results['Ibins'][i])
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
               the functions are:
               t      : The current time.
               x      : The current state.
               result : A dictionary for the saved results.
               k      : The solution number.
               status : The current status of the calculations.
               args   : Additional arguments needed by the function.
               The Coefficients function must return a tuple with the drift and
               diffusion coefficients and the time step. The Boundary function
               must return a tuple with the status (0 if the calculations must
               be continued and 1 if the calculation must be stopped) and the
               updated (according to the boundary conditions if necessary) state.
               The Bin function must return the updated results dictionary. The
               Bin function will be called a last time with k=Np and status=2 at
               the end of all calculations to finalize the results.
    
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
    rndm.seed(50001204)  # Initialize pseudo-random number generator
    result = {}          # Dictionary for results
    # For each SDE solution
    for k in range(Np):
        t = t0
        X = x0
        # Initial step
        a, b, dt = Coefficients(t, X, coefArgs)         # Calculate coefficients
        X += a*dt + b*np.sqrt(dt)*rndm.gauss(0.0, 1.0)  # Evolve solution
        t += dt
        status, X = Boundary(t, X, boundArgs)           # Apply boundary conditions
        result = Bin(result, k, t, X, status, binArgs)  # Bin results
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
    parser.add_argument('beta', type=float, help='Infection rate (percentage per day).')
    parser.add_argument('rho', type=float, help='Recovery rate (recoveries per day).')
    args = parser.parse_args()
    
    ############################################################################
    # Do the calculations                                                      #
    ############################################################################
    t0 = 0.0
    x0 = 1.0
    coefArgs = np.array([args.N, args.beta, args.rho, args.dt])
    boundArgs = np.array([args.N, args.T])
    binArgs = np.array([args.N, args.T, args.dt])
    print("Doing calculations ...")
    result = EulerMaruyama(args.Np, t0, x0, coefArgs, boundArgs, binArgs)
    # Add parameters for future calculations
    result['N'] = args.N
    result['T'] = args.T
    result['beta'] = args.beta
    result['rho'] = args.rho
    
    ############################################################################
    # Save results to file                                                     #
    ############################################################################
    print("Saving results ...")
    with open('SIS.pckl', 'wb') as fh:
        pickle.dump(result, fh)
