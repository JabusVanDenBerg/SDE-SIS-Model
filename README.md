# SDE-SIS-Model

Solve an SIS model for modelling an infection outbreak with stochastic differential equations. A detailed discription of the model can be found in the blog posts:
- [Modelling an Infection Outbreak like a Physicist. Part 1: The SIS Model](http://ufarasuntalk.home.blog/2020/05/10/modelling-an-infection-outbreak-like-a-physicist-part-1-the-sis-model/)
- [Modelling an Infection Outbreak like a Physicist. Part 2: Calculating Infection, Recovery, and Death Rates from Data](ufarasuntalk.home.blog/2020/05/18/modelling-an-infection-outbreak-like-a-physicist-part-2-calculating-infection-recovery-and-death-rates-from-data/)


## The SIS Model

The model assumes that people susceptible to be infected (S) are infected (I) at a rate of beta (% per day) and that infected people who recover at a rate rho (recoveries per day), will be again susceptible to be infected (people do not gain immunity), hence (S --> I --> S). Let N be the total population which can be infected and let Ns and Ni be the number of people susceptible and infected, respectively, then the flow from S to I is beta * Ni * Ns = beta * Ni * (N - Ni), while the flow from I back to S is rho * Ni. Additionally, it can also be assumed that infected people can die (D) at a rate delta (deaths per day), hence (S --> I --> S or S --> I --> D). The flow from I to D is then delta * Ni. Notice that the deaths should decrease N, but this is not taken into account and hence, the model is only valid if the total number of deaths are much less than N.

## The SDE Model

The SIS model can be solved using stochastic differential equations (SDE). The SDE governing the number of infected (X), is dX(t) = a(X,t) * dt + b(X,t) * dW(t), where a(x,t) = beta(t) * x * (N - x) - [rho(t) + delta(t)] * x, b(x,t) = sqrt(beta(t) * x * (N - x) + rho(t) * x), and dW(t) is a Wiener process. The SDE is solved iteratively, X(t + dt) = X(t) + a(X(t),t) * dt + b(X(t),t) * R(t) * sqrt(dt), from the initial state X(t0) = x0, where dt is the time step and R(t) is a random number choosen at each step such that it is Normal/Gaussian distribution with zero mean and unit variance.

## Examples

### Basic

The Basic example show how the basic model work with constant parameters. The `SIS.py` script takes six command line arguments, namely, `Np` the number of SDE solutions, `N` the population size, `T` the duration of the solution (in days), `dt` the time step (in days), `beta` the infection rate (percentage per day), and `rho` the recovery rate (recoveries per day), for example
```
python SIS.py 10000 5.9e7 100.0 0.25 4.6e-8 2.4
```
The `SIS.py` script will then create a `SIS.pckl` file with the results, which can be plotted with the `PlotSIS.py` script using
```
python PlotSIS.py SIS.pckl
```
See the code documentation for more information.

### South Africa / Data

The Data example show how the infection, recovery, and death rate can be extracted from data, specifically for South Africa. The `Parameters.py` script takes no command line arguments, but assumes there is a file `Data.txt` with the following columns:
- Date:       The date (not used).
- Day:        The number of the day relative to the first reported infection, which is day 0.
- Cases:      The total number of infections.
- Change:     The number of new infections per day.
- Tests:      The total number of tests preformed (not used).
- Change:     The number of new tests per day.
- Recovered:  The number of total recoveries.
- Change:     The number of recoveries per day.
- Deaths:     The total number of deaths.
- Change:     The number of deaths per day.
- Notes:      Notes of interest on that day (not used). This should begin with a `#` to indicate it as a comment.

The script will then create the graphs which can be found in this example with the calculated rates and other statistics.

### South Africa / Fit Data

The Fit Data example show how the infection, recovery, and death rate extracted from data, specifically the first 100 days for South Africa, can be used to model the data with the SDEs. The `Parameters.py` script is similar to the one described in the Data example, but also saves a `FitParameters.pckl` file with the temporal infection, recovery, and death rate to be used by the `SISzaFit.py` script. The `SISzaFit.py` script takes five command line arguments, namely, `Np` the number of SDE solutions, `N` the population size, `T` the duration of the solution (in days), `dt` the time step (in days), and `parameters` the file with the coefficients calculated by `Parameters.py`, for example
```
python SISzaFit.py 10000 5.9e7 100.0 FitParameters.pckl
```
The `SIS.py` script will then create a `SISzaFit.pckl` file with the results, which can be plotted with the `PlotSISzaFit.py` script using
```
python PlotSISzaFit.py SISzaFit.pckl
```
An additional feature of the `SISzaFit.py` script, compared to the basic `SIS.py` script, is that it starts each SDE solution with a different number of infections according to the initial time to mimic new cases entering the country. See the code documentation for more information.

## Disclaimer

The code is presented "as is" and is not guaranteed to be without errors or to be accurate. The code and ideas developed therein may be used, distributed, or changed for educational purposes and personal use, but should not be used for commercial purposes or scientific claims.
