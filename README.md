# SDE-SIS-Model

Solve a SIS model for modelling an infection outbreak with stochastic differential equations. A detailed discription of the model can be found in the blog posts:


## The SIS Model

The model assumes that people susceptible to be infected (S) are infected (I) at a rate of beta (% of people per day) and that infected people who recover at a rate rho (recoveries per day), will be again susceptible to be infected (people do not gain immunity), hence (S --> I --> S). Let N be the total population which can be infected and let Ns and Ni be the number of people susceptible and infected, respectively, then the flow from S to I is beta * Ni * Ns = beta * Ni * (N - Ni), while the flow from I back to S is rho * Ni.

## The SDE Model

The SIS model can be solved using stochastic differential equations (SDE). The SDE governing the number of infected (X), is dX(t) = a(X,t) * dt + b(X,t) * dW(t), where a(x,t) = beta(t) * x * (N - x) - rho(t) * x, b(x,t) = sqrt(beta(t) * x * (N - x) + rho(t) * x), and dW(t) is a Wiener process. The SDE is solved iteratively, X(t + dt) = X(t) + a(X(t),t) * dt + b(X(t),t) * R(t) * sqrt(dt), from the initial state X(t0) = x0, where dt is the time step and R(t) is a random number choosen at each step such that it is Normal/Gaussian distribution with zero mean and unit variance.

## Examples

### Basic

The Basic example show case how the basic model work with constant parameters. The `SIS.py` script takes six command line arguments, namely, `Np` the number of SDE solutions, `N` the population size, `T` the duration of the solution (in days), `dt` the time step (in days), `beta` the infection rate (percentage per day), and `rho` the recovery rate (recoveries per day), for example
```
python SIS.py 10000 5.9e7 100.0 0.25 4.6e-8 2.4
```
The `SIS.py` script will then create a `SIS.pckl` file with the results, which can be plotted with the `PlotSIS.py` script using
```
python PlotSIS.py SIS.pckl
```

## Disclaimer

The code is presented "as is" and not guaranteed to be without errors or to be accurate. The code and ideas developed therein may be used, distributed, or changed for educational purposes and personal use, but should not be used for commercial purposes or scientific claims.
