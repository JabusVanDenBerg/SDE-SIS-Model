# SDE-SIS-Model

Solve a SIS model for modelling an infection outbreak with stochastic differential equations.

## The SIS Model

The Model assumes that people susceptible to be infected (S) are infected (I) at a rate of beta (% of people per day) and that infected people who recover at a rate rho (recoveries per day), will be again susceptible to be infected (people do not gain immunity), hence (S --> I --> S). Let N be the total population which can be infected and let Ns and Ni be the number of people susceptible and infected, respectively, then the flow from S to I is beta * Ni * Ns = beta * Ni * (N - Ni), while the flow from I back to S is rho * Ni.

## The SDE Model

The SIS model can be solved using stochastic differential equations (SDE). The SDE governing the number of infected (X), is
dX(t) = a(X) * dt + b(X) * dW
where a(x) = beta * x * (N - x) - rho * x and b(x) = sqrt(beta * x * (N - x) + rho * x).

## Examples

