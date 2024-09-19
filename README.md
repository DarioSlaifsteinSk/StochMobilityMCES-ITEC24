# Stochastic Mobility Integration into Residential Energy Hubs
## ESARS-ITEC24 26-29 November 2024

## Suggestions for a good README


Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Description

Code for reproducing the case studies of "Stochastic Mobility Integration into Residential Energy Hubs" presented in the ESARS-ITEC 2024 Conference held in Naples on 26-29th of November.

The code implements a stochastic optimization model for the integration of electric vehicles and residential energy hubs. The model is based on a single-stage stochastic programming formulation that considers the uncertainty in the mobility patterns of the electric vehicles. The problem is modelled using Random Field Optimization. The code is implemented in `Julia`, `JuMP` and `InfiniteOpt` and uses the `Ipopt` solver.

The pipeline of paper is as follows:

![alt text](images/graph_abstract.png)

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

```julia
julia> cd("[INSERT_PATH_TO_FILES]/StochMobilityMCES-ITEC24/")

julia> ]

(@v1.10) pkg> activate .

(itec24) pkg> instantiate

julia> include("dynamic_power.jl")
```

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
For support on code usage please submit an issue on the repository.

## Authors and acknowledgment
This paper was done by Dario Slaifstein, Alvaro Menendez Agudin, Gautham Ram Chandra Mouli, Laura Ramirez Elizondo, and Pavol Bauer.

## License
For open source projects, say how it is licensed.