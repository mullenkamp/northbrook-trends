# northbrook-trends

This repo contains code to run a trend anaylsis on groundwater levels, flows, and precipitation within the Ashley catchment with an emphasis on the Rangiora area.

## Background
The trend anaylsis uses the [Prophet](https://facebook.github.io/prophet/) package designed for decomposing time series data into trends and seasonalities to use for forecasting. For this use case, the forecasting aspect was not necessary.

The output plots are grouped into seven sets of sites. The first three are groundwater level sites, four through six are flow sites, and seven are precipitation sites. The accompaning shapefiles provide the precise locations of all sites.

## Running of the script
The code is entirely wrapped up in the trends.py script. To run the script, install a Python environment using the conda yml environment file main.yml:

```
conda env create -f main.yml
```

The parameters for the script is near the top of the script. The main parameter that would likely to be changed is the mcmc_samples. This parameter defines the number of monte carlo samles to be run for the uncertainty analysis. I've set the default to 0 which doesn't do the uncertainty analysis and is very fast to run. I have run the uncertainty anaylsis with a value of 300 (as per the example defaults in the docs). This takes approximately 1 hour to run fully. I have also used the default uncertainty intervals of 80%. The other details on the implementation can be read in the code.

I have run the decomposion using weekly and monthly aggregate intervals of the data, and all permutations are run through the plotting functions. This doesn't mean that all plots are equally useful. For example, plots using monthly data should be solely used for the precipitation decomposition as monthly sums of precipitation ensure that most months do not have zero precipitation. Any aggregate finer than monthly for precipitation would likely create a lot of zeros which statistical models don't like (floors and ceilings of data).
