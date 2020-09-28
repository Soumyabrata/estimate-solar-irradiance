## Estimating Solar Irradiance Using Sky Imagers

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript: 

> S. Dev, F. M. Savoy, Y. H. Lee, S. Winkler, Estimating Solar Irradiance Using Sky Imagers, Atmospheric Measurement Techniques (AMT), 2019


All codes are written in `python` and `MATLAB`.

### Code
+ `Figure5.py`: Plots the measured solar irradiance along with clear-sky solar irradiance model. It also plots the percentage deviation of solar irradiance from clear sky data
+ `Figure11.m`: Plots the impact of training images on the RMSE values.
+ `other-models.py`: Computes the various benchmarking solar irradiance estimation models
+ `proposed-model.py`: Computes the proposed solar irradiance estimation model


The above code files use the following user-defined helper scripts.

#### Scripts
+ `generating-modelfiles.py`: Generates the model files for the various solar estimation models
+ `impact-training.py`: Generates the data files required for plotting the impact of training images on RMSE values
+ `import_WS.py`: Imports the weather station data
+ `normalize_array.py`: Normalizes any given input array
+ `remove_outliers.py`: Removes outliers from a given input array
+ `import_weather.py`: Imports the given weather data
+ `SG_solarmodel.py`: Implements the Singapore clear sky solar irradiance model
