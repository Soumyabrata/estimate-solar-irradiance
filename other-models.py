# Import the libraries

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import exifread
import operator
import bisect
import datetime
import pysolar
import math
import scipy.stats
from pysolar.solar import *
from tempfile import TemporaryFile
from matplotlib.dates import DateFormatter
from scipy import *
from pylab import *
from datetime import datetime, timedelta
from matplotlib import rcParams


# User defined functions
from import_weather import *
from SG_solarmodel import *


# import weather station data
CSV_file ='radiosonde_Sep2016.csv'
(time_range, solar_range, temperature_range, humidity_range, dewpoint_range, windspeed_range, winddirection, pressure, rainfallrate) = import_weather(CSV_file)
temperature_range = np.array(temperature_range)
solar_range = np.array(solar_range)
rainfallrate = np.array(rainfallrate)


# Extracting unique dates from the weather station data
date_list = []
for onetime in time_range:
    date_ext = onetime.strftime('%Y-%m-%d')
    date_list.append(date_ext)
unik_dates = list(set(date_list))
unik_dates = sorted(unik_dates)
print('Unique dates are ', unik_dates)



# ------------------------------
# sample weather explanation for a sample date
samplestring = ['2016-09-01']

x_vector = []
for no_of_files, mydatestring in enumerate(samplestring):


    # Performing for a single date
    mydateindex = []
    for kot, alldate in enumerate(date_list):
        if mydatestring in alldate:
            mydateindex.append(kot)

    # mydateindex is the list of index for a single date.
    mydateindex = np.array(mydateindex)

    mytemparray = temperature_range[mydateindex]
    mysolararray = solar_range[mydateindex]
    myrainarray = rainfallrate[mydateindex]
    mytimearray = [time_range[i] for i in mydateindex]

    Tmax = np.max(mytemparray)
    Tmin = np.min(mytemparray)

    delT = Tmax - Tmin

    Tavg = (Tmax + Tmin) / 2

    ftavg = 0.017 * (np.power(np.exp(1), np.power(np.exp(1), -(0.053 * Tavg))))

    total_rainfall = 0
    for rain_event in myrainarray:
        if (rain_event != 0):
            total_rainfall = total_rainfall + rain_event / 60
    PP = total_rainfall

    # Calculate the clear sky radiation
    latitude = 1.3429943
    longitude = 103.6810899
    clear_sky_rad = []
    HS_model = []
    BC_model = []
    Hunt_model = []
    DC_model = []

    # Accessing each datetime objects in a single day
    for timeinstant in mytimearray:

        # # When I am executing it from United States
        # timeinstant = timeinstant + timedelta(hours=12)

        # # When I am executing it from Dublin
        # timeinstant = timeinstant + timedelta(hours=7)

        timeinstant = datetime.datetime(timeinstant.year, timeinstant.month, timeinstant.day, timeinstant.hour, timeinstant.minute, timeinstant.second, 0, pytz.UTC)
        CSR = SG_model(timeinstant)
        clear_sky_rad.append(CSR)

        # HS model
        SV = CSR * 0.17 * (math.sqrt(Tmax - Tmin))
        HS_model.append(SV)

        # BC model
        BC_item = CSR * 0.7025 * (1 - (np.power(np.exp(1), -(0.0101 * (np.power(delT, 1.9034))))))
        BC_model.append(BC_item)

        # Hunt model
        Hunt_item = 0.1349 + (0.1596 * (CSR * (np.sqrt(Tmax - Tmin)))) - 0.0621 * Tmax - 0.8142 * PP + 0.0264 * PP * PP
        Hunt_model.append(Hunt_item)

        # DC model
        DC_item = CSR * 0.7107 * (
                    1 - (np.power(np.exp(1), -(0.2481 * ftavg * delT * delT * np.power(np.exp(1), Tmin / 67.4299)))))
        DC_model.append(DC_item)

    HS_model = np.array(HS_model)  # Hargreaves and Samani model
    BC_model = np.array(BC_model)  # Bristow and Campbell model
    Hunt_model = np.array(Hunt_model)  # Hunt model
    DC_model = np.array(DC_model)  # Donatelli and Campbell model

    clear_sky_rad = np.array(clear_sky_rad)  # Clear sky model

    # Extract time between 7 am and 7 pm
    YY = int(mydatestring[0:4])
    MM = int(mydatestring[5:7])
    DD = int(mydatestring[8:10])

    start_time = datetime.datetime(YY, MM, DD, 7, 0, 0)
    end_time = datetime.datetime(YY, MM, DD, 19, 0, 0)
    selecttime_index = []
    for kot2, time2 in enumerate(mytimearray):
        if time2 > start_time and time2 < end_time:
            selecttime_index.append(kot2)

    selecttime_index = np.array(selecttime_index)

    if (len(selecttime_index) != 0):

        mytimearray_ST = [mytimearray[i] for i in selecttime_index]
        mysolararray_ST = mysolararray[selecttime_index]
        HS_model_ST = HS_model[selecttime_index]
        BC_model_ST = BC_model[selecttime_index]
        Hunt_model_ST = Hunt_model[selecttime_index]
        DC_model_ST = DC_model[selecttime_index]

        clearsky_model_ST = clear_sky_rad[selecttime_index]

        fig = plt.figure(1 + no_of_files, figsize=(15, 3))
        plt.plot(mytimearray_ST, mysolararray_ST, 'r', label='Weather Station')
        plt.plot(mytimearray_ST, HS_model_ST, 'b', label='Hargreaves and Samani')
        plt.plot(mytimearray_ST, BC_model_ST, 'm', label='Bristow and Campbell')
        plt.plot(mytimearray_ST, Hunt_model_ST, 'c', label='Hunt')
        plt.plot(mytimearray_ST, DC_model_ST, 'g', label='Donatelli and Campbell')
        plt.plot(mytimearray_ST, clearsky_model_ST, 'k', label='Clear sky model', linestyle='-.', linewidth=2)
        plt.legend(loc='upper right')
        plt.title(mydatestring, fontsize=12)
        # plt.xlabel('Image luminance', fontsize=12)
        plt.ylabel(r'Solar irradiance [W/m$^2$]', fontsize=14)

        fig.autofmt_xdate()
        plt.grid(b=None, which='major', axis='both')
        formatter = DateFormatter('%H:%M')
        plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
        plt.grid(True)
        plt.savefig('./results/diffmodelsdash.pdf', format='pdf')
        plt.show()




# =====================================================================================================

# scatter plot and RMSE values
model_loc = './JSTARfiles/ModelFiles/ReportedInPaper/HS/*.txt'
model_files = glob.glob(model_loc)
model_files = sorted(model_files)

solar_cat = []
predictedsolar_cat = []

for no_of_files, file_name3 in enumerate(model_files):

    # read the file
    with open(file_name3) as f:  # f is a file header
        reader = csv.reader(f, delimiter=",")
        d = list(reader)  # d is a list of list here.
        total_rows = len(d)

        # variable i starts from 1 so that header is skipped
        for i in range(1, total_rows):
            item1 = d[i][1]
            item2 = d[i][2]

            solar_cat.append(float(item1))
            predictedsolar_cat.append(float(item2))

len_array = len(solar_cat)

# Converting to numpys
solar_cat = np.array(solar_cat)
predictedsolar_cat = np.array(predictedsolar_cat)
RMSE_L = np.sqrt(np.mean((predictedsolar_cat-solar_cat)**2))
print ('Root mean square for Hargreaves and Samani model = ', RMSE_L , 'Watt/m2')


arr1 = predictedsolar_cat.reshape(-1, 1)
arr2 = solar_cat.reshape(-1, 1)
arr3 = np.hstack((arr1, arr2))
rho, pval = scipy.stats.spearmanr(arr3)
print ('correlation for Hargreaves and Samani model = ', rho)


rcParams.update({'figure.autolayout': True})
plt.figure(2, figsize=(5, 4))
plt.scatter(solar_cat, predictedsolar_cat, marker='o' , linewidths=0.005 , alpha = 0.01)
plt.xlabel(r'Measured solar irradiance [W/m$^2$]', fontsize=14)
plt.ylabel(r'Estimated solar irradiance [W/m$^2$]', fontsize=14)
plt.grid(True)
plt.xlim((-200,1400))
plt.ylim((-100,800))
plt.savefig('./results/scatter-HS.png', format='png')
plt.show()




# =====================================================================================================
# Bristow and Campbell
# scatter plot and RMSE values
model_loc = './JSTARfiles/ModelFiles/ReportedInPaper/BC/*.txt'
model_files = glob.glob(model_loc)
model_files = sorted(model_files)

solar_cat = []
predictedsolar_cat = []

for no_of_files, file_name3 in enumerate(model_files):

    # read the file
    with open(file_name3) as f:  # f is a file header
        reader = csv.reader(f, delimiter=",")
        d = list(reader)  # d is a list of list here.
        total_rows = len(d)

        # variable i starts from 1 so that header is skipped
        for i in range(1, total_rows):
            item1 = d[i][1]
            item2 = d[i][2]

            solar_cat.append(float(item1))
            predictedsolar_cat.append(float(item2))

len_array = len(solar_cat)

# Converting to numpys
solar_cat = np.array(solar_cat)
predictedsolar_cat = np.array(predictedsolar_cat)
RMSE_L = np.sqrt(np.mean((predictedsolar_cat-solar_cat)**2))
print ('Root mean square for Bristow and Campbell model = ', RMSE_L , 'Watt/m2')


arr1 = predictedsolar_cat.reshape(-1, 1)
arr2 = solar_cat.reshape(-1, 1)
arr3 = np.hstack((arr1, arr2))
rho, pval = scipy.stats.spearmanr(arr3)
print ('correlation for Bristow and Campbell model = ', rho)


rcParams.update({'figure.autolayout': True})
plt.figure(3, figsize=(5, 4))
plt.scatter(solar_cat, predictedsolar_cat, marker='o' , linewidths=0.005 , alpha = 0.01)
plt.xlabel(r'Measured solar irradiance [W/m$^2$]', fontsize=14)
plt.ylabel(r'Estimated solar irradiance [W/m$^2$]', fontsize=14)
plt.grid(True)
plt.xlim((-200, 1400))
plt.ylim((-100, 800))
plt.savefig('./results/scatter-BC.png', format='png')
plt.show()








# =====================================================================================================
# Hunt
# scatter plot and RMSE values
model_loc = './JSTARfiles/ModelFiles/ReportedInPaper/HuntModel/*.txt'
model_files = glob.glob(model_loc)
model_files = sorted(model_files)

solar_cat = []
predictedsolar_cat = []

for no_of_files, file_name3 in enumerate(model_files):

    # read the file
    with open(file_name3) as f:  # f is a file header
        reader = csv.reader(f, delimiter=",")
        d = list(reader)  # d is a list of list here.
        total_rows = len(d)

        # variable i starts from 1 so that header is skipped
        for i in range(1, total_rows):
            item1 = d[i][1]
            item2 = d[i][2]

            solar_cat.append(float(item1))
            predictedsolar_cat.append(float(item2))

len_array = len(solar_cat)

# Converting to numpys
solar_cat = np.array(solar_cat)
predictedsolar_cat = np.array(predictedsolar_cat)
RMSE_L = np.sqrt(np.mean((predictedsolar_cat-solar_cat)**2))
print ('Root mean square for Hunt model = ', RMSE_L , 'Watt/m2')


arr1 = predictedsolar_cat.reshape(-1, 1)
arr2 = solar_cat.reshape(-1, 1)
arr3 = np.hstack((arr1, arr2))
rho, pval = scipy.stats.spearmanr(arr3)
print ('correlation for Hunt model = ', rho)


rcParams.update({'figure.autolayout': True})
plt.figure(4, figsize=(5, 4))
plt.scatter(solar_cat, predictedsolar_cat, marker='o' , linewidths=0.005 , alpha = 0.01)
plt.xlabel(r'Measured solar irradiance [W/m$^2$]', fontsize=14)
plt.ylabel(r'Estimated solar irradiance [W/m$^2$]', fontsize=14)
plt.grid(True)
plt.xlim((-200, 1400))
plt.ylim((-100, 800))
plt.savefig('./results/scatter-Hunt.png', format='png')
plt.show()








# =====================================================================================================
# Donatelli and Campbell
# scatter plot and RMSE values
model_loc = './JSTARfiles/ModelFiles/ReportedInPaper/DC/*.txt'
model_files = glob.glob(model_loc)
model_files = sorted(model_files)

solar_cat = []
predictedsolar_cat = []

for no_of_files, file_name3 in enumerate(model_files):

    # read the file
    with open(file_name3) as f:  # f is a file header
        reader = csv.reader(f, delimiter=",")
        d = list(reader)  # d is a list of list here.
        total_rows = len(d)

        # variable i starts from 1 so that header is skipped
        for i in range(1, total_rows):
            item1 = d[i][1]
            item2 = d[i][2]

            solar_cat.append(float(item1))
            predictedsolar_cat.append(float(item2))

len_array = len(solar_cat)

# Converting to numpys
solar_cat = np.array(solar_cat)
predictedsolar_cat = np.array(predictedsolar_cat)
RMSE_L = np.sqrt(np.mean((predictedsolar_cat-solar_cat)**2))
print ('Root mean square for Donatelli and Campbell model = ', RMSE_L , 'Watt/m2')


arr1 = predictedsolar_cat.reshape(-1, 1)
arr2 = solar_cat.reshape(-1, 1)
arr3 = np.hstack((arr1, arr2))
rho, pval = scipy.stats.spearmanr(arr3)
print ('correlation for Donatelli and Campbell model = ', rho)


rcParams.update({'figure.autolayout': True})
plt.figure(5, figsize=(5, 4))
plt.scatter(solar_cat, predictedsolar_cat, marker='o' , linewidths=0.005 , alpha = 0.01)
plt.xlabel(r'Measured solar irradiance [W/m$^2$]', fontsize=14)
plt.ylabel(r'Estimated solar irradiance [W/m$^2$]', fontsize=14)
plt.grid(True)
plt.xlim((-200, 1400))
plt.ylim((-100, 800))
plt.savefig('./results/scatter-DC.png', format='png')
plt.show()