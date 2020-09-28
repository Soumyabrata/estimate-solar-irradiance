# import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scripts.import_weather import *
from scripts.SG_solarmodel import *
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Import the weather data file
CSV_file ='./data/September2016.csv'

# extract the different values based on the format of the weather data.
(time_range,solar_range,temperature_range,humidity_range,dewpoint_range,windspeed_range,winddirection,pressure,rainfallrate) = import_weather(CSV_file)
temperature_range = np.array(temperature_range)
solar_range = np.array(solar_range)
rainfallrate = np.array(rainfallrate)


# Extracting unique dates
date_list = []
for onetime in time_range:
    date_ext = onetime.strftime('%Y-%m-%d')
    date_list.append(date_ext)
unik_dates = list(set(date_list))
unik_dates = sorted(unik_dates)
print('Unique dates are ', unik_dates)


# Sample date where we show the illustration for.
samplestring = ['2016-09-01']

x_vector = []
for no_of_files, mydatestring in enumerate(samplestring):
    
    print ('Plotting figures for: ', mydatestring)

    # Performing for a single date
    mydateindex = []
    for kot, alldate in enumerate(date_list):
        if mydatestring in alldate:
            mydateindex.append(kot)

    mydateindex = np.array(mydateindex)
    mytemparray = temperature_range[mydateindex]
    mysolararray = solar_range[mydateindex]
    myrainarray = rainfallrate[mydateindex]
    mytimearray = [time_range[i] for i in mydateindex]

    # Calculate the clear sky radiation
    # This latitude and longitude value is obtained for the location in Singapore.
    latitude = 1.3429943
    longitude = 103.6810899
    clear_sky_rad = []

    # Accessing each datetime objects in a single day
    for timeinstant in mytimearray:

        # Making datetime object as timezone aware
        timeinstant = datetime.datetime(timeinstant.year, timeinstant.month, timeinstant.day, timeinstant.hour,
                                        timeinstant.minute, timeinstant.second, 0, pytz.UTC)
        CSR = SG_model(timeinstant)
        clear_sky_rad.append(CSR)
    clear_sky_rad = np.array(clear_sky_rad)

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

    mytimearray_ST = [mytimearray[i] for i in selecttime_index]
    mysolararray_ST = mysolararray[selecttime_index]
    myclearsky_ST = clear_sky_rad[selecttime_index]

    myclearsky_ST = myclearsky_ST + 0.000000001 # so that we don't encounter divide by zero scenario.

    diff_array = np.abs(myclearsky_ST - mysolararray_ST)
    per_diff = (np.divide(diff_array, myclearsky_ST)) * 100


    # This no_of_files variable is included if we intend to list out the figures for multiple dates.
    fig = plt.figure(1 + 2 * no_of_files, figsize=(7, 3))
    ax = plt.subplot(111)
    ax.plot(mytimearray_ST, myclearsky_ST, 'k', label='Clear sky model', linestyle='-.', linewidth=2)
    ax.plot(mytimearray_ST, mysolararray_ST, 'r', label='Weather station')
    ax.legend(loc='upper right')
    ax.xaxis_date()
    plt.title(mydatestring, fontsize=12)
    plt.ylabel(r'Solar irradiance [W/m$^2$]', fontsize=12)
    fig.autofmt_xdate()
    plt.grid(b=None, which='major', axis='both')
    formatter = DateFormatter('%H:%M')
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.savefig('./results/Figure5a.pdf', format='pdf')
    plt.show()

    fig = plt.figure(2 + 2 * no_of_files, figsize=(7, 3))
    ax = plt.subplot(111)
    ax.plot(mytimearray_ST, per_diff, 'b', label='Deviation')
    ax.legend(loc='lower right')
    ax.xaxis_date()
    plt.title(mydatestring, fontsize=12)
    plt.ylabel('Deviation from clear sky [%]', fontsize=12)
    fig.autofmt_xdate()
    plt.grid(b=None, which='major', axis='both')
    formatter = DateFormatter('%H:%M')
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.savefig('./results/Figure5b.pdf', format='pdf')
    plt.show()
    
    
    
    
    
