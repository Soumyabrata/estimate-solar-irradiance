# Import the libraries

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import exifread
import operator
import pandas as pd
import bisect
import datetime
import pysolar
from pysolar.solar import *
from sklearn import linear_model, datasets
from scipy.interpolate import UnivariateSpline
from tempfile import TemporaryFile
from matplotlib.dates import DateFormatter

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# User defined functions
# from find_sun import *
from normalize_array import *
# from clear_outliers import *
# from sun_positions_day import *
# from LDR_luminance import *
from import_WS import *
from remove_outliers import *
# from nearest import *
from SG_solarmodel import *


# This contains the *generated* luminance files.
training_loc = './Solar_Radiation/JSTARfiles/LuminanceFiles/LuminanceFileswithCP*.txt'
training_files = glob.glob(training_loc)
training_files = sorted(training_files)
print (training_files)



# Importing the weather station
CSV_file ='radiosonde_Sep2016.csv'
(time_range,solar_range) = import_WS(CSV_file)



# Generate the files
for no_of_files, file_name1 in enumerate(training_files):

    # read the finalised HDR file
    with open(file_name1) as f:  # f is a file header
        reader = csv.reader(f, delimiter=",")
        d = list(reader)  # d is a list of list here.
        total_rows = len(d)

        img_date = []
        img_time = []

        img_unityLOWlumi = np.zeros(total_rows - 1)
        img_fnumber = np.zeros(total_rows - 1)
        img_etime = np.zeros(total_rows - 1)
        img_iso = np.zeros(total_rows - 1)

        # variable i starts from 1 so that header is skipped
        for i in range(1, total_rows):
            date_item = d[i][1]
            time_item = d[i][2]
            fnumber_item = d[i][3]
            etime_item = d[i][4]
            iso_item = d[i][5]
            unitylumi_item = d[i][6]

            img_date.append(date_item)
            img_time.append(time_item)

            img_unityLOWlumi[i - 1] = unitylumi_item
            img_fnumber[i - 1] = fnumber_item
            img_etime[i - 1] = etime_item
            img_iso[i - 1] = iso_item

        # This is for plotting dates in X-axis
        # Common for low- med- and high- LDR images

        time_datapoints = []

        for i in range(0, len(img_date)):
            YY = int(img_date[i][0:4])
            MON = int(img_date[i][5:7])
            DD = int(img_date[i][8:10])
            HH = int(img_time[i][0:2])
            MM = int(img_time[i][3:5])
            SS = int(img_time[i][6:8])
            sw = datetime.datetime(YY, MON, DD, HH, MM, SS)

            time_datapoints.append(sw)

        my_date_string = str(YY) + '-' + str(MON) + '-' + str(DD)
        # ---------------------------------------------------

        # ### Corresponding weather station data for a particular timeseries of a date

        WS_solar = np.zeros(len(time_datapoints))
        # print (len(time_datapoints))
        # print (time_datapoints)

        for i in range(0, len(time_datapoints)):
            time_here = time_datapoints[i]
            # print (time_here)
            index = bisect.bisect_right(time_range, time_here)
            # print (index)
            WS_solar[i] = float(solar_range[index])

        # print (len(WS_solar))
        # print (len(img_unityLOWlumi))
        # print (len(time_datapoints))

        # Check continuous repeated elements
        check_value = WS_solar[0]
        repeat_count = 0
        for t in range(0, len(WS_solar)):
            if check_value == WS_solar[t]:
                repeat_count = repeat_count + 1
            else:
                repeat_count = 0  # Please reset it
                check_value = WS_solar[t]

        print('Calculating for ', my_date_string)

        print('Number of repeated counts = ', repeat_count)

        if repeat_count < 50:

            (d_img, c_img) = remove_outliers(img_unityLOWlumi)
            (d_WS, c_WS) = remove_outliers(WS_solar)

            temp_removed = list(set(d_img) | set(d_WS))
            total_list = list(set(d_img) | set(c_img))
            index_taken = list(set(total_list) - set(temp_removed))
            index_taken = np.array(index_taken)

            # Converting to numpy array
            time_datapoints = np.array(time_datapoints)
            img_unityLOWlumi = np.array(img_unityLOWlumi)
            img_fnumber = np.array(img_fnumber)
            img_etime = np.array(img_etime)
            img_iso = np.array(img_iso)

            # Calculating the correlation co-efficient
            y_vectNORM = normalize_array(WS_solar[index_taken])
            y_vect = WS_solar[index_taken]
            x_vectNORM = normalize_array(img_unityLOWlumi[index_taken])
            x_vect = img_unityLOWlumi[index_taken]
            time_vect = list(time_datapoints[i] for i in index_taken)

            x_vectCOS = []
            clear_sky_rad = []
            for t, sometime in enumerate(time_vect):

                sometime_clearsky = datetime.datetime(sometime.year, sometime.month, sometime.day, sometime.hour,
                                                      sometime.minute, sometime.second, 0, pytz.UTC)

                # Making timezone aware
                sometime_angle = datetime.datetime(sometime.year, sometime.month, sometime.day, (sometime.hour - 8 ) % 24,
                                                sometime.minute, sometime.second, 0, pytz.UTC)

                elevation_angle = get_altitude(1.3429943, 103.6810899, sometime_angle)
                zenith_angle = 90 - elevation_angle
                theta = ((np.pi) / 180) * zenith_angle
                x_vectCOS.append(x_vect[t] * np.cos(theta))


                CSR = SG_model(sometime_clearsky)
                clear_sky_rad.append(CSR)

            clear_sky_rad = np.array(clear_sky_rad)

            x_vectCOS = np.array(x_vectCOS)
            x_vectCOS_NORM = normalize_array(x_vectCOS)

            # New normalization technique
            lumi_vector = x_vectCOS
            solar_vector = y_vectNORM
            a1 = np.multiply(lumi_vector, solar_vector)
            v1 = np.sum(a1)
            b1 = np.square(lumi_vector)
            v2 = np.sum(b1)
            factor_value = v1 / v2
            x_vectCOS_newNORM = factor_value * x_vectCOS

            # Florian's technique wrt Weather Station data
            L_vector = x_vectCOS
            S_vector = y_vect
            q1 = np.multiply(L_vector, S_vector)
            q2 = np.sum(q1)
            q3 = np.square(L_vector)
            q4 = np.sum(q3)
            factor_value4 = q2 / q4
            x_vectCOS_FN = factor_value4 * L_vector

            x_fnumber = img_fnumber[index_taken]
            x_iso = img_iso[index_taken]
            x_etime = img_etime[index_taken]

            r = np.corrcoef(x_vectCOS, y_vect)
            print(r[0, 1])
            print('---------')

            my_cor_string = 'R=' + str((round(r[0, 1] * 100)) / 100)

            # Plot the time series plots
            fig = plt.figure(1 + no_of_files, figsize=(15, 3))
            plt.plot(time_vect, x_vectCOS_FN, 'b', label='Proposed model', linestyle='--', linewidth=2)
            plt.plot(time_vect, S_vector, 'r', label='Weather station')
            plt.plot(time_vect, clear_sky_rad, 'k', label='Clear sky model', linestyle='-.', linewidth=2)
            plt.legend(loc='upper right')
            # plt.title(my_date_string, fontsize=12)
            plt.title('2016-09-01', fontsize=12)
            plt.ylabel(r'Solar irradiance [W/m$^2$]', fontsize=14)
            plt.grid(True)
            fig.autofmt_xdate()
            formatter = DateFormatter('%H:%M')
            plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
            plt.savefig('./results/example-proposeddash.pdf', format='pdf')
            plt.show()



            file_name2 = './model-files/ModelFiles-' + str(YY) + '-' + str(MON) + '-' + str(DD) + '.txt'
            text_file2 = open(file_name2, "w")
            # Header line
            text_file2.write("Sl_No, FNumber, ETime, ISO, Lumi, LumiNORM, Solar, SolarNORM \n")

            for i, item in enumerate(x_vect):
                text_file2.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                i, x_fnumber[i], x_etime[i], x_iso[i], x_vect[i], x_vectCOS_newNORM[i], y_vect[i], y_vectNORM[i]))

            #
            text_file2.close()
