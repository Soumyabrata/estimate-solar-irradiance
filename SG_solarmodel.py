import numpy as np
import datetime
from pysolar.solar import *
import pytz
import math



def SG_model(datetime_date):
	latitude = 1.3429943
	longitude = 103.6810899

	date_part = datetime_date - datetime.timedelta(hours=8) # Singapore is (UTC + 8) Hours

	
	# Singapore model
	elevation_angle = get_altitude(latitude, longitude, date_part)
	zenith_angle = 90-elevation_angle
	theta = (np.pi/180)*zenith_angle

	day_of_year = date_part.timetuple().tm_yday

	tau = 2*np.pi*(day_of_year - 1)/365
	E0 = 1.00011 + 0.034221*np.cos(tau) + 0.001280*np.sin(tau) + 0.000719*np.cos(2*tau) + 0.000077*np.sin(2*tau)
	Isc = 1366.1

	try:
		f1 = math.pow(np.cos(theta), 1.3644)
		f2 = math.pow(np.e, (-0.0013 * (90 - zenith_angle)))
		Gc = 0.8277*E0*Isc*f1*f2
	except:
		Gc=0
	
	return (Gc)


# # Some example
# my_hour = 10
# new_hour = int(my_hour)
# datetime_date = datetime.datetime(2016, 9, 1, new_hour, 0, 0, 0, pytz.UTC)
# Gc = SG_model(datetime_date)
# print (Gc)