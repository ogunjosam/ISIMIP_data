# All the prcp indices

import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt


def prcp_mean(prcp):
	"""

	:param prcp:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_years = prcp.reshape(-1, 365)
	return np.nanmean(prcp_years, axis=1)


def rx1_day(prcp):
	"""
	annual single day prcp max
	:param prcp:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_years = prcp.reshape(-1, 365)
	return np.nanmax(prcp_years, axis=1)


def rx5_day(prcp, stretch=5):
	"""
	max prcp in a 5 day stretch
	:param stretch: length of stretch to calculate over
	:param prcp:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_stretches = np.convolve(prcp, np.ones(stretch), 'same')
	prcp_years = prcp_stretches.reshape(-1, 365)
	return np.nanmax(prcp_years, axis=1)


def r95p(prcp, threshold=95, reference_prcp=None):
	"""
	Annual total PRCP when RR > 95th percentile
	:param reference_prcp:
	:param prcp:
	:param threshold:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_years = prcp.reshape(-1, 365)
	if reference_prcp is None:
		reference_prcp = prcp
	thresh_value = np.percentile(reference_prcp[reference_prcp >= 0.1], threshold)
	output = []
	for i in range(prcp_years.shape[0]):
		prcp_year = prcp_years[i, :]
		output.append(np.mean(prcp_year[prcp_year > thresh_value]))
	return output


def sdii(prcp, threshold=0.1):
	"""
	simple precipitation intensity index -- precipitation on wet days
	:param threshold:
	:param prcp:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_years = prcp.reshape(-1, 365)
	sdii_values = []
	for i in range(prcp_years.shape[0]):
		prcp_year = prcp_years[i, :]
		sdii_values.append(np.nanmean(prcp_year[prcp_year > threshold]))
	return sdii_values


def cdd(prcp, thresh=0.1):
	"""
	longest spell of consecutive dry days per year
	:param prcp:
	:param thresh:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_years = prcp.reshape(-1, 365)
	cdd_values = []
	for i in range(prcp_years.shape[0]):
		cdd_values.append(
			max((sum(1 for _ in group) for value, group in groupby(prcp_years[i, :]) if value < thresh), default=0))
	return cdd_values


def cwd(prcp, thresh=0.1):
	"""
	longest spell of consecutive wet days per year
	:param prcp:
	:param thresh:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_years = prcp.reshape(-1, 365)
	cwd_values = []
	for i in range(prcp_years.shape[0]):
		cwd_values.append(
			max((sum(1 for _ in group) for value, group in groupby(prcp_years[i, :]) if value > thresh), default=0))
	return cwd_values

def calculate_cdd(prcp, thresh=0.1):
   
    prcp = np.array(prcp)
    
    # Input validation
    if prcp.size == 0:
        raise ValueError("Input precipitation array is empty")
    if not np.isfinite(prcp).all():
        raise ValueError("Input contains non-finite values")
    if thresh < 0:
        raise ValueError("Threshold cannot be negative")
        
    # Handle both 1D and 2D inputs
    if prcp.ndim == 1:
        prcp = prcp.reshape(1, -1)
    elif prcp.ndim > 2:
        raise ValueError("Input array must be 1D or 2D")
    
    cwd_values = []
    
    for row in prcp:
        # Create binary array where 1 indicates wet day
        wet_days = (row < thresh).astype(int)
        
        # Find lengths of all wet spells
        wet_spells = [sum(1 for _ in group) for value, group in groupby(wet_days) if value == 1]
        
        # Get maximum spell length (0 if no wet spells found)
        max_spell = max(wet_spells) if wet_spells else 0
        cwd_values.append(max_spell)
    
    return np.array(cwd_values)

def calculate_cwd(prcp, thresh=0.1):
   
    prcp = np.array(prcp)
    
    # Input validation
    if prcp.size == 0:
        raise ValueError("Input precipitation array is empty")
    if not np.isfinite(prcp).all():
        raise ValueError("Input contains non-finite values")
    if thresh < 0:
        raise ValueError("Threshold cannot be negative")
        
    # Handle both 1D and 2D inputs
    if prcp.ndim == 1:
        prcp = prcp.reshape(1, -1)
    elif prcp.ndim > 2:
        raise ValueError("Input array must be 1D or 2D")
    
    cwd_values = []
    
    for row in prcp:
        # Create binary array where 1 indicates wet day
        wet_days = (row > thresh).astype(int)
        
        # Find lengths of all wet spells
        wet_spells = [sum(1 for _ in group) for value, group in groupby(wet_days) if value == 1]
        
        # Get maximum spell length (0 if no wet spells found)
        max_spell = max(wet_spells) if wet_spells else 0
        cwd_values.append(max_spell)
    
    return np.array(cwd_values)

def r10mm(prcp, threshold=10):
	"""
	annual count of days with precipitation greater than threshold, default of 10mm
	:param threshold:
	:param prcp:
	:return:
	"""
	prcp = np.array(prcp)
	prcp_years = prcp.reshape(-1, 365)
	r10_values = []
	for i in range(prcp_years.shape[0]):
		counts = np.sum(prcp_years[i] >= threshold)
		r10_values.append(counts)
	return r10_values


def test_indices():
	num_years = 10
	data = np.random.gamma(1, size=num_years * 365, scale=5)
	#plt.plot(data)
	#plt.show()
	func_list = [prcp_mean, rx1_day, rx5_day, r95p, sdii, cdd, cwd, r10mm]
	for func in func_list:
		print(func.__name__)
		output = func(data)
		# plt.plot(output)
		# plt.title(str(func.__name__))
		print(output)


if __name__ == '__main__':
	test_indices()
