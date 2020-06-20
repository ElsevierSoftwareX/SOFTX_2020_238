#!/usr/bin/env python
# Copyright (C) 2020  Aaron Viets
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import sys
import numpy as np
import math
import time
import timeit


#
# Below are functions to compute FFTs and inverse FFTs, including the special cases
# of purely real input, at greater-than-double precision.  The data type used is
# Python's numpy.float128, and precision depends on your machine's platform.  Check
# the resolution using numpy.finfo(numpy.float128).
#


# Compute pi and 2*pi to long double precision.
pi = np.floor(np.float128(np.pi * 1e14)) / 1e14 + 3.2384626433832795e-15
two_pi = np.floor(np.float128(np.pi * 2e14)) / 1e14 + 6.476925286766559e-15


# A function to find prime factors of N, the size of the input data
def find_prime_factors(N):

	prime_factors = np.array([], dtype = int)
	product = N
	factor = 2
	while(factor <= product):
		if product % factor:
			factor += 1
		else:
			prime_factors = np.append(prime_factors, factor)
			product = product // factor

	prime_factors = np.append(prime_factors, 1)

	return prime_factors


# For Bluestein's algorithm.  Find a good padded length.
def find_M(M_min):

	M = pow(2, int(np.ceil(np.log2(M_min))))
	prime_factors = 2 * np.ones(int(np.log2(M) + 1), dtype = np.int64)

	if 9 * M >= 16 * M_min and M >= 16:
		prime_factors[-5] = prime_factors[-4] = 3
		prime_factors[-3] = 1
		return int(M * 9 / 16), prime_factors[:-2]
	elif 5 * M >= 8 * M_min and M >= 8:
		prime_factors[-4] = 5
		prime_factors[-3] = 1
		return int(M * 5 / 8), prime_factors[:-2]
	elif 3 * M >= 4 * M_min and M >= 4:
		prime_factors[-3] = 3
		prime_factors[-2] = 1
		return int(M * 3 / 4), prime_factors[:-1]
	elif 7 * M >= 8 * M_min and M >= 8:
		prime_factors[-4] = 7
		prime_factors[-3] = 1
		return int(M * 7 / 8), prime_factors[:-2]
	elif 15 * M >= 16 * M_min and M >= 16:
		prime_factors[-5] = 3
		prime_factors[-4] = 5
		prime_factors[-3] = 1
		return int(M * 15 / 16), prime_factors[:-2]
	else:
		prime_factors[-1] = 1
		return int(M), prime_factors


# A function to find all positive integers less than N that are coprime to N
def find_coprime_group(N):

	coprime_group = np.array([1], dtype = int)
	for i in range(2, N - 2):
		if not i in coprime_group:
			i_is_coprime = True
			for multiple in range(N, i * N, N):
				if not multiple % i:
					i_is_coprime = False
					break
			if i_is_coprime:
				for j in range(1, 1 + int(np.log(N) // np.log(i))):
					np.append(coprime_group, pow(i, j))

	if N % 2 and N > 3 and (N - 2) not in coprime_group:
		np.append(coprime_group, N - 2)
	if N > 2 and (N - 1) not in coprime_group:
		np.append(coprime_group, N - 1)

	return coprime_group


# A function to find the primitive roots modulo N
def find_primitive_roots(N):

	if N < 5:
		return np.array([N - 1], dtype = int)

	coprime_group = find_coprime_group(N)
	N_coprimes = len(coprime_group)

	primitive_roots = np.array([], dtype = int)

	for coprime in coprime_group[1:]:
		element = coprime
		order = 1
		while element != 1:
			element = element * coprime % N
			order += 1

		if order == N_coprimes:
			np.append(primitive_roots, coprime)

	return primitive_roots


# A cheaper function to find only the smallest primitive root modulo a prime N
def find_smallest_primitive_root_prime(N):

	if N < 5:
		return N - 1

	coprime = 2
	while True:
		element = coprime
		order = 1
		while element != 1:
			element = element * coprime % N
			order += 1
		if order == N - 1:
			return coprime

		coprime += 1	


# A function to compute the array of exponentials
def find_exp_array(N, inverse = False):

	N = int(N)
	exp_array = np.zeros(N, dtype = np.complex256)

	# If this is the inverse DFT, just don't negate 2*pi
	if inverse:
		prefactor = two_pi * 1j
	else:
		prefactor = -two_pi * 1j

	if not N % 4:
		# It's a multiple of 4, so we know these values right away:
		exp_array[0] = 1 + 0j
		exp_array[N // 2] = -1 + 0j
		if inverse:
			exp_array[N // 4] = 0 + 1j
			exp_array[3 * N // 4] = 0 - 1j
		else:
			exp_array[N // 4] = 0 - 1j
			exp_array[3 * N // 4] = 0 + 1j

		# Only compute one fourth of the array, and use symmetry for the rest.
		for n in range(1, N // 4):
			exp_array[n] = np.exp(prefactor * n / N)
			exp_array[N // 2 - n] = -np.conj(exp_array[n])
			exp_array[N // 2 + n] = -exp_array[n]
			exp_array[N - n] = np.conj(exp_array[n])

	elif not N % 2:
		# It's a multiple of 2, so we know these values right away:
		exp_array[0] = 1 + 0j
		exp_array[N // 2] = -1 + 0j

		# Only compute one fourth of the array, and use symmetry for the rest.
		for n in range(1, N // 4 + 1):
			exp_array[n] = np.exp(prefactor * n / N)
			exp_array[N // 2 - n] = -np.conj(exp_array[n])
			exp_array[N // 2 + n] = -exp_array[n]
			exp_array[N - n] = np.conj(exp_array[n])

	else:
		# It's odd, but we still know this:
		exp_array[0] = 1 + 0j

		# Only compute half of the array, and use symmetry for the rest.
		for n in range(1, N // 2 + 1):
			exp_array[n] = np.exp(prefactor * n / N)
			exp_array[N - n] = np.conj(exp_array[n])

	return exp_array


# A function to compute the array of exponentials for Bluestein's algorithm
def find_exp_array2(N, inverse = False):

	# First compute the usual fft array
	exp_array = find_exp_array(2 * N, inverse = inverse)

	# Rearrange it
	return exp_array[pow(np.arange(N), 2) % (2 * N)]


# First, a discrete Fourier transform, evaluated according to the definition
def dft(td_data, exp_array = None, return_double = False, inverse = False):

	N = len(td_data)

	if exp_array is None:
		# Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		# the code do it once.
		exp_array = find_exp_array(N, inverse = inverse)

	fd_data = np.zeros(N, dtype = np.complex256)

	# The first term is the DC component, which is just the sum.
	fd_data[0] = sum(np.complex256(td_data))

	# Since this function is most often called by fft(), N is most likely a prime, so assume
	# there are no more trivial multiplications
	if N == 2:
		fd_data[1] += td_data[0]
		fd_data[1] -= td_data[1]
	else:
		for i in range(1, N):
			fd_data[i] += td_data[0]
			for j in range(1, N):
				fd_data[i] += td_data[j] * exp_array[i * j % N]

	if return_double:
		return np.complex128(fd_data)
	else:
		return fd_data


# A discrete inverse Fourier transform, evaluated according to the definition
def idft(fd_data, exp_array = None, return_double = False, normalize = True):

	if normalize:
		if return_double:
			return np.complex128(dft(fd_data, exp_array = exp_array, return_double = False, inverse = True) / len(fd_data))
		else:
			return dft(fd_data, exp_array = exp_array, return_double = False, inverse = True) / len(fd_data)
	else:
		return dft(fd_data, exp_array = exp_array, return_double = return_double, inverse = True)


# If the input is real, the output is conjugate-symmetric: fd_data[n] = conj(fd_data[N - n]).
# We can reduce the number of operations by a factor of ~2.  Also, we have the option to only
# output half of the result, since the second half is redundant.
def rdft(td_data, exp_array = None, return_double = False, return_full = False):

	N = len(td_data)
	N_out = N // 2 + 1

	if exp_array is None:
		# Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		# the code do it once.
		exp_array = find_exp_array(N)

	if return_full:
		fd_data = np.zeros(N, dtype = np.complex256)
	else:
		fd_data = np.zeros(N_out, dtype = np.complex256)

	# The first term is the DC component, which is just the sum.
	fd_data[0] = sum(np.complex256(td_data))

	# Since this function is most often called by fft(), N is most likely a prime, so assume
	# there are no more trivial multiplications
	if N == 2:
		fd_data[1] += td_data[0]
		fd_data[1] -= td_data[1]
	else:
		for i in range(1, N_out):
			fd_data[i] += td_data[0]
			for j in range(1, N):
				fd_data[i] += td_data[j] * exp_array[i * j % N]

	if return_full and N > 2:
		# Then fill in the second half
		fd_data[N_out : N] = np.conj(fd_data[1 : N - N_out + 1][::-1])

	if return_double:
		return np.complex128(fd_data)
	else:
		return fd_data


# Inverse of the above real-input DFT.  So the output of this is real and the input is assumed
# to be shortened to N // 2 + 1 samples to avoid redundancy.
def irdft(fd_data, exp_array = None, return_double = False, N = None, normalize = True):

	N_in = len(fd_data)

	if N is None:
		# Find N, the original number of samples. If the imaginary part of the last
		# sample is zero, assume N was even
		if np.imag(fd_data[-1]) == 0:
			N = (N_in - 1) * 2
		elif np.real(fd_data[-1]) == 0:
			N = N_in * 2 - 1
		elif abs(np.imag(fd_data[-1]) / np.real(fd_data[-1])) < 1e-14:
			N = (N_in - 1) * 2
		else:
			N = N_in * 2 - 1

	if exp_array is None:
		# Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		# the code do it once.
		exp_array = find_exp_array(N, inverse = True)

	td_data = np.zeros(N, dtype = np.float128)

	# The first term is the DC component, which is just the sum.
	td_data[0] = sum(np.float128(np.real(fd_data))) + sum(np.float128(np.real(fd_data)[1 : 1 + N - N_in]))

	# Since this function is most often called by irfft(), N is most likely a prime, so assume
	# there are no more trivial multiplications
	if N == 2:
		td_data[1] += np.real(fd_data[0])
		td_data[1] -= np.real(fd_data[1])
	else:
		for i in range(1, N):
			td_data[i] += np.real(fd_data[0])
			for j in range(1, N_in):
				td_data[i] += np.real(fd_data[j] * exp_array[i * j % N])
			for j in range(N_in, N):
				td_data[i] += np.real(np.conj(fd_data[N - j]) * exp_array[i * j % N])

	if normalize:
		if return_double:
			return np.float64(td_data / N)
		else:
			return td_data / N
	else:
		if return_double:
			return np.float64(td_data)
		else:
			return td_data


# A fast Fourier transform using the Cooley-Tukey algorithm, which
# factors the length N to break up the transform into smaller transforms
def fft(td_data, prime_factors = None, exp_array = None, return_double = False, inverse = False, M = None, M_prime_factors = None, M_exp_array2 = None, M_exp_array = None):

	N = len(td_data)

	if N < 2:
		if return_double:
			return np.complex128(td_data)
		else:
			return np.complex256(td_data)

	if prime_factors is None:
		# Find prime factors
		prime_factors = find_prime_factors(N)

		# Check if we will need to use prime_fft() for this
		if prime_factors[-2] >= 37:
			# Find the first member greater than or equal to 37
			i = 0
			while prime_factors[i] < 37:
				i += 1
			M, M_prime_factors = find_M(2 * np.product(prime_factors[i:]) - 1)

			M_exp_array2 = find_exp_array2(np.product(prime_factors[i:]), inverse = inverse)
			M_exp_array = find_exp_array(M)

	if exp_array is None and prime_factors[0] < 37:
		# Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		# the code do it once.
		exp_array = find_exp_array(N, inverse = inverse)

	if prime_factors[0] >= 37:
		# Use Bluestein's algorithm for a prime-length fft
		return prime_fft(td_data, return_double = return_double, inverse = inverse, exp_array2 = M_exp_array2, M = M, prime_factors = M_prime_factors, exp_array = M_exp_array)
	elif prime_factors[0] == N:
		# Do an ordinary DFT
		return dft(td_data, exp_array = exp_array, return_double = return_double)
	else:
		# We will break this up into smaller Fourier transforms
		fd_data = np.zeros(N, dtype = np.complex256)
		num_ffts = prime_factors[0]
		N_mini = N // num_ffts
		for i in range(num_ffts):
			fd_data[i * N_mini : (i + 1) * N_mini] = fft(td_data[i::num_ffts], prime_factors = prime_factors[1:], exp_array = exp_array[::num_ffts], return_double = False, M = M, M_prime_factors = M_prime_factors, M_exp_array2 = M_exp_array2, M_exp_array = M_exp_array)

		# Now we need to "mix" the output appropriately.  First, copy all but the first fft.
		fd_data_copy = np.copy(fd_data[N_mini:])

		# Apply phase rotations to all but the first fft
		for i in range(N_mini, N):
			exp_index = (i * (i // N_mini)) % N
			# Do a multiplication only if we have to
			if(exp_index):
				fd_data[i] *= exp_array[exp_index]

		# Add the first fft to all the others
		for i in range(N_mini, N, N_mini):
			fd_data[i : i + N_mini] += fd_data[:N_mini]

		# Now we have to use the copied data.  Apply phase rotations and add to all other locations.
		for i in range(N_mini, N):
			copy_index = i - N_mini
			dst_indices = list(range(i % N_mini, N, N_mini))
			# We've already taken care of the below contribution (2 for loops ago), so remove it
			dst_indices.remove(i)
			for j in dst_indices:
				exp_index = (j * (i // N_mini)) % N
				# Do a multiplication only if we have to
				if(exp_index):
					fd_data[j] += fd_data_copy[copy_index] * exp_array[exp_index]
				else:
					fd_data[j] += fd_data_copy[copy_index]
		# Done
		if return_double:
			return np.complex128(fd_data)
		else:
			return fd_data


# An inverse fast Fourier transform that factors the length N to break up the
# transform into smaller transforms
def ifft(fd_data, prime_factors = None, exp_array = None, return_double = False, normalize = True):

	if normalize:
		if return_double:
			return np.complex128(fft(fd_data, prime_factors = prime_factors, exp_array = exp_array, return_double = False, inverse = True) / len(fd_data))
		else:
			return fft(fd_data, prime_factors = prime_factors, exp_array = exp_array, return_double = False, inverse = True) / len(fd_data)
	else:
		return fft(fd_data, prime_factors = prime_factors, exp_array = exp_array, return_double = return_double, inverse = True)


# If the input is real, the output is conjugate-symmetric: fd_data[n] = conj(fd_data[N - n]).
# We can reduce the number of operations by a factor of ~2.  Also, we have the option to only
# output half of the result, since the second half is redundant.
def rfft(td_data, prime_factors = None, exp_array = None, return_double = False, return_full = False, M = None, M_prime_factors = None, M_exp_array2 = None, M_exp_array = None):

	N = len(td_data)
	N_out = N // 2 + 1

	if N < 2:
		if return_double:
			return np.complex128(td_data)
		else:
			return np.complex256(td_data)

	if prime_factors is None:
		# Find prime factors
		prime_factors = find_prime_factors(N)

		# Check if we will need to use prime_rfft() for this
		if prime_factors[-2] >= 61:
			# Find the first member greater than or equal to 61
			i = 0
			while prime_factors[i] < 61:
				i += 1
			M_in = np.product(prime_factors[i:])
			M_out = M_in // 2 + 1
			M, M_prime_factors = find_M(M_in + M_out - 1)
			M_exp_array2 = find_exp_array2(M_in)
			M_exp_array = find_exp_array(M)

	if exp_array is None and prime_factors[0] < 61:
		# Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		# the code do it once.
		exp_array = find_exp_array(N)

	if prime_factors[0] >= 61:
		# Use Bluestein's algorithm for a prime-length fft
		return prime_rfft(td_data, return_double = return_double, return_full = return_full, exp_array2 = M_exp_array2, M = M, prime_factors = M_prime_factors, exp_array = M_exp_array)
	elif prime_factors[0] == N:
		# Do an ordinary DFT
		return rdft(td_data, exp_array = exp_array, return_double = return_double, return_full = return_full)
	else:
		# We will break this up into smaller Fourier transforms.  Therefore, we still
		# need to allocate enough memory for N elements.
		fd_data = np.zeros(N, dtype = np.complex256)
		num_ffts = prime_factors[0]
		N_mini = N // num_ffts
		N_mini_out = N_mini // 2 + 1		
		for i in range(num_ffts):
			fd_data[i * N_mini : (i + 1) * N_mini] = rfft(td_data[i::num_ffts], prime_factors = prime_factors[1:], exp_array = exp_array[::num_ffts], return_double = False, return_full = True, M = M, M_prime_factors = M_prime_factors, M_exp_array2 = M_exp_array2, M_exp_array = M_exp_array)

		# Now we need to "mix" the output appropriately.  First, copy all but the first fft.
		populated_indices = [x for x in range(N_mini, N) if x % N_mini < N_mini_out]
		fd_data_copy = fd_data[populated_indices]

		# Apply phase rotations to all but the first fft
		for i in range(N_mini, N_out):
			exp_index = (i * (i // N_mini)) % N
			# Do a multiplication only if we have to
			if(exp_index):
				fd_data[i] *= exp_array[exp_index]

		# Add the first fft to all the others
		for i in range(N_mini, N_out, N_mini):
			fd_data[i : i + N_mini] += fd_data[:N_mini]

		# Now we have to use the copied data.  Apply phase rotations and add to all other locations.
		for i in range(len(fd_data_copy)):
			original_index = N_mini + i // N_mini_out * N_mini + i % N_mini_out
			dst_indices = list(range(original_index % N_mini, N_out, N_mini))
			if original_index in dst_indices:
				# We've already taken care of this contribution (2 for loops ago), so remove it
				dst_indices.remove(original_index)
			for j in dst_indices:
				exp_index = (j * (original_index // N_mini)) % N
				# Do a multiplication only if we have to
				if(exp_index):
					fd_data[j] += fd_data_copy[i] * exp_array[exp_index]
				else:
					fd_data[j] += fd_data_copy[i]

			if original_index % N_mini and original_index % N_mini < (N_mini + 1) // 2:
				# Then handle the contribution from the complex conjugate
				original_index += N_mini - 2 * (original_index % N_mini)
				dst_indices = list(range(original_index % N_mini, N_out, N_mini))
				if original_index in dst_indices:
					# We've already taken care of this contribution, so remove it
					dst_indices.remove(original_index)
				for j in dst_indices:
					exp_index = (j * (original_index // N_mini)) % N
					# Do a multiplication only if we have to
					if(exp_index):
						fd_data[j] += np.conj(fd_data_copy[i]) * exp_array[exp_index]
					else:
						fd_data[j] += np.conj(fd_data_copy[i])

		if not N % 2:
			# The Nyquist component is real
			fd_data[N_out - 1] = np.real(fd_data[N_out - 1]) + 0j

		if return_full and N > 2:
			# Then fill in the second half
			fd_data[N_out : N] = np.conj(fd_data[1 : N - N_out + 1][::-1])
			if return_double:
				return np.complex128(fd_data)
			else:
				return fd_data
		else:
			# Shorten the array
			if return_double:
				return np.complex128(fd_data[:N_out])
			else:
				return fd_data[:N_out]


# Inverse of the above real-input FFT.  So the output of this is real and the input is assumed
# to be shortened to N // 2 + 1 samples to avoid redundancy.
def irfft(fd_data, prime_factors = None, exp_array = None, return_double = False, normalize = True, M_fft = None, M_fft_prime_factors = None, M_fft_exp_array2 = None, M_fft_exp_array = None, M_irfft = None, M_irfft_prime_factors = None, M_irfft_exp_array2 = None, M_irfft_exp_array = None):

	N_in = len(fd_data)

	if N_in < 2:
		if return_double:
			return np.float64(fd_data)
		else:
			return np.float128(fd_data)

	if prime_factors is None:
		# First, find N, the original number of samples. If the imaginary part of the last
		# sample is zero, assume N was even
		if np.imag(fd_data[-1]) == 0:
			N = (N_in - 1) * 2
		elif np.real(fd_data[-1]) == 0:
			N = N_in * 2 - 1
		elif abs(np.imag(fd_data[-1]) / np.real(fd_data[-1])) < 1e-14:
			N = (N_in - 1) * 2
		else:
			N = N_in * 2 - 1
		# Find prime factors
		prime_factors = find_prime_factors(N)

		# Check if we will need to use prime_irfft() for this
		if prime_factors[-2] >= 17:
			# Find the first member greater than or equal to 17
			i = 0
			while prime_factors[i] < 17:
				i += 1
			M_irfft, M_irfft_prime_factors = find_M(2 * np.product(prime_factors[i:]) - 1)

			M_irfft_exp_array2 = find_exp_array2(np.product(prime_factors[i:]), inverse = True)
			M_irfft_exp_array = find_exp_array(M_irfft)

		# Check if we will need to use prime_fft() for this
		if prime_factors[-2] >= 37:
			# Find the first member greater than or equal to 37
			i = 0
			while prime_factors[i] < 37:
				i += 1
			M_fft, M_fft_prime_factors = find_M(2 * np.product(prime_factors[i:]) - 1)

			M_fft_exp_array2 = find_exp_array2(np.product(prime_factors[i:]), inverse = True)
			M_fft_exp_array = find_exp_array(M_fft)

	else:
		N = np.product(prime_factors)

	if exp_array is None and prime_factors[0] < 17:
		# Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		# the code do it once.
		exp_array = find_exp_array(N, inverse = True)

	if prime_factors[0] >= 17:
		# Use Bluestein's algorithm for a prime-length fft
		return prime_irfft(fd_data, return_double = return_double, N = N, normalize = normalize, exp_array2 = M_irfft_exp_array2, M = M_irfft, prime_factors = M_irfft_prime_factors, exp_array = M_irfft_exp_array)
	elif prime_factors[0] == N:
		# Do an ordinary DFT
		return irdft(fd_data, exp_array = exp_array, return_double = return_double, N = N, normalize = normalize)
	else:
		# We will break this up into smaller Fourier transforms
		td_data = np.zeros(N, dtype = np.float128)
		num_ffts = prime_factors[0]
		N_mini = N // num_ffts
		td_data[:N_mini] = irfft(fd_data[0::num_ffts], prime_factors = prime_factors[1:], exp_array = exp_array[::num_ffts], return_double = False, normalize = False, M_fft = M_fft, M_fft_prime_factors = M_fft_prime_factors, M_fft_exp_array2 = M_fft_exp_array2, M_fft_exp_array = M_fft_exp_array, M_irfft = M_irfft, M_irfft_prime_factors = M_irfft_prime_factors, M_irfft_exp_array2 = M_irfft_exp_array2, M_irfft_exp_array = M_irfft_exp_array)

		# The rest of the transforms will, in general, produce complex output
		td_data_complex = np.zeros(N - N_mini, dtype = np.complex256)
		for i in range(1, num_ffts):
			td_data_complex[(i - 1) * N_mini : i * N_mini] = fft(np.concatenate((fd_data, np.conj(fd_data)[1 : 1 + N - len(fd_data)][::-1]))[i::num_ffts], prime_factors = prime_factors[1:], exp_array = exp_array[::num_ffts], return_double = False, M = M_fft, M_prime_factors = M_fft_prime_factors, M_exp_array2 = M_fft_exp_array2, M_exp_array = M_fft_exp_array)

		# Now we need to "mix" the output appropriately.  Start by adding the first ifft to the others.
		for i in range(N_mini, N, N_mini):
			td_data[i : i + N_mini] += td_data[:N_mini]

		# Now use the complex data.  Apply phase rotations and add real parts to all other locations.
		for i in range(N_mini, N):
			complex_index = i - N_mini
			dst_indices = list(range(i % N_mini, N, N_mini))
			for j in dst_indices:
				exp_index = (j * (i // N_mini)) % N
				# Do a multiplication only if we have to
				if(exp_index):
					td_data[j] += np.real(td_data_complex[complex_index] * exp_array[exp_index])
				else:
					td_data[j] += np.real(td_data_complex[complex_index])
		# Done
		if normalize:
			if return_double:
				return np.float64(td_data / N)
			else:
				return td_data / N
		else:
			if return_double:
				return np.float64(td_data)
			else:
				return td_data


# Bluestein's algorithm for FFTs of prime length, for which the Cooley-Tukey algorithm is
# ineffective.  Make the replacement nk -> -(k - n)^2 / 2 + n^2 / 2 + k^2 / 2.
# Then X_k = sum_(n=0)^(N-1) x_n * exp(-2*pi*i*n*k/N)
#	   = exp(-pi*i*k^2/N) * sum_(n=0)^(N-1) x_n * exp(-pi*i*n^2/N) * exp(pi*i*(k-n)^2/N)
# This can be done as a cyclic convolution between the sequences a_n = x_n * exp(-pi*i*n^2/N)
# and b_n = exp(pi*i*n^2/N), with the output multiplied by conj(b_k).
# a_n and b_n can be padded with zeros to make their lengths a power of 2.  The zero-padding
# for a_n is done simply by adding zeros at the end, but since the index k - n can be negative
# and b_{-n} = b_n, the padding has to be done differently.  Since k - n can take on 2N - 1
# values, it is necessary to make the new arrays a length N' >= 2N - 1.  The new arrays are
#
#	 |--
#	 | a_n,		0 <= n < N
# A_n = -|
#	 | 0,		N <= n < N'
#	 |--
#
#	 |--
#	 | b_n,		0 <= n < N
# B_n = -| 0,		N <= n <= N' - N
#	 | b_{N'-n},	N' - N <= n < N'
#	 |--
#
# The convolution of A_n and B_n can be evaluated using the convolution theorem and the
# Cooley-Tukey FFT algorithm:
# X_k = conj(b_k) * ifft(fft(A_n) * fft(B_n))[:N]

def prime_fft(td_data, return_double = False, inverse = False, exp_array2 = None, M = None, prime_factors = None, exp_array = None):

	N = len(td_data)

	# Find the array of exponentials.
	if exp_array2 is None:
		exp_array2 = find_exp_array2(N, inverse = inverse)

	# Find the sequences we need, padding with zeros as necessary.
	if M is None or prime_factors is None:
		M, prime_factors = find_M(2 * N - 1)
	if exp_array is None:
		exp_array = find_exp_array(M)
	A_n = np.concatenate((td_data * exp_array2, np.zeros(M - N, dtype = np.complex256)))
	b_n = np.conj(exp_array2)
	B_n = np.concatenate((b_n, np.zeros(M - 2 * N + 1, dtype = np.complex256), b_n[1:][::-1]))

	# Do the convolution using the convolution theorem and the Cooley-Tukey algorithm, and
	# multiply by exp_array2.
	long_data = ifft(fft(A_n, prime_factors = prime_factors, exp_array = exp_array) * fft(B_n, prime_factors = prime_factors, exp_array = exp_array), prime_factors = prime_factors, exp_array = np.conj(exp_array))
	fd_data = exp_array2 * long_data[:N]

	if return_double:
		return np.complex128(fd_data)
	else:
		return fd_data


# If the input is real, the output is conjugate-symmetric: fd_data[n] = conj(fd_data[N - n]).
# We can reduce the number of operations by a factor of ~2.  Also, we have the option to only
# output half of the result, since the second half is redundant.
def prime_rfft(td_data, return_double = False, return_full = False, exp_array2 = None, M = None, prime_factors = None, exp_array = None):

	N = len(td_data)
	N_out = N // 2 + 1

	# Find the array of exponentials.
	if exp_array2 is None:
		exp_array2 = find_exp_array2(N)

	# Find the sequences we need, padding with zeros as necessary.
	if M is None or prime_factors is None:
		M, prime_factors = find_M(N + N_out - 1)
	if exp_array is None:
		exp_array = find_exp_array(M)
	A_n = np.concatenate((td_data * exp_array2, np.zeros(M - N, dtype = np.complex256)))
	b_n = np.conj(exp_array2)
	B_n = np.concatenate((b_n[:N_out], np.zeros(M - N - N_out + 1, dtype = np.complex256), b_n[1:][::-1]))

	# Do the convolution using the convolution theorem and the Cooley-Tukey algorithm, and
	# multiply by exp_array2.
	long_data = ifft(fft(A_n, prime_factors = prime_factors, exp_array = exp_array) * fft(B_n, prime_factors = prime_factors, exp_array = exp_array), prime_factors = prime_factors, exp_array = np.conj(exp_array))

	fd_data = exp_array2[:N_out] * long_data[:N_out]
	if return_full:
		fd_data = np.concatenate((fd_data[:N_out], np.conj(fd_data[1:N-N_out+1][::-1])))

	if return_double:
		return np.complex128(fd_data)
	else:
		return fd_data


# Inverse of the above real-input FFT.  So the output of this is real and the input is assumed
# to be shortened to N // 2 + 1 samples to avoid redundancy.
def prime_irfft(fd_data, return_double = False, N = None, normalize = True, exp_array2 = None, M = None, prime_factors = None, exp_array = None):

	N_in = len(fd_data)

	if N is None:
		# Find N, the original number of samples. If the imaginary part of the last
		# sample is zero, assume N was even
		if np.imag(fd_data[-1]) == 0:
			N = (N_in - 1) * 2
		elif np.real(fd_data[-1]) == 0:
			N = N_in * 2 - 1
		elif abs(np.imag(fd_data[-1]) / np.real(fd_data[-1])) < 1e-14:
			N = (N_in - 1) * 2
		else:
			N = N_in * 2 - 1

	# Find the array of exponentials.
	if exp_array2 is None:
		exp_array2 = find_exp_array2(N, inverse = True)

	# Find the sequences we need, padding with zeros as necessary.
	if M is None or prime_factors is None:
		M, prime_factors = find_M(2 * N - 1)
	if exp_array is None:
		exp_array = find_exp_array(M)
	A_n = np.concatenate((fd_data * exp_array2[:N_in], np.conj(fd_data[1:N-N_in+1][::-1]) * exp_array2[N_in:], np.zeros(M - N, dtype = np.complex256)))
	b_n = np.conj(exp_array2)
	B_n = np.concatenate((b_n, np.zeros(M - 2 * N + 1, dtype = np.complex256), b_n[1:][::-1]))

	# Do the convolution using the convolution theorem and the Cooley-Tukey algorithm, and
	# multiply by exp_array2.
	long_data = ifft(fft(A_n, prime_factors = prime_factors, exp_array = exp_array) * fft(B_n, prime_factors = prime_factors, exp_array = exp_array), prime_factors = prime_factors, exp_array = np.conj(exp_array))
	td_data = np.real(exp_array2 * long_data[:N])

	if normalize:
		if return_double:
			return np.float64(td_data / N)
		else:
			return td_data / N
	else:
		if return_double:
			return np.float64(td_data)
		else:
			return td_data


def compare_speed(length, iterations = 100, numerator = 'prime_fft', denominator = 'dft'):

	mysetup = 'import numpy as np\nimport longDoubleFFT as fft'

	num_command = "fft.%s(np.random.rand(%d))" % (numerator, length)
	denom_command = "fft.%s(np.random.rand(%d))" % (denominator, length)

	num = timeit.timeit(setup = mysetup, stmt = num_command, number = iterations)
	denom = timeit.timeit(setup = mysetup, stmt = denom_command, number = iterations)

	return num / denom


def test(a):
	z = np.random.rand((a + 2) // 2) + 1j * np.random.rand((a + 2) // 2)
	z[0] = np.real(z[0])
	if not a % 2:
		z[-1] = np.real(z[-1])
	dog = irfft(z)
	cat = prime_irfft(z)
	if not a % 2:
		mouse = np.fft.irfft(z)
		return cat / dog + mouse / dog - 2
	else:
		return cat / dog - 1


def test2(a):
	z = np.random.rand(a) + 1j * np.random.rand(a)
	dog = ifft(z)
	cat = prime_fft(z, inverse = True) / a
	mouse = np.fft.ifft(z)
	return cat / dog + mouse / dog - 2


def test3(a):
	z = np.random.rand(a)
	dog = rfft(z)
	cat = prime_rfft(z)
	mouse = np.fft.rfft(z)
	return cat / dog + mouse / dog - 2


def test4(a):
	z = np.random.rand(a)
	dog = fft(z)
	cat = prime_fft(z)
	mouse = np.fft.fft(z)
	return cat / dog + mouse / dog - 2


def test_many(start = 2, end = 100):
	for i in range(start, end):
		if max(abs(test(i))) > 1e-7:
			print("FAIL on test(%d).  Off by %e" % (i, max(abs(test(i) - 1))))
			print("prime factors are %s" % find_prime_factors(i))


def test_many2(start = 2, end = 100):
	for i in range(start, end):
		if max(abs(test2(i))) > 1e-7:
			print("FAIL on test2(%d).  Off by %e" % (i, max(abs(test2(i) - 1))))
			print("prime factors are %s" % find_prime_factors(i))


def test_many3(start = 2, end = 100):
	for i in range(start, end):
		if max(abs(test3(i))) > 1e-7:
			print("FAIL on test3(%d).  Off by %e" % (i, max(abs(test3(i) - 1))))
			print("prime factors are %s" % find_prime_factors(i))


def test_many4(start = 2, end = 100):
	for i in range(start, end):
		if max(abs(test4(i))) > 1e-7:
			print("FAIL on test4(%d).  Off by %e" % (i, max(abs(test4(i) - 1))))
			print("prime factors are %s" % find_prime_factors(i))


factorials_inv = np.zeros(1755, dtype = np.float128)
factorials_inv[0] = 1.0
current = 1
for n in range(1, 1755):
	current *= n
	factorials_inv[n] = 1.0 / np.float128(str(current))


#
# Below are several useful window functions, all to long double precision.
#


#
# DPSS window
#


# Compute a discrete prolate spheroidal sequence (DPSS) window,
# which maximizes the energy concentration in the central lobe

# A function to multiply a symmetric Toeplitz matrix times a vector and normalize.
# Assume that only the first row of the matrix is stored, to save memory.
# Assume that only half of the vector is stored, due to symmetry.
def mat_times_vec(mat, vec):
	N = len(mat)
	n = len(vec)
	outvec = np.zeros(n, dtype = type(mat[0]))
	for i in range(n):
		reordered_mat = np.copy(mat[N-1-i:N-1-n-i:-1])
		reordered_mat[i:N-n] += mat[:N-n-i]
		reordered_mat[:i] += mat[i:0:-1]
		outvec[i] = np.matmul(vec, reordered_mat)
	# Normalize
	return outvec / outvec[-1]


def DPSS(N, alpha, return_double = False, max_time = 10):

	N = int(N)

	# Estimate how long each process should take.  This is based on data taken from a Macbook
	# Pro purchased in 2016.
	seconds_per_iteration_double = 4.775e-10 * N * N + 1.858e-6 * N
	seconds_per_iteration_longdouble = 1.296e-9 * N * N + 5.064e-6 * N

	double_iterations = int(max_time / 2.0 / seconds_per_iteration_double)
	longdouble_iterations = int(max_time / 2.0 / seconds_per_iteration_longdouble)

	# Start with ordinary double precision to make it run faster.
	# Angular cutoff frequency times sample period
	omega_c_Ts = np.float64(two_pi * alpha / N)

	# The DPSS window is the eigenvector associated with the largest eigenvalue of the symmetric
	# Toeplitz matrix (Toeplitz means all elements along negative sloping diagonals are equal),
	# where the zeroth column and row are the sampled sinc function below:
	sinc = np.zeros(N)
	sinc[0] = omega_c_Ts
	for i in range(1, N):
		sinc[i] = np.sin(omega_c_Ts * i) / i


	# Start by approximating the DPSS window with a Kaiser window with the same value of alpha.
	# Note that kaiser() takes beta = pi * alpha as an argument.  Due to symmetry, we need to
	# store only half of the window.
	dpss = np.kaiser(N, pi * alpha)[: N // 2 + N % 2]

	# Now use power iteration to get our approximation closer to the true DPSS window.  This
	# entails simply applying the eigenvalue equation over and over until we are satisfied
	# with the accuracy of the eigenvector.  This method assumes the existance of an eigenvalue
	# that is larger in magnitude than all the other eigenvalues.

	# Compute an estimate of the error: how much the window changes during each iteration.
	# We will compare this to how much it changes in each iteration at the end as an
	# indicator of how much the window improved over the original Kaiser window.
	new_dpss = mat_times_vec(sinc, dpss)
	first_error = sum(pow(dpss - new_dpss, 2))

	for i in range(double_iterations):
		new_dpss = mat_times_vec(sinc, new_dpss)

	# Now do this with extra precision
	omega_c_Ts = two_pi * alpha / N
	sinc = np.zeros(N, dtype = np.float128)
	sinc[0] = omega_c_Ts
	for i in range(1, N):
		sinc[i] = np.sin(omega_c_Ts * i) / i

	for i in range(longdouble_iterations):
		new_dpss = mat_times_vec(sinc, new_dpss)

	dpss = new_dpss
	new_dpss = mat_times_vec(sinc, dpss)
	last_error = sum(pow(dpss - new_dpss, 2))
	dpss = new_dpss

	print("After %d iterations, the RMS error of the DPSS window is approximately %e of what it was originally." % (2 + double_iterations + longdouble_iterations, np.sqrt(last_error / first_error)))

	if return_double:
		dpss = np.float64(dpss)
	return np.concatenate((dpss, dpss[::-1][N%2:]))


#
# Kaiser window, an approximation to the DPSS window
#

# Modified Bessel function of the first kind
def I0(x):
	out = 1.0
	m = 1
	resolution = np.finfo(x).resolution / 20
	next_term = pow((x / 2), 2 * m) * factorials_inv[m] * factorials_inv[m]
	while next_term / out > resolution:
		out += next_term
		m += 1
		next_term = pow((x / 2), 2 * m) * factorials_inv[m] * factorials_inv[m]
		# Improve accuracy a little by doing this:
		#for i in range(m):
		#       m += 1
		#       next_term += pow((x / 2), 2 * m) / np.float128(str(pow(math.factorial(m), 2)))
	return out


def kaiser(N, beta, return_double = False):
	N = int(N)
	beta = np.float128(beta)
	two = np.float128(2)
	win = np.zeros(N, dtype = np.float128)
	denom = I0(beta)
	for n in range(N):
		win[n] = I0(beta * np.sqrt(1.0 - pow(two * n / (N - 1) - 1, 2))) / denom
	if return_double:
		return np.float64(win)
	else:
		return win


#
# Dolph-Chebyshev window
#


def compute_Tn(x, n):
	if x < -1:
		if n % 2:
			return -np.cosh(n * np.arccosh(-x))
		else:
			return np.cosh(n * np.arccosh(-x))
	elif x <= 1:
		return np.cos(n * np.arccos(x))
	else:
		return np.cosh(n * np.arccosh(x))


def compute_W0_lagged(N, alpha):

	n = N // 2 + 1
	beta = np.cosh(np.arccosh(pow(10.0, alpha)) / (N - 1))

	W0 = np.zeros(n, dtype = np.complex256)
	denominator = pow(10.0, alpha)
	factor = -pi * 1j * (N - 1) / N
	for k in range(0, n):
		W0[k] = np.exp(factor * k) * compute_Tn(beta * np.cos(pi * k / N), N - 1) / denominator

	# If we want an even window length, the Nyquist component must be real.
	if not N % 2:
		W0[-1] = np.abs(W0[-1])

	return W0


def DolphChebyshev(N, alpha, return_double = False):

	N = int(N)
	win = irfft(compute_W0_lagged(N, alpha), return_double = return_double)
	return win / max(win)


#
# A resampler
#


def resample(data, N_out, return_double = False):

	N_out = int(N_out)

	# Number of input samples
	N_in = len(data)

	# Max and min
	N_max = max(N_in, N_out)
	N_min = min(N_in, N_out)

	if N_in < 2 or N_in == N_out:
		return data

	# Is the input data complex?  If so, the output should be as well.
	is_complex = isinstance(data[0], complex)

	if N_out == 0:
		if is_complex:
			if return_double:
				return np.array([], dtype = np.complex128)
			else:
				return np.array([], dtype = np.complex256)
		else:
			if return_double:
				return np.array([], dtype = np.float64)
			else:
				return np.array([], dtype = np.float128)

	if N_out == 1:
		# Return the average
		return np.array([sum(data) / N_in])

	if is_complex:
		resampled = np.zeros(N_out, dtype = np.complex256)
	else:
		resampled = np.zeros(N_out, dtype = np.float128)

	if N_in == 2:
		# Linear interpolation.  If we've reached this point, we know that N_out >= 3.
		if is_complex:
			diff = np.complex256(data[1]) - np.complex256(data[0])
		else:
			diff = np.float128(data[1]) - np.float128(data[0])
		resampled = diff * np.array(range(N_out)) / (N_out - 1) + data[0]

	else:
		# Are we upsampling or downsampling?
		upordown = 'up' if N_in < N_out else 'down'

		# Find the least common multiple of input and output lengths to determine the
		# lenth of the sinc array.
		short_length = N_min - 1
		long_length = N_max - 1
		LCM = long_length
		while LCM % short_length:
			LCM += long_length

		# Number of sinc taps per sample at the higher sample rate
		sinc_taps_per_sample = LCM // long_length
		# Number of sinc taps per sample at the lower sample rate
		long_sinc_taps_per_sample = LCM // short_length

		sinc_length = min(N_min, 192) * long_sinc_taps_per_sample
		sinc_length -= (sinc_length + 1) % 2
		sinc = np.zeros(sinc_length, dtype = np.float128)
		sinc[sinc_length // 2] = 1.0

		# Frequency resolution in units of frequency bins of sinc
		alpha = (1 + min(192, N_min) / 24.0)
		# Low-pass cutoff frequency as a fraction of the sampling frequency of sinc
		f_cut = 0.5 / long_sinc_taps_per_sample - alpha / sinc_length
		if f_cut > 0:
			for i in range(1, sinc_length // 2 + 1):
				sinc[sinc_length // 2 + i] = sinc[sinc_length // 2 - i] = np.sin(two_pi * ((f_cut * i) % 1)) / (two_pi * f_cut * i)
		else:
			sinc = np.ones(sinc_length, dtype = np.float128)

		# Apply a Kaiser window.  Note that the chosen cutoff frequency is below the
		# lower Nyquist rate just enough to be at the end of the main lobe.
		sinc *= kaiser(sinc_length, pi * alpha)

		# Normalize the sinc filter.  Since, in general, not every tap gets used for
		# each output sample, the normalization has to be done this way:
		if upordown == 'down':
			taps_per_input = sinc_taps_per_sample
			taps_per_output = long_sinc_taps_per_sample
		else:
			taps_per_input = long_sinc_taps_per_sample
			taps_per_output = sinc_taps_per_sample

		for i in range(taps_per_input):
			sinc[i::taps_per_input] /= np.sum(sinc[i::taps_per_input])

		# Extend the input array at the ends to prepare for filtering
		half_sinc_length = sinc_length // 2
		N_ext = half_sinc_length // taps_per_input
		data = np.concatenate((-data[1:1+N_ext][::-1] + 2 * np.real(data[0]), data, -data[-N_ext-1:-1][::-1] + 2 * np.real(data[-1])))

		# Filter.  The center of sinc should line up with the first and last input
		# at the first and last output, respectively.
		for i in range(N_out):
			sinc_start = (half_sinc_length - i * taps_per_output % taps_per_input) % taps_per_input
			data_start = (i * taps_per_output - half_sinc_length + N_ext * taps_per_input + taps_per_input - 1) // taps_per_input
			sinc_subset = sinc[sinc_start::taps_per_input]
			resampled[i] = np.sum(sinc_subset * data[data_start:data_start+len(sinc_subset)])

	if return_double:
		if is_complex:
			return np.complex128(resampled)
		else:
			return np.float64(resampled)
	else:
		return resampled


#
# A function to get the frequency-responce of an FIR filter, showing the lobes
#


def freqresp(filt, delay_samples = 0, samples_per_lobe = 8, return_double = False):

	N = len(filt)

	if N == 0:
		return np.array([])

	# In case the user gives invalid inputs
	delay_samples = int(round(delay_samples)) % N
	samples_per_lobe = int(round(abs(samples_per_lobe)))
	if samples_per_lobe == 0:
		samples_per_lobe += 1

	# Make a longer version of the filter so that we can
	# get better frequency resolution.
	N_prime = samples_per_lobe * N
	# Start with zeros
	filt_prime = np.zeros(N_prime, dtype = np.float128)
	# The beginning and end have filter coefficients
	filt_prime[: N - delay_samples] = np.float128(filt[delay_samples:])
	if delay_samples > 0:
		filt_prime[-delay_samples:] = np.float128(filt[:delay_samples])

	# Now take an FFT
	return rfft(filt_prime, return_double = return_double)


