/**
 * onlinehoftsrc.c
 * Implementation for libonlinehoftsrc
 *
 * Copyright (C) 2008 Leo Singer
 *
 * This library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */


/* enable safe printf routines */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dirent.h>
#include <FrameL.h>
#include <FrVect.h>
#include <lal/Date.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>

#include "onlinehoft.h"


typedef struct {
	char* ifo; // e.g. H1
	char* nameprefix; // e.g. "H-H1_DMT_C00_L2-"
	char* namesuffix; // e.g. "-16.gwf"
	char* channelname; // e.g. "H1:DMT-STRAIN"
} _ifodesc_t;


struct onlinehoft_tracker {
	int was_discontinuous;
	uint64_t lastReadGpsRemainder;
	uint64_t gpsRemainder;
	uint32_t minLatency;
	char* dirprefix; // e.g. "/archive/frames/online/hoft"
	uint8_t state_require, state_deny, dq_require, dq_deny;
	const _ifodesc_t* ifodesc;
};


static const _ifodesc_t _ifodescs[] =
{
	{"H1", "H-H1_DMT_C00_L2-", "-16.gwf", "H1:DMT-STRAIN"},
	{"H2", "H-H2_DMT_C00_L2-", "-16.gwf", "H2:DMT-STRAIN"},
	{"L1", "L-L1_DMT_C00_L2-", "-16.gwf", "L1:DMT-STRAIN"},
	/* sorry, we don't have any specs about VIRGO data quality flags yet! */
	/*{"V1", "V-V1_DMT_HREC-",   "-16.gwf", "V1:h_16384Hz"},*/
	{NULL, NULL, NULL, NULL}
};


static int _onlinehoft_uint64fromstring(const char* const begin, const char* const end, uint64_t* result)
{
	const char* ptr;
	*result = 0;
	for (ptr = begin; ptr < end; ptr++)
	{
		if (!isdigit(*ptr))
			return 0;
		*result *= 10;
		*result += *ptr - '0';
	}
	return 1;
}


static int _onlinehoft_uint16fromstring2(const char* const begin, uint16_t* result)
{
	const char* ptr;
	*result = 0;
	for (ptr = begin; *ptr != '\0'; ptr++)
	{
		if (!isdigit(*ptr))
			return 0;
		*result *= 10;
		*result += *ptr - '0';
	}
	return 1;
}


static uint64_t _onlinehoft_poll_era(onlinehoft_tracker_t* tracker, uint16_t era)
{
	// Compute the name of the directory for the given era.
	char* dirname = NULL;
	asprintf(&dirname, "%s/%s%u", tracker->dirprefix, tracker->ifodesc->nameprefix, era);

	// This should never happen, unless we run out of memory.
	if (!dirname)
		return 0;

	// Try to open the directory listing.
	DIR* dirp = opendir(dirname);
	free(dirname);
	if (!dirp)
		return 0;

	uint64_t gpsRemainder = UINT64_MAX;

	struct dirent* dp;
	errno = 0;
	size_t nameprefix_len = strlen(tracker->ifodesc->nameprefix);
	size_t namesuffix_len = strlen(tracker->ifodesc->namesuffix);
	while ((dp = readdir(dirp)))
	{
		// check to see if the current directory entry starts with the nameprefix
		if (strncmp(tracker->ifodesc->nameprefix, dp->d_name, nameprefix_len))
			continue;

		// Cut off name prefix, leaving stem (e.g. "955484688-16.gwf")
		const char* const namestem = &dp->d_name[nameprefix_len];

		// Find first delimiter '-' in string, which terminates GPS time
		const char* const delimptr = strchr(namestem, '-');

		// If delimiter not found, go to next entry
		if (!delimptr) continue;

		// If rest of name does not match namesuffix, go to next entry
		if (strncmp(tracker->ifodesc->namesuffix, delimptr, namesuffix_len)) continue;

		// Try to extract GPS time from name stem, or go to enxt entry
		uint64_t gpsTime;
		if (!_onlinehoft_uint64fromstring(namestem, delimptr, &gpsTime))
			continue;

		uint64_t newGpsRemainder = gpsTime >> 4;

		// If GPS time is older than current gpsRemainder, go to next entry
		if (newGpsRemainder < tracker->gpsRemainder)
			continue;

		if (newGpsRemainder < gpsRemainder)
			gpsRemainder = newGpsRemainder;
	}

	closedir(dirp);

	if (gpsRemainder == UINT64_MAX)
		return 0;
	else
		return gpsRemainder;
}


static uint16_t _onlinehoft_era_for_remainder(uint64_t remainder)
{
	return (remainder << 4) / 100000;
}


static uint64_t _onlinehoft_poll(onlinehoft_tracker_t* tracker)
{
	// Try to open the directory listing.
	DIR* dirp = opendir(tracker->dirprefix);
	if (!dirp)
	{
		fprintf(stderr, "_onlinehoft_poll:openddir(\"%s\"):%s\n", tracker->dirprefix, strerror(errno));
		return 0;
	}

	uint16_t currentEra = _onlinehoft_era_for_remainder(tracker->gpsRemainder);
	uint16_t earliestEra = UINT16_MAX;
	uint16_t latestEra = 0;

	struct dirent* dp;
	size_t nameprefix_len = strlen(tracker->ifodesc->nameprefix);
	while ((dp = readdir(dirp)))
	{
		if (strncmp(tracker->ifodesc->nameprefix, dp->d_name, nameprefix_len))
			continue;

		const char* namestem = &dp->d_name[nameprefix_len];

		uint16_t newEra;
		if (!_onlinehoft_uint16fromstring2(namestem, &newEra))
			continue;

		if (newEra < currentEra)
			continue;

		if (newEra < earliestEra)
			earliestEra = newEra;

		if (newEra > latestEra)
			latestEra = newEra;
	}

	closedir(dirp);

	if (earliestEra > latestEra)
		return 0;

	uint16_t era;
	for (era = earliestEra; era < latestEra; era++)
	{
		uint64_t newRemainder = _onlinehoft_poll_era(tracker, era);
		if (newRemainder)
			return newRemainder;
	}

	return 0;
}


static const _ifodesc_t* _onlinehoft_find(const char* ifo)
{
	if (!ifo)
		return NULL;

	const _ifodesc_t* orig;
	for (orig = _ifodescs; orig->ifo; orig++)
		if (!strcmp(orig->ifo, ifo))
			return orig;

	return NULL;
}


onlinehoft_tracker_t* onlinehoft_create(const char* ifo)
{
	const _ifodesc_t* orig = _onlinehoft_find(ifo);
	if (!orig) return NULL;

	char* onlinehoftdir = getenv("ONLINEHOFT");
	if (!onlinehoftdir) return NULL;

	onlinehoft_tracker_t* tracker = calloc(1, sizeof(onlinehoft_tracker_t));
	if (!tracker) return NULL;

	if (asprintf(&tracker->dirprefix, "%s/%s", onlinehoftdir, ifo) < 1)
	{
		free(tracker);
		return NULL;
	}

	LIGOTimeGPS time_now;
	if (XLALGPSTimeNow(&time_now))
		tracker->gpsRemainder = ((time_now.gpsSeconds - tracker->minLatency) >> 4);

	tracker->was_discontinuous = 1;
	tracker->lastReadGpsRemainder = 0;
	tracker->minLatency = 90;
	tracker->ifodesc = orig;
	tracker->state_require = ONLINEHOFT_STATE_DEFAULT_REQUIRE;
	tracker->state_deny = ONLINEHOFT_STATE_DEFAULT_DENY;
	tracker->dq_require = ONLINEHOFT_DQ_DEFAULT_REQUIRE;
	tracker->dq_deny = ONLINEHOFT_DQ_DEFAULT_DENY;

	return tracker;
}


void onlinehoft_set_masks(onlinehoft_tracker_t* tracker,
	uint8_t state_require, uint8_t state_deny, uint8_t dq_require, uint8_t dq_deny)
{
	tracker->state_require = state_require;
	tracker->state_deny = state_deny;
	tracker->dq_require = dq_require;
	tracker->dq_deny = dq_deny;
}


void onlinehoft_destroy(onlinehoft_tracker_t* tracker)
{
	if (tracker)
	{
		free(tracker->dirprefix);
		tracker->dirprefix = NULL;
		free(tracker);
	}
}


static FrFile* _onlinehoft_next_file(onlinehoft_tracker_t* tracker)
{
	FrFile* frFile;

	do {
		LIGOTimeGPS time_now;
		while (XLALGPSTimeNow(&time_now)
			&& (tracker->gpsRemainder << 4) + tracker->minLatency > (uint64_t)time_now.gpsSeconds)
			sleep(1);

		char* filename;
		if (asprintf(&filename, "%s/%s%u/%s%" PRIu64 "%s",
				 tracker->dirprefix, tracker->ifodesc->nameprefix,
				 _onlinehoft_era_for_remainder(tracker->gpsRemainder),
				 tracker->ifodesc->nameprefix, tracker->gpsRemainder << 4, tracker->ifodesc->namesuffix) < 1)
		return NULL;

		frFile = NULL;
		FILE* fp = fopen(filename, "r");
		if (fp)
		{
			++tracker->gpsRemainder;
			frFile = FrFileINew(filename);
			fclose(fp);
		} else if (errno != ENOENT) {
			++tracker->gpsRemainder;
		} else {
			uint64_t newRemainder;
			while (!(newRemainder = _onlinehoft_poll(tracker)))
				sleep(8);
			// Poll directory again because the frame builder could have been
			// adding files while we were polling the directory.  This second
			// poll eliminates the race condition, assuming frames are created
			// sequentially.
			while (!(newRemainder = _onlinehoft_poll(tracker)))
				sleep(8);
			tracker->gpsRemainder = newRemainder;
		}
		free(filename);
	} while (!frFile);

	return frFile;
}


uint64_t onlinehoft_seek(onlinehoft_tracker_t* tracker, uint64_t gpsSeconds)
{
	if (!tracker) return 0;

	tracker->gpsRemainder = gpsSeconds >> 4;
	return (tracker->gpsRemainder) << 4;
}


FrVect* onlinehoft_next_vect(onlinehoft_tracker_t* tracker)
{
	if (!tracker) return NULL;

	FrFile* frFile = _onlinehoft_next_file(tracker);
	if (!frFile) return NULL;
	FrVect* vect = FrFileIGetVect(frFile, tracker->ifodesc->channelname, ((tracker->gpsRemainder-1) << 4), 16);
	FrFileIEnd(frFile);

	// If FrFileIGetVect failed, return NULL
	if (!vect) return vect;

	// If GPS start time is wrong, return NULL
	{
		uint64_t expected_gps_start = (tracker->gpsRemainder-1) << 4;
		uint64_t retrieved_gps_start = (uint64_t)vect->GTime;
		if (expected_gps_start != retrieved_gps_start)
		{
			FrVectFree(vect);
			fprintf(stderr, "onlinehoft_next_vect: expected timestamp %lu, but got %lu\n",
					expected_gps_start, retrieved_gps_start);
			return NULL;
		}
	}

	// If duration is wrong, return NULL
	{
		uint32_t expected_nsamples = 16 * 16384;
		uint32_t retrieved_nsamples = vect->nx[0];
		if (expected_nsamples != retrieved_nsamples)
		{
			FrVectFree(vect);
			fprintf(stderr, "onlinehoft_next_vect: expected %u samples, but got %u\n",
					expected_nsamples, retrieved_nsamples);
			return NULL;
		}
	}

	tracker->was_discontinuous = (tracker->gpsRemainder != (tracker->lastReadGpsRemainder+1));
	tracker->lastReadGpsRemainder = tracker->gpsRemainder;
	return vect;
}


const char* onlinehoft_get_channelname(const onlinehoft_tracker_t* tracker)
{
	return &tracker->ifodesc->channelname[3];
}


int onlinehoft_was_discontinuous(const onlinehoft_tracker_t* tracker)
{
	return tracker->was_discontinuous;
}
