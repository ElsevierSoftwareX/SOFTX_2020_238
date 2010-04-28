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

#include <dirent.h>
#include <FrameL.h>
#include <FrVect.h>
#include <lal/Date.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>

typedef struct {
	int was_discontinuous;
	uint32_t lastReadGpsRemainder;
	uint32_t gpsRemainder;
	uint32_t minLatency;
	char* dirprefix; // e.g. "/archive/frames/online/hoft"
	char* ifo; // e.g. H1
	char* nameprefix; // e.g. "H-H1_DMT_C00_L2-"
	char* namesuffix; // e.g. "-16.gwf"
	char* channelname; // e.g. "H1:DMT-STRAIN"
} onlinehoft_tracker_t;


static const onlinehoft_tracker_t _onlinehoft_trackers[] =
{
	{1, 0, 0, 90, 0, "H1", "H-H1_DMT_C00_L2-", "-16.gwf", "H1:DMT-STRAIN"},
	{1, 0, 0, 90, 0, "H2", "H-H2_DMT_C00_L2-", "-16.gwf", "H2:DMT-STRAIN"},
	{1, 0, 0, 90, 0, "L1", "L-L1_DMT_C00_L2-", "-16.gwf", "L1:DMT-STRAIN"},
	{1, 0, 0, 90, 0, "V1", "V-V1_DMT_HREC-",   "-16.gwf", "V1:h_16384Hz"},
	{0, 0, 0, 0, 0, 0, 0, 0, 0}
};


static int _onlinehoft_uint32fromstring(const char* const begin, const char* const end, uint32_t* result)
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


static uint32_t _onlinehoft_poll_era(onlinehoft_tracker_t* tracker, uint16_t era)
{
	// Compute the name of the directory for the given era.
	char* dirname = NULL;
	asprintf(&dirname, "%s/%s%u", tracker->dirprefix, tracker->nameprefix, era);

	// This should never happen, unless we run out of memory.
	if (!dirname)
		return 0;

	// Try to open the directory listing.
	DIR* dirp = opendir(dirname);
	free(dirname);
	if (!dirp)
		return 0;

	uint32_t gpsRemainder = UINT32_MAX;

	struct dirent* dp;
	errno = 0;
	size_t nameprefix_len = strlen(tracker->nameprefix);
	size_t namesuffix_len = strlen(tracker->namesuffix);
	while (dp = readdir(dirp))
	{
		// check to see if the current directory entry starts with the nameprefix
		if (strncmp(tracker->nameprefix, dp->d_name, nameprefix_len))
			continue;

		// Cut off name prefix, leaving stem (e.g. "955484688-16.gwf")
		const char* const namestem = &dp->d_name[nameprefix_len];

		// Find first delimiter '-' in string, which terminates GPS time
		const char* const delimptr = strchr(namestem, '-');

		// If delimiter not found, go to next entry
		if (!delimptr) continue;

		// If rest of name does not match namesuffix, go to next entry
		if (strncmp(tracker->namesuffix, delimptr, namesuffix_len)) continue;

		// Try to extract GPS time from name stem, or go to enxt entry
		uint32_t gpsTime;
		if (!_onlinehoft_uint32fromstring(namestem, delimptr, &gpsTime))
			continue;

		uint32_t newGpsRemainder = gpsTime >> 4;

		// If GPS time is older than current gpsRemainder, go to next entry
		if (newGpsRemainder < tracker->gpsRemainder)
			continue;

		if (newGpsRemainder < gpsRemainder)
			gpsRemainder = newGpsRemainder;
	}

	closedir(dirp);

	if (gpsRemainder == UINT32_MAX)
		return 0;
	else
		return gpsRemainder;
}


static uint16_t _onlinehoft_era_for_remainder(uint32_t remainder)
{
	return (remainder << 4) / 100000;
}


static uint32_t _onlinehoft_poll(onlinehoft_tracker_t* tracker)
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
	size_t nameprefix_len = strlen(tracker->nameprefix);
	while (dp = readdir(dirp))
	{
		if (strncmp(tracker->nameprefix, dp->d_name, nameprefix_len))
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
		uint32_t newRemainder = _onlinehoft_poll_era(tracker, era);
		if (newRemainder)
			return newRemainder;
	}

	return 0;
}


static const onlinehoft_tracker_t* _onlinehoft_find(const char* ifo)
{
	if (!ifo)
		return NULL;

	const onlinehoft_tracker_t* orig;
	for (orig = _onlinehoft_trackers; orig->ifo; orig++)
		if (!strcmp(orig->ifo, ifo))
			return orig;

	return NULL;
}


onlinehoft_tracker_t* onlinehoft_create(const char* ifo)
{
	const onlinehoft_tracker_t* orig = _onlinehoft_find(ifo);
	if (!orig) return NULL;

	char* onlinehoftdir = getenv("ONLINEHOFT");
	if (!onlinehoftdir) return NULL;

	onlinehoft_tracker_t* tracker = calloc(1, sizeof(onlinehoft_tracker_t));
	if (!tracker) return NULL;

	memcpy(tracker, orig, sizeof(onlinehoft_tracker_t));

	if (asprintf(&tracker->dirprefix, "%s/%s", onlinehoftdir, ifo) < 1)
	{
		free(tracker);
		return NULL;
	}

	LIGOTimeGPS time_now;
	if (XLALGPSTimeNow(&time_now))
		tracker->gpsRemainder = ((time_now.gpsSeconds - tracker->minLatency) >> 4);

	return tracker;
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
			&& (tracker->gpsRemainder << 4) + tracker->minLatency > time_now.gpsSeconds)
			sleep(1);

		char* filename;
		if (asprintf(&filename, "%s/%s%u/%s%d%s",
				 tracker->dirprefix, tracker->nameprefix,
				 _onlinehoft_era_for_remainder(tracker->gpsRemainder),
				 tracker->nameprefix, tracker->gpsRemainder << 4, tracker->namesuffix) < 1)
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
			uint32_t newRemainder;
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


uint32_t onlinehoft_seek(onlinehoft_tracker_t* tracker, uint32_t gpsSeconds)
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
	FrVect* vect = FrFileIGetVect(frFile, tracker->channelname, ((tracker->gpsRemainder-1) << 4), 16);
	FrFileIEnd(frFile);

	// If FrFileIGetVect failed, return NULL
	if (!vect) return vect;

	// If GPS start time is wrong, return NULL
	{
		uint32_t expected_gps_start = tracker->gpsRemainder << 4;
		uint32_t retrieved_gps_start = (uint32_t)vect->GTime;
		if (expected_gps_start != retrieved_gps_start)
		{
			FrVectFree(vect);
			fprintf(stderr, "onlinehoft_next_vect: expected timestamp %d, but got %d",
					expected_gps_start, retrieved_gps_start);
			return NULL;
		}
	}

	// If duration is wrong, return NULL
	{
		uint32_t expected_nsamples = 16 * 16384;
		uint32_t retrieved_nsamples = vect->nData[0];
		if (expected_nsamples != retrieved_nsamples)
		{
			FrVectFree(vect);
			fprintf(stderr, "onlinehoft_next_vect: expected %d samples, but got %d",
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
	return &tracker->channelname[3];
}


int onlinehoft_was_discontinuous(const onlinehoft_tracker_t* tracker)
{
	return tracker->was_discontinuous;
}
