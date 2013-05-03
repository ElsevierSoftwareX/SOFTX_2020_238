/*
 * Copyright (C) 2008--2013  Kipp Cannon, Chad Hanna
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from the C library
 */


#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * stuff from glib/gobject
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * stuff from FFTW
 */


#include <fftw3.h>


/*
 * stuff from LAL
 */


#ifdef HAVE_LAL_FFTWMUTEX_H
#include <lal/FFTWMutex.h>
#endif


/*
 * ============================================================================
 *
 *                            FFTW Wisdom Protection
 *
 * ============================================================================
 */


/*
 * FIXME:  we don't *really* need both locks, but we can't rely on LAL to
 * actually have a lock behind the macros, so we implement our own as well
 * until LAL can be trusted to lock fftw
 */

static GStaticMutex gstlal_fftw_lock_mutex = G_STATIC_MUTEX_INIT;


void gstlal_fftw_lock(void)
{
#ifdef LAL_FFTW_PTHREAD_MUTEX_LOCK
	LAL_FFTW_PTHREAD_MUTEX_LOCK;
#endif
	g_static_mutex_lock(&gstlal_fftw_lock_mutex);
}


void gstlal_fftw_unlock(void)
{
	g_static_mutex_unlock(&gstlal_fftw_lock_mutex);
#ifdef LAL_FFTW_PTHREAD_MUTEX_UNLOCK
	LAL_FFTW_PTHREAD_MUTEX_UNLOCK;
#endif
}


/*
 * ============================================================================
 *
 *                             FFTW Wisdom Import
 *
 * ============================================================================
 */


void gstlal_load_fftw_wisdom(void)
{
	char *filename;
	int savederrno;

	gstlal_fftw_lock();

	/*
	 * double precision
	 */

	savederrno = errno;
	filename = getenv(GSTLAL_FFTW_WISDOM_ENV);
	if(filename) {
		FILE *f = fopen(filename, "r");
		if(!f)
			GST_ERROR("cannot open double-precision FFTW wisdom file \"%s\": %s", filename, strerror(errno));
		else {
			if(!fftw_import_wisdom_from_file(f))
				GST_ERROR("failed to import double-precision FFTW wisdom from \"%s\": wisdom not loaded", filename);
			fclose(f);
		}
	} else if(!fftw_import_system_wisdom())
		GST_WARNING("failed to import system default double-precision FFTW wisdom: %s", strerror(errno));
	errno = savederrno;

	/*
	 * single precision
	 */

	savederrno = errno;
	filename = getenv(GSTLAL_FFTWF_WISDOM_ENV);
	if(filename) {
		FILE *f = fopen(filename, "r");
		if(!f)
			GST_ERROR("cannot open single-precision FFTW wisdom file \"%s\": %s", filename, strerror(errno));
		else {
			if(!fftwf_import_wisdom_from_file(f))
				GST_ERROR("failed to import single-precision FFTW wisdom from \"%s\": wisdom not loaded", filename);
			fclose(f);
		}
	} else if(!fftwf_import_system_wisdom())
		GST_WARNING("failed to import system default single-precision FFTW wisdom: %s", strerror(errno));
	errno = savederrno;

	/*
	 * done
	 */

	gstlal_fftw_unlock();
}
