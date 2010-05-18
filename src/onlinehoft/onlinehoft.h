/**
 * onlinehoft.h
 * Interface for libonlinehoftsrc, an online LIGO/VIRGO frame access library
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

#include <FrameL.h>
#include <FrVect.h>
#include <stdint.h>

#ifndef _ONLINEHOFT_H
#define _ONLINEHOFT_H

struct onlinehoft_tracker;
typedef struct onlinehoft_tracker onlinehoft_tracker_t;


// State vector and data quality flags, described on
// https://www.lsc-group.phys.uwm.edu/daswg/wiki/S6OnlineGroup/CalibratedData


// State vector flags
#define ONLINEHOFT_STATE_SCI     ((uint8_t) 0x01) // operator set to go to science mode
#define ONLINEHOFT_STATE_CON     ((uint8_t) 0x02) // conlog unsets this bit is non-harmless epics changes
#define ONLINEHOFT_STATE_UP      ((uint8_t) 0x04) // set by locking scripts
#define ONLINEHOFT_STATE_NOTINJ  ((uint8_t) 0x08) // injections unset this bit
#define ONLINEHOFT_STATE_EXC     ((uint8_t) 0x10) // unauthorized excitations cause this bit to be unset

// By default, require all flags but !INJ to be on
#define ONLINEHOFT_STATE_DEFAULT_REQUIRE (ONLINEHOFT_STATE_SCI | ONLINEHOFT_STATE_CON | ONLINEHOFT_STATE_UP | ONLINEHOFT_STATE_EXC)
#define ONLINEHOFT_STATE_DEFAULT_DENY ((uint8_t) 0)

// Data quality flags
#define ONLINEHOFT_DQ_SCIENCE    ((uint8_t) 0x01) // SV_SCIENCE & LIGHT
#define ONLINEHOFT_DQ_INJECTION  ((uint8_t) 0x02) // Injection: same as statevector
#define ONLINEHOFT_DQ_UP         ((uint8_t) 0x04) // Injection: same as statevector
#define ONLINEHOFT_DQ_CALIBRATED ((uint8_t) 0x08) // SV_UP & LIGHT & (not TRANSIENT)
#define ONLINEHOFT_DQ_BADGAMMA   ((uint8_t) 0x10) // Calibration is bad (outside 0.8 < gamma < 1.2)
#define ONLINEHOFT_DQ_LIGHT      ((uint8_t) 0x20) // Light in the arms ok
#define ONLINEHOFT_DQ_MISSING    ((uint8_t) 0x30) // Indication that data was dropped in DMT (currently not implemented)

#define ONLINEHOFT_DQ_DEFAULT_REQUIRE ( ONLINEHOFT_DQ_SCIENCE | ONLINEHOFT_DQ_UP | ONLINEHOFT_DQ_CALIBRATED | ONLINEHOFT_DQ_LIGHT)

#define ONLINEHOFT_DQ_DEFAULT_DENY (ONLINEHOFT_DQ_BADGAMMA | ONLINEHOFT_DQ_MISSING)


onlinehoft_tracker_t* onlinehoft_create(const char* ifo);
void onlinehoft_set_masks(onlinehoft_tracker_t* tracker,
	uint8_t state_require, uint8_t state_deny, uint8_t dq_require, uint8_t dq_deny);
uint64_t onlinehoft_seek(onlinehoft_tracker_t* tracker, uint64_t gpsSeconds);
void onlinehoft_destroy(onlinehoft_tracker_t* tracker);
FrVect* onlinehoft_next_vect(onlinehoft_tracker_t* tracker);
const char* onlinehoft_get_channelname(const onlinehoft_tracker_t* tracker);
int onlinehoft_was_discontinuous(const onlinehoft_tracker_t* tracker);

#endif
