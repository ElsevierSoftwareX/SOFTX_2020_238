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

typedef struct onlinehoft_tracker_t onlinehoft_tracker_t;

onlinehoft_tracker_t* onlinehoft_create(const char* ifo);
uint32_t onlinehoft_seek(onlinehoft_tracker_t* tracker, uint32_t gpsSeconds);
void onlinehoft_destroy(onlinehoft_tracker_t* tracker);
FrVect* onlinehoft_next_vect(onlinehoft_tracker_t* tracker);

#endif
