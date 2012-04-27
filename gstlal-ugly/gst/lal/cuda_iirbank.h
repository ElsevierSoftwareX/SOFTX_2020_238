/*
 * Copyright (C) 2010 Qi Chu <chicsheep@gmail.com>
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


#include <gstlal_iirbankCuda.h>
/*
 * ============================================================================
 *
 * 				    Utilities
 *
 * ============================================================================
 */

unsigned iir_channels(const GSTLALIIRBankCuda *element);

guint64 get_available_samples(GSTLALIIRBankCuda *element);

void set_metadata(GSTLALIIRBankCuda *element, GstBuffer *buf, guint64 outsamples, gboolean gap);


/*
 * transform input samples to output samples using IIR in Gpu
 */

GstFlowReturn filter(GSTLALIIRBankCuda *element, GstBuffer *outbuf);

