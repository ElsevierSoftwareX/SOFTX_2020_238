/*
 *  NDS-based src element
 *  see https://www.lsc-group.phys.uwm.edu/daswg/wiki/NetworkDataServer2
 *
 *  Copyright (C) 2009  Leo Singer
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_NDSSRC_H__
#define __GSTLAL_NDSSRC_H__


#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>

#include <daqc.h>


G_BEGIN_DECLS


#define GSTLAL_NDSSRC_TYPE \
	(gstlal_ndssrc_get_type())
#define GSTLAL_NDSSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_NDSSRC_TYPE, GSTLALNDSSrc))
#define GSTLAL_NDSSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_NDSSRC_TYPE, GSTLALNDSSrcClass))
#define GSTLAL_NDSSRC_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GSTLAL_NDSSRC_TYPE, GSTLALNDSSrcClass))
#define GST_IS_GSTLAL_NDSSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_NDSSRC_TYPE))
#define GST_IS_GSTLAL_NDSSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_NDSSRC_TYPE))


typedef struct {
	GstBaseSrcClass parent_class;

	/**
	 * regex for parsing URIs
	 */

	GRegex *regex;
} GSTLALNDSSrcClass;


typedef struct {
	GstBaseSrc basesrc;

	char* host;
	gint port;
	enum nds_version version;

	daq_t* daq;
	gboolean needs_seek;

	char* channelName;
	enum chantype channelType;
	daq_channel_t* availableChannels;
	int countAvailableChannels;

} GSTLALNDSSrc;


GType gstlal_ndssrc_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_NDSSRC_H__ */
