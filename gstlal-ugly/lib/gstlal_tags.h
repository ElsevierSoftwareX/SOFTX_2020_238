/*
 * Copyright (C) 2000,2001,2008  Kipp C. Cannon
 * Copyrigth (C) 2010  Leo Singer
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


#ifndef __GSTLAL_TAGS_H__
#define __GSTLAL_TAGS_H__


#include <gst/gst.h>


G_BEGIN_DECLS


/*
 * Custom tags
 */


#define GSTLAL_TAG_INSTRUMENT "instrument"
#define GSTLAL_TAG_CHANNEL_NAME "channel-name"
#define GSTLAL_TAG_UNITS "units"


void gstlal_register_tags(void);


G_END_DECLS


#endif	/* __GSTLAL_TAGS_H__ */
