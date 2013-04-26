/*
 * framecpp filesink
 *
 * Copyright (C) 2013  Branson Stephens, Kipp Cannon
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


#ifndef __FRAMECPP_FILESINK_H__
#define __FRAMECPP_FILESINK_H__


#include <glib.h>
#include <gst/gst.h>

G_BEGIN_DECLS

/*
 * framecpp_filesink element
 */

#define FRAMECPP_FILESINK_TYPE \
        (framecpp_filesink_get_type())
#define FRAMECPP_FILESINK(obj) \
        (G_TYPE_CHECK_INSTANCE_CAST((obj), FRAMECPP_FILESINK_TYPE, FRAMECPPFilesink))
#define FRAMECPP_FILESINK_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_CAST((klass), FRAMECPP_FILESINK_TYPE, FRAMECPPFilesinkClass))               
#define GST_IS_FRAMECPP_FILESINK_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_TYPE((klass), FRAMECPP_FILESINK_TYPE))

typedef struct {
        GstBinClass parent_class;
} FRAMECPPFilesinkClass;

typedef struct {
        GstBin element;
        gchar *frame_type;
        gchar *instrument;
        gchar *path;
        GstClockTime timestamp;
        // the Multifilesink object.
        GstElement *mfs;
} FRAMECPPFilesink;


GType framecpp_filesink_get_type(void);

G_END_DECLS


#endif /* __FRAMECPP_FILESINK_H__ */
