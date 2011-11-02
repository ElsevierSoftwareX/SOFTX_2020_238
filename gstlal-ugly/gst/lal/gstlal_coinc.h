/*
 * Copyright (C) 2009 Leo Singer <leo.singer@ligo.org>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#ifndef __GSTLAL_COINC_H__
#define __GSTLAL_COINC_H__


#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                            Coincidence Generator
 *
 * ============================================================================
 */


#define GSTLAL_COINC_TYPE \
	(gstlal_coinc_get_type())
#define GSTLAL_COINC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_COINC_TYPE, GSTLALCoinc))
#define GSTLAL_COINC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_COINC_TYPE, GSTLALCoincClass))
#define GST_IS_GSTLAL_COINC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_COINC_TYPE))
#define GST_IS_GSTLAL_COINC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_COINC_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALCoincClass;


typedef struct {
	GstElement element;
	GstCollectPads *collect;
	GstPad* srcpad;
	GHashTable* trigger_sequence_hash;
	guint64 dt;
	guint padcounter;
} GSTLALCoinc;


GType gstlal_coinc_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_COINC_H__ */
