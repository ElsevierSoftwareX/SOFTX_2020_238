/*
 * Copyright (C) 2015 Qi Chu <qi.chu@uwa.edu.au>
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
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __COHFAR_ACCUMBACKGROUND_H__
#define __COHFAR_ACCUMBACKGROUND_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <postcoh/postcohtable.h>
#include <cohfar/background_stats.h>

G_BEGIN_DECLS
#define COHFAR_ACCUMBACKGROUND_TYPE \
	(cohfar_accumbackground_get_type())
#define COHFAR_ACCUMBACKGROUND(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), COHFAR_ACCUMBACKGROUND_TYPE, CohfarAccumbackground))
#define COHFAR_ACCUMBACKGROUND_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), COHFAR_ACCUMBACKGROUND_TYPE, CohfarAccumbackgroundClass))
#define GST_IS_COHFAR_ACCUMBACKGROUND(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), COHFAR_ACCUMBACKGROUND_TYPE))
#define GST_IS_COHFAR_ACCUMBACKGROUND_CLASS(klass) \
	(G_type_CHECK_CLASS_TYPE((klass), COHFAR_ACCUMBACKGROUND_TYPE))


typedef struct {
	GstElementClass parent_class;
} CohfarAccumbackgroundClass;


typedef struct {
	GstElement element;

	GstPad *sinkpad, *srcpad;

	char *ifos;
	int nifo;
	int ncombo; // ifo combination
	int write_ifo_mapping[MAX_NIFO];
	BackgroundStats **stats_snapshot;
	BackgroundStats **stats_prompt;
	BackgroundStatsPointerList *stats_list;

	int snapshot_interval;
	int hist_trials;
	gchar *history_fname;
	gchar *output_prefix;
	gchar *output_name;

	/*
	 * timestamp book-keeping
	 */
	GstClockTime t_end;
	GstClockTime t_roll_start;
} CohfarAccumbackground;


GType cohfar_accumbackground_get_type(void);


G_END_DECLS


#endif	/* __COHFAR_ACCUMBACKGROUND_H__ */
