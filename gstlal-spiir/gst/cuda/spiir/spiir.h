/*
 * Copyright (C) 2013 Qi Chu
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


#ifndef __CUDA_GST_LAL_IIRBANK_H__
#define __CUDA_GST_LAL_IIRBANK_H__


#include <complex.h>

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>


#include <gsl/gsl_matrix.h>
#include <cuda_runtime.h>

G_BEGIN_DECLS
#define CUDA_IIRBANK_TYPE \
	(cuda_iirbank_get_type())
#define CUDA_IIRBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), CUDA_IIRBANK_TYPE, GSTLALIIRBankCuda))
#define CUDA_IIRBANK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), CUDA_IIRBANK_TYPE, GSTLALIIRBankCudaClass))
#define GST_IS_CUDA_IIRBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), CUDA_IIRBANK_TYPE))
#define GST_IS_CUDA_IIRBANK_CLASS(klass) \
	(G_type_CHECK_CLASS_TYPE((klass), CUDA_IIRBANK_TYPE))

typedef struct _Complex8_F
{
	float re;
	float im;
} COMPLEX8_F;

typedef struct _Complex8_D
{
	double re;
	double im;
} COMPLEX8_D;

typedef struct {

	COMPLEX8_F *d_a1_s;
	COMPLEX8_F *d_b0_s;
	COMPLEX8_F *d_y_s;
	int *d_d_i;

	float *h_input_s;
	COMPLEX8_F *h_output_s;
	float *d_input_s;
	COMPLEX8_F *d_output_s;

	unsigned int num_templates;
	unsigned int num_filters;
	int dmax, dmin;
	unsigned int rate;
	unsigned int pre_input_length;
	unsigned int pre_output_length;
	unsigned int nb;

} IIRBankCuda_s;

typedef struct {
	GstBaseTransformClass parent_class;

	void (*rate_changed)(GstElement *, gint, void *);
} GSTLALIIRBankCudaClass;


typedef struct {
	GstBaseTransform element;

	/*
	 * input stream
	 */

	gint rate, width;
	gint deviceID; // gpu device ID
	cudaStream_t stream;
	GstAdapter *adapter;
	guint zeros_in_adapter;

	/*
	 * filter info
	 */

	GMutex *iir_matrix_lock;
	GCond *iir_matrix_available;
        gsl_matrix_int *delay;
        gsl_matrix_complex *a1;
        gsl_matrix_complex *b0;
        gsl_matrix_complex *y;
	gint64 latency;

	/*
	 * timestamp book-keeping
	 */

	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	IIRBankCuda_s *bank;

} GSTLALIIRBankCuda;


GType cuda_iirbank_get_type(void);


G_END_DECLS


#endif	/* __GST_LAL_IIRBANKCUDA_H__ */
