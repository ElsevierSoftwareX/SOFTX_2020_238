/*
 * A template bank.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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


#ifndef __GSTLAL_TEMPLATEBANK_H__
#define __GSTLAL_TEMPLATEBANK_H__


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include <lal/RealFFT.h>
#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
/*#include <rfftw.h>*/


G_BEGIN_DECLS


#define GSTLAL_TEMPLATEBANK_TYPE \
	(gstlal_templatebank_get_type())
#define GSTLAL_TEMPLATEBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TEMPLATEBANK_TYPE, GSTLALTemplateBank))
#define GSTLAL_TEMPLATEBANK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TEMPLATEBANK_TYPE, GSTLALTemplateBankClass))
#define GST_IS_GSTLAL_TEMPLATEBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TEMPLATEBANK_TYPE))
#define GST_IS_GSTLAL_TEMPLATEBANK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TEMPLATEBANK_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALTemplateBankClass;


typedef struct {
	GstElement element;

	GstAdapter *adapter;

	GstPad *matrixpad;
	GstPad *chifacspad;
	GstPad *srcpad;
	GstPad *sumsquarespad;

	char *reference_psd_filename;
	char *template_bank_filename;
	int sample_rate;
	double t_start;
	double t_end;
	double t_total_duration;
	unsigned int snr_length;

	gboolean next_is_discontinuity;
	unsigned long next_sample;
	GstClockTime output_timestamp;

	gsl_matrix *U;
	gsl_vector *S;
	gsl_matrix *V;
	gsl_vector *chifacs;

	int fft_input_length;
	gsl_vector *fft_f;
	gsl_vector *fft_s;

	/*gsl_fft_real_wavetable *fft_real;
	gsl_fft_halfcomplex_wavetable *fft_hc;
	gsl_fft_real_workspace *fft_work;*/
	
	REAL8FFTPlan *fwdplan;
	REAL8FFTPlan *revplan;
	COMPLEX16Vector *fft_sv;
	COMPLEX16Vector *fft_fv;
	
	COMPLEX16Vector **fft_filters;

	/*rfftw_plan fwdplan;
	rfftw_plan revplan;*/

} GSTLALTemplateBank;


GType gstlal_templatebank_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TEMPLATEBANK_H__ */
