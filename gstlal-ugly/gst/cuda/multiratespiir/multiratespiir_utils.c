/* GStreamer
 * Copyright (C) 2014 Qi Chu
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more deroll-offss.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#ifdef __cplusplus
extern "C" {
#endif

#include "multiratespiir_utils.h"
#include <math.h>
#include <string.h>
#include <LIGOLw_xmllib/LIGOLwHeader.h>

#ifdef __cplusplus
}
#endif

void
cuda_multirate_spiir_read_bank_id(const char *fname, gint *bank_id)
{
	XmlNodeStruct	xns;
	XmlParam	xparam = {0, NULL};
	strncpy((char *)xns.tag, "Param-bank_id:param", XMLSTRMAXLEN);
	xns.processPtr = readParam;
	xns.data = &xparam;

	// start parsing, get the information of depth
	parseFile(fname, &xns, 1);

	*bank_id = atoi((const gchar*)xparam.data);
	xmlCleanupParser();
	xmlMemoryDump();

}
void
cuda_multirate_spiir_read_ndepth_and_rate(const char *fname, guint *num_depths, gint *rate)
{
	XmlNodeStruct	xns;
	XmlParam	xparam = {0, NULL};
	strncpy((char *)xns.tag, "Param-sample_rate:param", XMLSTRMAXLEN);
	xns.processPtr = readParam;
	xns.data = &xparam;

	// start parsing, get the information of depth
	parseFile(fname, &xns, 1);

	int ndepth = 0;
	int maxrate = 1;
	int temp;
	gchar **rates = g_strsplit((const gchar*)xparam.data, (const gchar*)",", 16);
	while (rates[ndepth])
	{
		temp = atoi((const char*)rates[ndepth]);
		maxrate = maxrate > temp ? maxrate : temp;
		++ndepth;	
	}
	g_strfreev(rates);
	freeParam(&xparam);
	*num_depths = (guint)ndepth;
	*rate = (gint)maxrate;

	xmlCleanupParser();
	xmlMemoryDump();
}


void cuda_multirate_spiir_init_cover_samples (guint *num_head_cover_samples, guint *num_tail_cover_samples, gint rate, guint num_depths, gint down_filtlen, gint up_filtlen)
{
	guint i = num_depths;
	gint rate_start = up_filtlen, rateleft; 
	gboolean success = FALSE;
	for (i=num_depths-1; i>0; i--)
	       rate_start = rate_start * 2 + down_filtlen/2 ;

	*num_head_cover_samples = rate_start;
	*num_tail_cover_samples = rate_start - up_filtlen;


}

void cuda_multirate_spiir_update_exe_samples (gint *num_exe_samples, gint new_value)
{
	*num_exe_samples = new_value;
}

gboolean cuda_multirate_spiir_parse_bank (gdouble *bank, guint *num_depths, gint *
		outchannels)
{
	// FIXME: do some check?
	*num_depths = (guint) bank[0];
	*outchannels = (gint) bank[1] * 2;
	return TRUE;
}

guint cuda_multirate_spiir_get_outchannels(CudaMultirateSPIIR *element)
{
		return element->outchannels;
}

guint cuda_multirate_spiir_get_num_head_cover_samples(CudaMultirateSPIIR *element)
{
	return element->num_head_cover_samples;
}

void cuda_multirate_spiir_add_two_data(float *data1, float *data2, gint len)
{
	int i;
	for(i=0; i<len; i++)
		data1[i] = data1[i] + data2[i];
}

guint64 cuda_multirate_spiir_get_available_samples(CudaMultirateSPIIR *element)
{
	return gst_adapter_available(element->adapter) / ( element->width / 8 );
}




