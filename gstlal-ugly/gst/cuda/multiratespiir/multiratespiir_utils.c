
#include "multiratespiir_utils.h"
#include <math.h>

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

void cuda_multirate_spiir_update_exe_samples (guint *num_exe_samples, guint new_value)
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



