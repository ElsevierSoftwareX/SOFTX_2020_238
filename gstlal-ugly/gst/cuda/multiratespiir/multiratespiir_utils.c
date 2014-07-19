
#include "multiratespiir_utils.h"
#include <math.h>

gint cuda_multirate_spiir_init_cover_samples (gint rate, gint num_depths, gint down_filtlen, gint up_filtlen)
{
	gint i = num_depths;
	gint rate_start = up_filtlen, rateleft; 
	gboolean success = FALSE;
	for (i=num_depths-1; i>0; i--)
	       rate_start = rate_start * 2 + down_filtlen/2 ;

	while (!success) {
		rateleft = rate_start;
		for (i=num_depths-1; i>0; i--) {
			if ((rateleft - down_filtlen/2) % 2 == 1) {
				rate_start = rate_start + 2;
				break;
			} else
				rateleft = (rateleft - down_filtlen/2)/2;
		}
		if (i == 0)
			success = TRUE;
	}
	for (i=num_depths-1; i>0; i--)
		rateleft = (rateleft - up_filtlen/2)*2;
	return (rate_start);


}

gboolean cuda_multirate_spiir_parse_bank (gdouble *bank, gint *num_depths, gint *
		outchannels)
{
	// FIXME: do some check?
	*num_depths = (gint) bank[0];
	*outchannels = (gint) bank[1];
	return TRUE;
}

gint cuda_multirate_spiir_get_num_templates(CudaMultirateSPIIR *element)
{
		return element->outchannels;
}

gint cuda_multirate_spiir_get_num_cover_samples(CudaMultirateSPIIR *element)
{
	return element->num_cover_samples;
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



