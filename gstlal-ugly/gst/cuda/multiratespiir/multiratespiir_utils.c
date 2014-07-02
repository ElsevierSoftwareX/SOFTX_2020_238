
#include "multiratespiir_utils.h"

gint cuda_multirate_spiir_init_cover_samples (gint rate, gint num_depths, gint down_filtlen, gint up_filtlen)
{
	gint i = num_depths;
	gint rateleft = rate; 
	for (i=num_depths; i>0; i--)
		rateleft = (rateleft - down_filtlen)/2;
	for (i=num_depths; i>0; i--)
		rateleft = (rateleft - up_filtlen)*2;
	return (rate - rateleft);


}


gint cuda_multirate_spiir_get_num_templates(CudaMultirateSPIIR *element)
{
	if( element->matrix_initialised)
		return element->outchannels;
	else 
		return 1;
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



