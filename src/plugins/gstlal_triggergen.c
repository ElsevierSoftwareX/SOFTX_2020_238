#include <glib.h>
#include <gst/gst.h>
#include <gsl/gsl_matrix.h>
#include <gstlal.h>
#include <gstlal_triggergen.h>
#include <gst/base/gstbasesink.h>




/* Override the render function.  This will write files out */
static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buffer)
  {
  int samples;
  double max;
  int numchannels = 1;
  gsl_matrix_view m;
  /* Use the built in Macro to extract the buffers caps */
  GstCaps *caps = GST_BUFFER_CAPS(buffer);
  /* Now that we have the caps extract the structure */
  GstStructure *structure;
  structure = gst_caps_get_structure(caps, 0);

  /* Now that we have the caps structure try to extract the number of channels*/
  gst_structure_get_int(structure, "channels", &numchannels);


  samples = GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer);
  
  m = gsl_matrix_view_array((double *) GST_BUFFER_DATA(buffer), samples, numchannels); 

  max = gsl_matrix_max(&(m.matrix));
  fprintf(stderr,"Max SNR %f\n",max);

  /* We are done with the buffer so free it */
  gst_buffer_unref(buffer);
  return GST_FLOW_OK;
  }


static GstBaseSink *parent_class = NULL;

static void base_init(gpointer g_class)
  {

  static GstElementDetails plugin_details = {
                  "Trigger Generator",
		  "Filter",
      	 	  "SNRs in Triggers out",
		  "Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"};

  GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);
  gst_element_class_set_details (element_class, &plugin_details);


  gst_element_class_add_pad_template(
                element_class,
                gst_pad_template_new(
                        "sink",
                        GST_PAD_SINK,
                        GST_PAD_ALWAYS,
                        gst_caps_from_string(
                                "audio/x-raw-float, " \
                                "channels = (int) [ 1, MAX ], " \
                                "endianness = (int) BYTE_ORDER, " \
                                "width = (int) {32, 64}"
                        )
                )
        );



  }




static void class_init(gpointer class, gpointer class_data)
{
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(class);
        parent_class = g_type_class_ref(GST_TYPE_BASE_SINK);
	/* Override the render function */
	gstbasesink_class->render = render;

}


static void instance_init(GTypeInstance *object, gpointer class)
{
        GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
        GstPad *pad;

        /* configure sink pad */
        pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
        /*gst_pad_set_setcaps_function(pad, setcaps);*/
	gst_object_unref(pad);
}


GType gstlal_triggergen_get_type(void)
{
        static GType type = 0;

        if(!type) {
                static const GTypeInfo info = {
                        .class_size = sizeof(GSTLALTriggerGenClass),
                        .class_init = class_init,
                        .base_init = base_init,
                        .instance_size = sizeof(GSTLALTriggerGen),
                        .instance_init = instance_init,
                };
                type = g_type_register_static(GST_TYPE_ELEMENT, "lal_triggergen", &info, 0);
        }

        return type;
}

