#include <math.h>
#include <glib.h>
#include <gst/gst.h>
#include <gsl/gsl_matrix.h>
#include <gstlal.h>
#include <gstlal_triggergen.h>
#include <gst/base/gstbasesink.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LALStdlib.h>


#define DEFAULT_SNR_THRESH 5.5

static double eta(double m1, double m2)
  {
  return m1*m2/(m1+m2)/(m1+m2);
  }

static double mchirp(double m1, double m2)
  {
  return pow(m1*m2,0.6) / pow(m1+m2,0.2);
  }

static double effective_distance(double snr, double sigmasq)
  {
  return sqrt(sigmasq) / snr;
  }

static void setup_bankfile_input(GSTLALTriggerGen *element)
  {
  SnglInspiralTable *bank = NULL;
  SnglInspiralTable *tmp = NULL;
  int i = 0;
  int j = 0;
  if (element->bank_filename)
    {
    int numtemps = LALSnglInspiralTableFromLIGOLw( &bank, element->bank_filename,-1,-1);
    element->mass1 = (double *) calloc(numtemps,sizeof(double) );
    element->mass2 = (double *) calloc(numtemps,sizeof(double) );
    element->tau0 = (double *) calloc(numtemps,sizeof(double) );
    element->tau3 = (double *) calloc(numtemps,sizeof(double) );
    element->sigmasq = (double *) calloc(numtemps,sizeof(double) );
    element->Gamma = (double *) calloc(10 * numtemps,sizeof(double) );
    i = 0;
    while (bank)
      {
      element->mass1[i] = bank->mass1;
      element->mass2[i] = bank->mass2; 
      element->tau0[i] = bank->tau0;
      element->tau3[i] = bank->tau3;
      element->sigmasq[i] = bank->sigmasq;
      for (j=0; j < 10; j++)
        {
        element->Gamma[10*i + j] = bank->Gamma[j];
        }
      tmp = bank;
      bank = bank->next;
      free(tmp);
      i++;
      }
    }
  }

static void append_trigger(LIGOTimeGPS epoch, double SNR, int channel, SnglInspiralTable *head, GSTLALTriggerGen *element)
  {
  int i;
  (head)->snr = SNR;
  (head)->end_time = epoch;
  if (!(element->mass1)) (head)->mass1 = channel;
  else 
    {
    (head)->mass1 = element->mass1[channel];
    (head)->mass2 = element->mass2[channel];
    (head)->tau0 = element->tau0[channel];
    (head)->tau3 = element->tau3[channel];
    (head)->sigmasq = element->sigmasq[channel];
    (head)->eta = eta( (head)->mass1, (head)->mass2 );
    (head)->mchirp = mchirp( (head)->mass1, (head)->mass2 );
    (head)->eff_distance = effective_distance( (head)->snr, (head)->sigmasq );
    for (i=0; i < 10; i++)
      {
      /* gammas are the 10 coefficients of a 4x4 symmetric matrix */
      (head)->Gamma[i] = element->Gamma[10*channel +i];
      }
    }
  }
 
static void write_trigger_file( SnglInspiralTable *first, char *filename )
  {
  LALStatus *status = NULL;
  LIGOLwXMLStream * xml = NULL;
  SnglInspiralTable *tmp = NULL;
  MetadataTable table;
  table.snglInspiralTable  = first;
  /* write the table to the file */
  xml = XLALOpenLIGOLwXMLFile( filename );
  status = (LALStatus *) calloc(1,sizeof(LALStatus));
  LALBeginLIGOLwXMLTable (status, xml, sngl_inspiral_table);
  LALWriteLIGOLwXMLTable (status, xml, table, sngl_inspiral_table);
  LALEndLIGOLwXMLTable (status, xml);
  XLALCloseLIGOLwXMLFile(xml);

  /* free the linked list */
  while (first)
    {
    tmp = first;
    first = first->next;
    free(tmp);
    }
  free(status);
  }

/* Override the render function.  This will write files out.  It is really
 * the only thing that "does anything" */
static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buffer)
  {
  /* This is a bit confusing, but you need to cast the sink structure into a GSTLALTriggerGen to access the snr_thresh property */
  GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(sink);
  int samples;
  int numchannels = 1;
  int rate = 0;
  int i,j;
  double SNR = 0;
  guint64 base_time = GST_BUFFER_TIMESTAMP(buffer);
  double *data = (double *) GST_BUFFER_DATA(buffer);
  LIGOTimeGPS epoch;
  SnglInspiralTable *head = NULL;
  SnglInspiralTable *first = head;
  char filename[255];
  double *max_snr = NULL;
  LIGOTimeGPS *time_data = NULL;
  int first_trigger;

  /* check to see if we have a bank file and need to read it. Should only be 
   * done once */
  if ( element->bank_filename && !(element->mass1) )
    {
    setup_bankfile_input(element);
    }
  
  sprintf(filename, "INSPIRAL-%llu.xml",(long long unsigned) base_time);

  /* Use the built in Macro to extract the buffers caps */
  GstCaps *caps = GST_BUFFER_CAPS(buffer);
  /* Now that we have the caps extract the structure */
  GstStructure *structure;
  
  structure = gst_caps_get_structure(caps, 0);

  /* Now that we have the caps structure try to extract the number of channels*/
  gst_structure_get_int(structure, "channels", &numchannels);
  gst_structure_get_int(structure, "rate", &rate);
  max_snr = (double *) calloc(numchannels, sizeof(double));
  time_data = (LIGOTimeGPS *) calloc(numchannels, sizeof(LIGOTimeGPS));

  samples = GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer);

  for (i = 0; i < samples; i++)
    {
    for (j = 0; j < numchannels; j++)
      {
      SNR = fabs(data[numchannels*i + j]);
      XLALINT8NSToGPS(&epoch,base_time);
      XLALGPSAdd( &epoch, (REAL8) i / rate);
      if (SNR > element->snr_thresh)
        {
        if (SNR > max_snr[j])
          {
          max_snr[j] = SNR;
          time_data[j] = epoch;
          }
	}
      }
    }
  
  first_trigger = 1;
  for (j = 0; j < numchannels; j++)
    {
    if (max_snr[j]) 
      {
      head = (SnglInspiralTable *) calloc(1,sizeof(SnglInspiralTable));
      if (first_trigger) 
        {
        first = head;
        first_trigger = 0;
        }
      append_trigger(time_data[j], max_snr[j], j, head, element);
      head = head->next;
      }
    }

  write_trigger_file(first, filename);
  free(max_snr);
  free(time_data);
  return GST_FLOW_OK;
  }

/* These functions allow "command line arguments" */

enum property {
        ARG_SNR_THRESH = 1, 
	ARG_BANK_FILENAME
};

static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
  {
  GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);

  switch(id) 
    {
    case ARG_SNR_THRESH:
      element->snr_thresh = g_value_get_double(value);
      break;
    case ARG_BANK_FILENAME:
      free(element->bank_filename);
      element->bank_filename = g_value_dup_string(value);
      break;
    }
  }


static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec)
  {
  GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);

  switch(id)
    {
    case ARG_SNR_THRESH:
      g_value_set_double(value,element->snr_thresh);
      break;
    case ARG_BANK_FILENAME:
      g_value_set_string(value,element->bank_filename);
      break;
    }
  }



static GstBaseSink *parent_class = NULL;

static void finalize(GObject *object)
  {
  GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
  if (element->mass1) free(element->mass1);
  element->mass1 = NULL;
  if (element->mass2) free(element->mass2);
  element->mass2 = NULL;
  if (element->tau0) free(element->tau0);
  element->tau0 = NULL;
  if (element->tau3) free(element->tau3);
  element->tau3 = NULL;
  if (element->sigmasq) free(element->sigmasq);
  element->sigmasq = NULL;
  if (element->Gamma) free(element->Gamma);
  element->Gamma = NULL;
  if (element->bank_filename) free(element->bank_filename);
  element->bank_filename = NULL;
  G_OBJECT_CLASS(parent_class)->finalize(object);
  }
  
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
        GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(class);
        parent_class = g_type_class_ref(GST_TYPE_BASE_SINK);
	/* Override the render function */
	gstbasesink_class->render = render;
	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
        gobject_class->finalize = finalize;

        g_object_class_install_property(gobject_class, ARG_BANK_FILENAME, g_param_spec_string("bank-filename", "Bank file name", "Path to XML file used to generate the template bank", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SNR_THRESH, g_param_spec_double("snr-thresh", "SNR Threshold", "SNR Threshold that determines a trigger", 0, G_MAXDOUBLE, DEFAULT_SNR_THRESH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));


}


static void instance_init(GTypeInstance *object, gpointer class)
{
        GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
        GstPad *pad;

        /* configure sink pad */
        pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
        /*gst_pad_set_setcaps_function(pad, setcaps);*/
	gst_object_unref(pad);
	element->snr_thresh = DEFAULT_SNR_THRESH;
        /*element->bank_filename = NULL;*/
        element->mass1 = NULL;
        element->mass2 = NULL;
        element->tau0 = NULL;
        element->tau3 = NULL;
        element->sigmasq = NULL;
        element->Gamma = NULL;
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
                type = g_type_register_static(GST_TYPE_BASE_SINK, "lal_triggergen", &info, 0);
        }

        return type;
}

