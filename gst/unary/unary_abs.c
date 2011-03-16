/*
 * Copyright (C) 2010 Leo Singer
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#include <unary_base.h>

#include <math.h>


static GstFlowReturn
transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstAudioFilter *audiofilter = GST_AUDIO_FILTER (trans);
  GstBufferFormat format = audiofilter->format.format;

  gpointer data = GST_BUFFER_DATA (buf);
  gpointer data_end = GST_BUFFER_DATA (buf) + GST_BUFFER_SIZE (buf);

  if (format >= GST_FLOAT64_LE) {
    double *ptr, *end = data_end;
    for (ptr = data; ptr < end; ptr++)
      *ptr = fabs (*ptr);
  } else if (format >= GST_FLOAT32_LE) {
    float *ptr, *end = data_end;
    for (ptr = data; ptr < end; ptr++)
      *ptr = fabsf (*ptr);
  } else {
    g_assert_not_reached ();
  }

  return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


static void
base_init (gpointer class)
{
  gst_element_class_set_details_simple (GST_ELEMENT_CLASS (class),
      "Absolute value",
      "Filter/Audio",
      "Calculate absolute value, y = |x|", "Leo Singer <leo.singer@ligo.org>");
}


static void
class_init (gpointer class, gpointer class_data)
{
  GstBaseTransformClass *basetransform_class = GST_BASE_TRANSFORM_CLASS (class);
  basetransform_class->transform_ip = GST_DEBUG_FUNCPTR (transform_ip);
}


GType
unary_abs_get_type (void)
{
  static GType type = 0;

  if (!type) {
    static const GTypeInfo info = {
      .class_size = sizeof (UnaryBaseClass),
      .base_init = base_init,
      .class_init = class_init,
      .instance_size = sizeof (UnaryBase),
    };
    type = g_type_register_static (UNARY_BASE_TYPE, "UnaryAbs", &info, 0);
  }

  return type;
}
