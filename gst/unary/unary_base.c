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
      "Unary operation base class",
      "Filter/Audio",
      "Base class for elements that provide unary arithmetic operations",
      "Leo Singer <leo.singer@ligo.org>");

  gst_audio_filter_class_add_pad_templates (GST_AUDIO_FILTER_CLASS (class),
      gst_caps_from_string ("audio/x-raw-float, "
          "rate = (int) [1, MAX], "
          "endianness = (int) BYTE_ORDER, "
          "width = (int) {32, 64}, " "channels = (int) [1, MAX]")
      );
}


GType
unary_base_get_type (void)
{
  static GType type = 0;

  if (!type) {
    static const GTypeInfo info = {
      .class_size = sizeof (UnaryBaseClass),
      .base_init = base_init,
      .instance_size = sizeof (UnaryBase),
    };
    type =
        g_type_register_static (GST_TYPE_AUDIO_FILTER, "UnaryBase", &info, 0);
  }

  return type;
}
