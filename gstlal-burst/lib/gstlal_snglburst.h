#ifndef __GSTLAL_SNGLBURST_H__
#define __GSTLAL_SNGLBURST_H__

#include <glib.h>
#include <lal/LIGOMetadataTables.h>

G_BEGIN_DECLS


int gstlal_snglburst_array_from_file(char *bank_filename, SnglBurst **bankarray);


G_END_DECLS
#endif	/* __GSTLAL_SNGLBURST_H__ */
