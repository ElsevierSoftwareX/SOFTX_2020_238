#include <gstlal_snglburst.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/SnglBurstUtils.h>
#include <lal/LIGOLwXMLBurstRead.h>
#include <lal/LALStdlib.h>


int gstlal_snglburst_array_from_file(char *bank_filename, SnglBurst **bankarray)
{
	SnglBurst *this = NULL;
	SnglBurst *bank = NULL;
	int num = 0;
	bank = this = XLALSnglBurstTableFromLIGOLw(bank_filename);
	/* count the rows */
	while (bank) {
		num++;
		bank = bank->next;
		}

	*bankarray = bank = (SnglBurst *) calloc(num, sizeof(SnglBurst));

	/* FIXME do some basic sanity checking */

	/*
	 * copy the linked list of templates constructed into the template
	 * array.
	 */

	while (this) {
		SnglBurst *next = this->next;
		this->snr = 0;
		*bank = *this;
		bank->next = NULL;
		bank++;
		XLALFree(this);
		this = next;
	}

	return num;
}
