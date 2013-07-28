/* float */

#define TYPE_STRING float
#define TYPE float
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING

/* double */

#define TYPE_STRING double
#define TYPE double
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING

/* complex */

#define TYPE_STRING float_complex
#define TYPE float complex
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING

/* double complex */

#define TYPE_STRING double_complex
#define TYPE double complex
#include "gstlal_peakfinder.ht"
#undef TYPE
#undef TYPE_STRING
