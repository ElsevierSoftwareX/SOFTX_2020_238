
/* The quality of downsampler should be choosen carefully. Tests show that
 * quality = 6 will produce 1.5 times single events than quality = 9
 */
#define DOWN_FILT_LEN 192 
#define DOWN_QUALITY 9

/* The quality of upsampler can be as small as 1. It won't affect the
 * number of single events
 */
#define UP_FILT_LEN 16 
#define UP_QUALITY 1
