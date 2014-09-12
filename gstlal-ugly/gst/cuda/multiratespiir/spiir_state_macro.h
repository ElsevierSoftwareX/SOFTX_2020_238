
#define SPSTATE(i) (*(spstate+i)) 
#define SPSTATEDOWN(i) (SPSTATE(i)->downstate)
#define SPSTATEUP(i) (SPSTATE(i)->upstate)

typedef enum {
SP_OK = 0,
SP_RESAMPLER_NOT_INITED = -1,
SP_BANK_LOAD_ERR = -2
} SpInitReturn;

