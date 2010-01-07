#include <Python.h>
#include <stdlib.h>


static struct PyMethodDef methods[] = {
	{NULL, NULL, 0}
};


void init_misc(void)
{
	(void) Py_InitModule("gstlal._misc", methods);
}
