#!/bin/bash

# NOTE: Online mode can be activated by using lldata instead of fakeadvLIGO

gstlal_excesspower  \
	--verbose  \
	--data-source fakeadvLIGO  \
	-s 2048  \
	-f gstlal_excessopower.ini 
