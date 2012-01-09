#!/bin/bash

gstlal_excesspower  \
	--verbose  \
	--diagnostics  \
	--data-source fakeadvLIGO  \
	-s 2048  \
	-f ~/work/codedev/excesspower/pipeline/configurations/gstlal_excessopower.ini 
