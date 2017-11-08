#!/usr/bin/env python

import doctest, sys
from gstlal.stats import trigger_rate

sys.exit(doctest.testmod(trigger_rate).failed)
