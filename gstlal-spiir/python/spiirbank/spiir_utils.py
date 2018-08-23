import os
import re
from glue.ligolw import ligolw, lsctables, array, param, utils, types

# FIXME:  require calling code to provide the content handler
class DefaultContentHandler(ligolw.LIGOLWContentHandler):
    pass
array.use_in(DefaultContentHandler)
param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)

def get_bankid_from_bankname(bankname):
	tmp_name = os.path.split(bankname)[-1]
	tmp_name = re.sub(r'[HLV]1', '', tmp_name)
	search_result = re.search(r'\d{1,4}', tmp_name)
	try:
		bankid = search_result.group()
	except:
		raise ValueError("bankid should be the first 3/4 digits of the given name, could not find the digits from %s" % tmp_name)

	bankid_strip = bankid.lstrip('0')
	if bankid_strip is '':
		return 0
	else:
		return int(bankid_strip)

def parse_iirbank_string(bank_string):
	"""
	parses strings of form 
	
	H1:bank1.xml,H2:bank2.xml,L1:bank3.xml,H2:bank4.xml,... 
	
	into a dictionary of lists of bank files.
	"""
	out = {}
	if bank_string is None:
		return out
	for b in bank_string.split(','):
		ifo, bank = b.split(':')
		out.setdefault(ifo, []).append(bank)
	return out



def get_maxrate_from_xml(filename, contenthandler = DefaultContentHandler, verbose = False):
    xmldoc = utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

    for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_iir_bank_Bank"):

        sample_rates = [int(float(r)) for r in param.get_pyvalue(root, 'sample_rate').split(',')]
    
    return max(sample_rates)


