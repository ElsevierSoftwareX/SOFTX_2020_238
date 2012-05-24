import socket
import sys
from gstlal.pyzeroconf import Zeroconf

class MyListener(object):
	def addService(self, zeroconf, type, name):
		print >>sys.stderr, "Service", name, "added"
		print >>sys.stderr, "Type is", type
		info = zeroconf.getServiceInfo(type, name)
		print >>sys.stderr, "Address is", str(socket.inet_ntoa(info.getAddress()))
		print >>sys.stderr, "Port is", info.getPort()
		print >>sys.stderr, "Weight is", info.getWeight()
		print >>sys.stderr, "Priority is", info.getPriority()
		print >>sys.stderr, "Server is", info.getServer()
		print >>sys.stderr, "Text is", info.getText()
		print >>sys.stderr, "Properties are", info.getProperties()

	def removeService(self, zeroconf, type, name):
		print >>sys.stderr, "Service", name, "removed"

if __name__ == '__main__':	
	print "Multicast DNS Service Discovery for Python Browser test"
	Zeroconf.ServiceBrowser(Zeroconf.Zeroconf("0.0.0.0"), "_http._tcp.local.", MyListener())
	raw_input("Testing browsing for a service.  Press Enter to quit ...\n")
