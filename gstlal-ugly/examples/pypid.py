#!/usr/bin/env python3

import os.path
import time

class PidMon(dict):
    def __init__(self, pid):
        self.pid = pid        
    
    def update(self):
        tim = time.time()
        dirname = os.path.join('/proc', self.pid)
        taskdirname = os.path.join(dirname, 'task')
        self.saveStats(tim, os.path.join('/proc', self.pid, 'stat'))
        for subprocid in os.listdir(taskdirname):
            if subprocid != self.pid:
                self.saveStats(tim, os.path.join(taskdirname, subprocid, 'stat'))
    
    def saveStats(self, tim, path):
        stats = self.readStats(path)
        pid = stats['pid']
        if pid not in self.keys():
            self[pid] = [pid, stats['comm'], []]
        self[pid][2].append( (tim, stats['utime'], stats['stime']) )
    
    @staticmethod
    def readStats(path):
        file = open(path, 'r')
        line = file.readline()
        file.close()
        fields = line.split()
        return {
                              'pid':int(fields[0]),
                             'comm':fields[1],
                            'state':fields[2],
                             'ppid':int(fields[3]),
                             'pgrp':int(fields[4]),
                          'session':int(fields[5]),
                           'tty_nr':int(fields[6]),
                            'tpgid':int(fields[7]),
                            'flags':int(fields[8]),
                           'minflt':int(fields[9]),
                          'cminflt':int(fields[10]),
                           'majflt':int(fields[11]),
                          'cmajflt':int(fields[12]),
                            'utime':int(fields[13]),
                            'stime':int(fields[14]),
                           'cutime':int(fields[15]),
                           'cstime':int(fields[16]),
                         'priority':int(fields[17]),
                             'nice':int(fields[18])#,
#                     'itrealvalue':int(fields[19]),
#                       'starttime':int(fields[20]),
#                           'vsize':int(fields[21]),
#                             'rss':int(fields[22]),
#                          'rsslim':int(fields[23]),
#                       'startcode':int(fields[24]),
#                         'endcode':int(fields[25]),
#                      'startstack':int(fields[26]),
#                         'kstkesp':int(fields[27]),
#                         'kstkeip':int(fields[28]),
#                          'signal':int(fields[29]),
#                         'blocked':int(fields[30]),
#                       'sigignore':int(fields[31]),
#                        'sigcatch':int(fields[32]),
#                           'wchan':int(fields[33]),
#                           'nswap':int(fields[34]),
#                          'cnswap':int(fields[35]),
#                     'exit_signal':int(fields[36]),
#                       'processor':int(fields[37]),
#                     'rt_priority':int(fields[38]),
#                          'policy':int(fields[39]),
#           'delayacct_blkio_ticks':int(fields[40]),
#                      'guest_time':int(fields[41]),
#                     'cguest_time':int(fields[42])
        }


if __name__ == '__main__':
    import sys
    import time
    
    pidmon = PidMon(sys.argv[1])
    for x in range(1200):
        pidmon.update()
        time.sleep(.5)
    
    file = open('pidmon.html', 'w')
    print("<html><head><title>pidmon</title></head><body>", file=file)
    
    from pylab import *
    keys = [k[1] for k in sorted([(val[1],val[0]) for val in pidmon.values()])]
    for key in keys:
        val = pidmon[key]
        figure()
        ctime, utime, stime = zip(*val[2])
        ctime = array(ctime)
        utime = array(utime)
        stime = array(stime)
        plot(ctime[:-1], diff(utime)/diff(ctime), label='user')
        plot(ctime[:-1], diff(stime)/diff(ctime), label='system')
        plot(ctime[:-1], diff(utime+stime)/diff(ctime), label='user+system')
        xlabel('ctime')
        ylabel('d(utime+stime)/d(ctime)')
        title('%s %s' % (val[0], val[1]))
        legend()
        filename = '%s.png' % key
        print(filename)
        savefig(filename)
        print("<img src=\"%s\"><br>" % filename, file=file)
    print("</body><html>", file=file)
    file.close()
