"""
Text-mode progress bars
"""
__copyright__ = "Copyright 2010, Leo Singer"
__author__    = "Leo Singer <leo.singer@ligo.org>"
__all__       = ["ProgressBar"]


import sys


def getTerminalSize():
    """Return terminal size as a (columns, rows) tuple."""
    # From http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct, os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
        '1234'))
        except:
            return None
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (env['LINES'], env['COLUMNS'])
        except:
            cr = (25, 80)
    return int(cr[1]), int(cr[0])


class ProgressBar:
    """Display a text progress bar."""
    
    def __init__(self, text='Working', max=1, value=0, textwidth=24, fid=sys.stderr):
        self.text = text
        self.max = max
        self.value = value
        self.textwidth = textwidth
        self.fid = fid
        self.twiddle = 0
    
    def show(self):
        """Redraw the text progress bar."""
        from math import floor, ceil
        
        if len(self.text) > self.textwidth:
            label = self.text[0:self.textwidth]
        else:
            label = self.text.rjust(self.textwidth)
        
        terminalSize = getTerminalSize()
        if terminalSize is None:
            terminalSize = 80
        else:
            terminalSize = terminalSize[0]
        
        barWidth = terminalSize - self.textwidth - 10
        
        if self.value is None or self.value < 0:
            if self.twiddle == 0:
                pattern = ' ..'
                self.twiddle = 1
            elif self.twiddle == 1:
                pattern = '. .'
                self.twiddle = 2
            else:
                pattern = '.. '
                self.twiddle = 0
            barSymbols = (pattern * int(ceil(barWidth/3.0)))[0:barWidth]
            progressFractionText = '   . %'
        else:
            solidBlock = '|'
            partialBlocks = ' .:!'
            blank = ' '
            
            progressFraction = float(self.value) / self.max
            nBlocks = progressFraction * barWidth
            nBlocksInt = int(floor(progressFraction * barWidth))
            partialBlock = partialBlocks[int(floor((nBlocks - nBlocksInt) * len(partialBlocks)))]
            nBlanks = barWidth - nBlocksInt - 1
            barSymbols = (solidBlock * nBlocksInt) + partialBlock + (blank * nBlanks)
            progressFractionText = ('%.1f%%' % (100*progressFraction)).rjust(6)
        
        print >>self.fid, '\r\x1B[1m' + label + '\x1B[0m [' + barSymbols + ']' + progressFractionText,
        self.fid.flush()
    
    def update(self, value = None, text = None):
        """Redraw the progress bar, optionally changing the value and text."""
        if text is not None:
            self.text = text
        if value is not None:
            self.value = value
        self.show()


def demo():
    """Demonstrate progress bar."""
    from time import sleep
    maxProgress = 1000
    progressbar = ProgressBar(max=maxProgress)
    for i in range(-100,maxProgress):
        sleep(0.01)
        progressbar.update(i)
    print >>sys.stderr, ''


if __name__ == '__main__':
	demo()
