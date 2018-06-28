import sys, select, termios, tty
import threading
import time


class keyboard_thread(threading.Thread):
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        self.key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.03)
			self.key = self.getKey()

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def read(self):
        return self.key
