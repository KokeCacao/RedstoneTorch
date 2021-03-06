import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None
        self.once_msg = set()

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1, once=0):
        if once:
            if message not in self.once_msg:
                self.once_msg.add(message)
            else:
                return

        if message is None: return
        message = str(message) + "\n"
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1 and self.file is not None:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass