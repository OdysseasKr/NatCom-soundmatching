import os
import datetime

class Logger:
    def __init__(self, path):
        super().__init__()
        datetime_now = str(datetime.datetime.now()).replace(" ","-")
        if not os.path.exists(path):
            os.mkdir(path)
        self.log_file = open(os.path.join(path,datetime_now+".log"), 'w')

    def write(self, string):
        self.log_file.write(string+'\n')

    def flush(self):
        self.log_file.flush()

    def close(self):
        self.log_file.close()