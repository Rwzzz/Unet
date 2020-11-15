import os
import sys

class Logger():
    def __init__(self, filename='log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self,message):
        #输出到STDOUT终端
        #self.terminal.write(message)
        #重定向到在指定文件
        self.log.write(message)

    def flush(self):
        pass
    def close(self):
        self.log.close()


