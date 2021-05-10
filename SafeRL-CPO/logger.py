import threading
import pickle
import glob
import time
import os

class Logger:
    def __init__(self, env_name, save_name):
        now = time.localtime()
        self.save_name = save_name
        save_name = '{}/{}_log'.format(env_name, save_name.lower())
        if not os.path.isdir(save_name):
            os.makedirs(save_name)
        self.log_name = save_name+"/record_%04d%02d%02d_%02d%02d%02d.pkl" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        self.log = []
        self.start_t = time.time()
        self.lock = threading.Lock()


    def write(self, data):
        with self.lock:
            self.log.append(data)
            if time.time() - self.start_t > 10:
                self.save()
                self.start_t = time.time()


    def save(self):
        with open(self.log_name, 'wb') as f:
            pickle.dump(self.log, f)
