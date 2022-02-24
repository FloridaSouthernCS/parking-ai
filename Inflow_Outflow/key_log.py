# key_log.py
from pynput.keyboard import Listener
import threading

class log(threading.Thread):
    def __init__(self, valid_keys):
        threading.Thread.__init__(self)
        self.valid_keys = valid_keys
        self.keys_clicked = []
        self.temp = lambda key: on_press(key, self)

    def run(self):
        with Listener(
                on_press=(self.temp)
                ) as self.listener:
            self.listener.join()

    def stop(self):
        self.listener.stop()
        print("Logger Stopped")
            
def on_press(key, self):
        keys_clicked = self.keys_clicked
        valid_keys = self.valid_keys

        try:
            if valid_keys == "":
                keys_clicked.append(key.char)
                # print(keys_clicked[-1], "clicked")
            elif key.char in valid_keys:
                keys_clicked.append(key.char)
                # print(keys_clicked[-1], "clicked")
        except Exception as e:
            pass