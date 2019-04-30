import datetime

class ToggleLogger(object):
    def __init__(self, date_format, enabled=True):
        self.date_format = date_format
        self.enabled = enabled

    def info(self, message):
        if self.enabled:
            date = datetime.datetime.now()
            date_str = date.strftime(self.date_format)

            print('{} - {}'.format(date_str, message))
