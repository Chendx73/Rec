import logging
import datetime


class Log:
    def __init__(self):
        self.root_dir = 'log/'
        self.filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%s") + ".log"
        logging.basicConfig(format="%(asctime)s - %(filename)s[line-%(lineno)d] - %(levelname)s - %(message)s",
                            level=logging.INFO, filename=self.root_dir + self.filename, filemode='a')
        self.logger = logging.getLogger(__name__)
