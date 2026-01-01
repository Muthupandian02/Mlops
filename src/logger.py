import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",f"{datetime.now().strftime('%m_%d_%Y')}") #joining the path D:\Mlops\logs\12_30_2025 

os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] line: %(lineno)d %(filename)10s  - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# logger = logging.getLogger(__name__)
