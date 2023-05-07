# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2022/03/03
"""
import os
import logging
import sys
import datetime
from loguru import logger


def check_dir(bd):
	if not os.path.exists(bd):
		old_mask = os.umask(0o022)
		os.makedirs(bd)
		os.umask(old_mask)


log_dir = os.getenv("LOG_DIR", r"./logs")
log_level = os.getenv("LOG_LEVEL", "INFO")
check_dir(log_dir)
date = datetime.date.today()


class InterceptHandler(logging.Handler):
	def emit(self, record):
		logger_opt = logger.opt(depth=6, exception=record.exc_info)
		msg = self.format(record)
		logger_opt.log(record.levelno, msg)


logging.basicConfig(handlers=[InterceptHandler()], level=0)
logger.configure(handlers=[{"sink": sys.stderr, "level": 'INFO'}])  # 配置日志到标准输出流
# logger.add(
#     f"{ log_dir }/{ date }.log", rotation="10 MB", encoding='utf-8', colorize=False, level=log_level
# )  # 配置日志到输出到文件
logger.add(
    f"{ log_dir }/server.log", rotation="1 day", retention=7, encoding='utf-8', colorize=False, level=log_level
)  # 配置日志到输出到文件
