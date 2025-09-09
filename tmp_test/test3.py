from zoneinfo import ZoneInfo
from datetime import datetime

start_time = datetime.now(ZoneInfo("Asia/Shanghai"))
print("代码运行时间戳：", start_time.strftime("%Y-%m-%d %H:%M:%S"))
print("代码运行时间戳：", start_time.strftime("%Y%m%d_%H%M%S"))