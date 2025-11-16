import requests
import traceback
from datetime import datetime, time, timedelta
import pytz

# ===============================
# 用户配置
# ===============================
BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"

# 放水监控时段（你可以随时让我更新）
DRAIN_PERIODS = [
    ("09:32", "09:52"),
    ("11:18", "11:43"),
    ("14:07", "14:29"),
    ("17:55", "18:16"),
    ("21:08", "21:31"),
    ("23:22", "23:47"),
]

# 时区：马来西亚（固定）
TZ = pytz.timezone("Asia/Kuala_Lumpur")

# ===============================
# 将时间字符串转为带时区 datetime
# ===============================
def to_tz_datetime(date: datetime, hm: str):
    hour, minute = map(int, hm.split(":"))
    dt = datetime(date.year, date.month, date.day, hour, minute)
    return TZ.localize(dt)

# ===============================
# 判断是否在放水时段
# ===============================
def is_now_in_period(now_dt):
    for (start_str, end_str) in DRAIN_PERIODS:
        start_dt = to_tz_datetime(now_dt, start_str)
        end_dt = to_tz_datetime(now_dt, end_str)

        # 若跨日则延长 end_dt
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        if start_dt <= now_dt <= end_dt:
            return True, start_dt, end_dt

    return False, None, None

# ===============================
# Telegram 推送函数
# ===============================
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    requests.post(url, data=data)

# ===============================
# 主程序（每分钟执行）
# ===============================
def main():
    try:
        now_dt = datetime.now(TZ)  # 强制带时区
        in_period, start_dt, end_dt = is_now_in_period(now_dt)

        if in_period:
            send_telegram(
                f"🔥【DG 放水提醒】\n\n"
                f"📌 当前时间：{now_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⏰ 放水时段：{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}\n"
                f"🚀 建议立即查看 DG 桌面走势（长龙 + 多连 + 断连开单）"
            )

    except Exception as e:
        send_telegram(
            "❗ DG Monitor 脚本捕获异常，已忽略并继续运行：\n"
            f"{e}\n\n"
            f"Traceback (truncated):\n{traceback.format_exc()}"
        )

if __name__ == "__main__":
    main()
