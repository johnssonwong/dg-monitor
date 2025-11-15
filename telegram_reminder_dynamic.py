import requests
from datetime import datetime, timedelta
import pytz

# ===============================
# 配置
# ===============================
BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"
TIMEZONE = "Asia/Kuala_Lumpur"

# 高胜率放水段（带胜率等级）
HIGH_PROB_PERIODS_WEEKDAY = {
    0: [("09:28","10:05","🔥🔥🔥"),("15:26","16:10","🔥🔥🔥"),("20:33","21:22","🔥🔥🔥")],
    1: [("09:28","10:05","🔥🔥🔥"),("15:26","16:10","🔥🔥🔥"),("20:33","21:22","🔥🔥🔥")],
    2: [("09:28","10:05","🔥🔥🔥"),("15:26","16:10","🔥🔥🔥"),("20:33","21:22","🔥🔥🔥")],
    3: [("09:28","10:05","🔥🔥🔥"),("15:26","16:10","🔥🔥🔥"),("20:33","21:22","🔥🔥🔥")],
    4: [("09:28","10:05","🔥🔥🔥"),("15:26","16:10","🔥🔥🔥"),("20:33","21:22","🔥🔥🔥")],
    5: [("10:00","10:40","🔥🔥"),("13:42","14:18","🔥🔥"),("17:55","18:40","🔥🔥"),("23:12","23:58","🔥🔥")],
    6: [("10:00","10:40","🔥🔥"),("13:42","14:18","🔥🔥"),("17:55","18:40","🔥🔥"),("23:12","23:58","🔥🔥")],
}

# ===============================
REMINDER_STATE = {}

# ===============================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": message})

def is_in_period(now_time, start_str, end_str):
    start = datetime.strptime(start_str, "%H:%M").replace(
        year=now_time.year, month=now_time.month, day=now_time.day
    )
    end = datetime.strptime(end_str, "%H:%M").replace(
        year=now_time.year, month=now_time.month, day=now_time.day
    )
    if end < start:
        end += timedelta(days=1)
    return start <= now_time <= end, start, end

# ===============================
def main():
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    weekday = now.weekday()
    periods_today = HIGH_PROB_PERIODS_WEEKDAY.get(weekday, [])
    
    for start_str, end_str, level in periods_today:
        in_period, start, end = is_in_period(now, start_str, end_str)
        key = f"{start_str}-{end_str}"
        
        if in_period:
            remaining = int((end - now).total_seconds() / 60)
            message = (
                f"🎊 当前高胜率放水时段 {level}\n"
                f"🕒 时间：{start_str} - {end_str}\n"
                f"⏳ 预计放水结束时间：{end_str}\n"
                f"🔥 剩余约 {remaining} 分钟\n"
                f"✅ 可按策略入场（追连、多连、断连开单）"
            )
            # 动态刷新：每分钟发送更新（或只在剩余分钟变化时发送）
            last_remaining = REMINDER_STATE.get(key)
            if last_remaining != remaining:
                send_telegram(message)
                REMINDER_STATE[key] = remaining
        else:
            # 放水结束提醒
            if REMINDER_STATE.get(key) is not None:
                duration = int((end - start).total_seconds() / 60)
                message = f"✅ 放水已结束，共持续 {duration} 分钟"
                send_telegram(message)
                REMINDER_STATE[key] = None

if __name__ == "__main__":
    main()
