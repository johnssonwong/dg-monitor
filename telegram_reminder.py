import requests
from datetime import datetime, timedelta
import pytz
import os

# ===============================
# 配置
# ===============================
BOT_TOKEN = "8134230045:AAH6C_H53R_JRH98fGTqZFHsjkKALhsTh8"
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
def send_telegram(message, message_id=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        if message_id:
            # 编辑消息
            edit_url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
            requests.get(edit_url, params={"chat_id": CHAT_ID, "message_id": message_id, "text": message}, timeout=10)
            return message_id
        else:
            # 发送新消息
            r = requests.get(url, params={"chat_id": CHAT_ID, "text": message}, timeout=10)
            data = r.json()
            if data.get("ok"):
                return data["result"]["message_id"]
    except Exception as e:
        print("Telegram发送失败:", e)
    return message_id

def is_in_period(now_time, start_str, end_str):
    start = datetime.strptime(start_str, "%H:%M").replace(
        year=now_time.year, month=now_time.month, day=now_time.day
    )
    end = datetime.strptime(end_str, "%H:%M").replace(
        year=now_time.year, month=now_time.month, day=now_time.day
    )
    if end < start:  # 跨午夜处理
        end += timedelta(days=1)
    return start <= now_time <= end, start, end

# ===============================
def main():
    try:
        tz = pytz.timezone(TIMEZONE)
        now = datetime.now(tz)
        weekday = now.weekday()
        periods_today = HIGH_PROB_PERIODS_WEEKDAY.get(weekday, [])
        print(f"[DEBUG] 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} 周{weekday}")

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
                # 动态更新同一条消息
                message_id = REMINDER_STATE.get(key)
                REMINDER_STATE[key] = send_telegram(message, message_id)

            else:
                # 放水结束提醒
                if REMINDER_STATE.get(key):
                    duration = int((end - start).total_seconds() / 60)
                    send_telegram(f"✅ 放水已结束，共持续 {duration} 分钟")
                    REMINDER_STATE[key] = None

    except Exception as e:
        print("[ERROR] 脚本异常:", e)

if __name__ == "__main__":
    main()
