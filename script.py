import requests
from datetime import datetime, timedelta
import pytz
import os
import json

# ------------------------
# Telegram 配置
BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"

# 马来西亚时区
tz = pytz.timezone('Asia/Kuala_Lumpur')

# ------------------------
# 放水/中等胜率时间段（预测，仅参考策略）
HIGH_WATER_PERIODS = [
    ("09:28", "10:05"),
    ("10:47", "11:33"),
    ("13:42", "14:18"),
    ("15:26", "16:10"),
    ("17:55", "18:40"),
    ("20:33", "21:22"),
    ("23:12", "23:58")
]

MEDIUM_WATER_PERIODS = [
    ("00:00", "00:40"),
    ("01:00", "01:40"),
    ("08:00", "09:28"),
    ("14:18", "15:26"),
    ("16:10", "17:55"),
    ("18:40", "19:40"),
    ("21:22", "22:05"),
    ("23:00", "23:12")
]

STATE_FILE = "water_state.json"

# ------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"high_active": False, "medium_active": False, "high_start": None, "medium_start": None}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# ------------------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
    except:
        pass

# ------------------------
def check_period(periods, now_time):
    current_time = now_time.strftime("%H:%M")
    for start, end in periods:
        if start <= current_time <= end:
            start_dt = datetime.strptime(start, "%H:%M").replace(year=now_time.year, month=now_time.month, day=now_time.day)
            end_dt = datetime.strptime(end, "%H:%M").replace(year=now_time.year, month=now.month, day=now_time.day)
            remaining = int((end_dt - now_time).total_seconds() // 60)
            return True, start_dt, end_dt, remaining
    return False, None, None, None

# ------------------------
def main():
    now = datetime.now(tz)
    state = load_state()

    # 高胜率检测
    high_active, high_start_dt, high_end_dt, high_remain = check_period(HIGH_WATER_PERIODS, now)
    # 中等胜率检测
    medium_active, medium_start_dt, medium_end_dt, medium_remain = check_period(MEDIUM_WATER_PERIODS, now)

    # 高胜率放水处理
    if high_active:
        if not state.get("high_active"):
            state["high_active"] = True
            state["high_start"] = now.strftime("%H:%M")
            msg = f"🎊 现在是【放水时段 高胜率】🔥\n🚨 预计放水结束时间: {high_end_dt.strftime('%H:%M')}\n🔥 剩余时间约 {high_remain} 分钟"
            send_telegram(msg)
    elif state.get("high_active"):
        # 放水结束
        start_time = state.get("high_start")
        end_time = now.strftime("%H:%M")
        start_dt = datetime.strptime(start_time, "%H:%M")
        end_dt = datetime.strptime(end_time, "%H:%M")
        duration = int((end_dt - start_dt).total_seconds() // 60)
        msg = f"⏹ 放水时段已结束 🏁\n⏱ 共持续 {duration} 分钟"
        send_telegram(msg)
        state["high_active"] = False
        state["high_start"] = None

    # 中等胜率处理
    if medium_active:
        if not state.get("medium_active"):
            state["medium_active"] = True
            state["medium_start"] = now.strftime("%H:%M")
            msg = f"✨ 现在是【中等胜率时段】💡\n🚨 预计结束时间: {medium_end_dt.strftime('%H:%M')}\n⏱ 剩余时间约 {medium_remain} 分钟"
            send_telegram(msg)
    elif state.get("medium_active"):
        start_time = state.get("medium_start")
        end_time = now.strftime("%H:%M")
        start_dt = datetime.strptime(start_time, "%H:%M")
        end_dt = datetime.strptime(end_time, "%H:%M")
        duration = int((end_dt - start_dt).total_seconds() // 60)
        msg = f"⏹ 中等胜率时段已结束 🏁\n⏱ 共持续 {duration} 分钟"
        send_telegram(msg)
        state["medium_active"] = False
        state["medium_start"] = None

    save_state(state)

if __name__ == "__main__":
    main()
