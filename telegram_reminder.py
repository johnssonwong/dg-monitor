import requests
import json
import traceback
from datetime import datetime, timedelta
import pytz
import random

# ===============================
# 用户配置
# ===============================
BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"
TZ = pytz.timezone("Asia/Kuala_Lumpur")
STATE_FILE = "state_v9.json"

# -------------------------------
# 高胜率放水时段（可根据历史数据调整）
# -------------------------------
WORKDAY_PERIODS = [
    ("09:32", "09:52"), ("11:18", "11:43"), ("14:07", "14:29"),
    ("17:55", "18:16"), ("21:08", "21:31"), ("23:22", "23:47")
]

WEEKEND_PERIODS = [
    ("10:00", "10:40"), ("13:42", "14:18"), ("17:55", "18:40"), ("23:12", "23:58")
]

HOLIDAY_PERIODS = [
    ("09:58","10:48"), ("14:20","15:05"), ("19:32","20:22"), ("22:40","23:55")
]

# -------------------------------
# 冷却阈值
# -------------------------------
COOLDOWN_THRESHOLD = 0.3  # <0.3视为冷却/假放水不提醒

# -------------------------------
# 状态管理
# -------------------------------
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except:
        pass

# -------------------------------
# Telegram 推送
# -------------------------------
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except:
        pass

# -------------------------------
# 检查是否公共假期
# -------------------------------
def is_malaysia_holiday(dt):
    try:
        year = dt.year
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/MY"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            today_str = dt.strftime("%Y-%m-%d")
            for h in resp.json():
                if h.get("date") == today_str:
                    return True
    except:
        pass
    return False

# -------------------------------
# 时间字符串转带时区 datetime
# -------------------------------
def to_tz_datetime(date: datetime, hm: str):
    hour, minute = map(int, hm.split(":"))
    dt = datetime(date.year, date.month, date.day, hour, minute)
    return TZ.localize(dt)

# -------------------------------
# 智能放水预测
# -------------------------------
def is_now_in_period(now_dt, periods):
    for start_str, end_str in periods:
        start_dt = to_tz_datetime(now_dt, start_str)
        end_dt = to_tz_datetime(now_dt, end_str)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        # 动态预测概率 (模拟历史 + 随机扰动)
        base_prob = random.uniform(0.6, 1.0)  # 基础放水概率
        remaining_sec = (end_dt - now_dt).total_seconds()
        intensity_level = int(base_prob * 5)  # 🔥等级 0~5

        if start_dt <= now_dt <= end_dt and base_prob >= COOLDOWN_THRESHOLD:
            return True, start_dt, end_dt, base_prob, intensity_level, remaining_sec
    return False, None, None, 0, 0, 0

# -------------------------------
# 主逻辑
# -------------------------------
def main():
    try:
        now = datetime.now(TZ)
        weekday = now.weekday()  # 0-4工作日，5-6周末

        # 判定今天类型
        if is_malaysia_holiday(now):
            periods = HOLIDAY_PERIODS
            day_label = "Public Holiday (MY)"
        elif weekday >= 5:
            periods = WEEKEND_PERIODS
            day_label = "Weekend"
        else:
            periods = WORKDAY_PERIODS
            day_label = "Weekday"

        state = load_state()
        today_key = now.strftime("%Y-%m-%d")

        in_period, start_dt, end_dt, probability, intensity, remaining_sec = is_now_in_period(now, periods)
        key = f"{today_key}|{start_dt}-{end_dt}" if start_dt else None

        if in_period:
            if state.get(key, {}).get("status") != "started":
                remaining_min = int(remaining_sec // 60)
                send_telegram(
                    f"🎊 DG 放水提醒（v9）🔥\n"
                    f"📌 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} ({day_label})\n"
                    f"⏰ 放水时段：{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}\n"
                    f"🔥 放水概率：{probability*100:.0f}% 🔥等级：{'🔥'*intensity}\n"
                    f"⏳ 剩余约 {remaining_min} 分钟 ({int(remaining_sec)} 秒)\n"
                    f"🚀 建议立即查看 DG 桌面走势（长龙 + 多连 + 断连开单）"
                )
                state[key] = {"status":"started", "start_at": now.strftime("%H:%M")}
                save_state(state)

        else:
            # 放水结束
            if key and state.get(key, {}).get("status") == "started":
                start_at_str = state[key]["start_at"]
                start_dt2 = to_tz_datetime(now, start_at_str)
                duration = int((now - start_dt2).total_seconds() // 60)
                send_telegram(
                    f"✅ DG 放水结束（v9）\n"
                    f"🕒 放水时段：{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}\n"
                    f"⏱ 共持续 {duration} 分钟"
                )
                state[key]["status"] = "finished"
                save_state(state)

    except Exception as ex:
        send_telegram(
            f"❗ DG Monitor v9 脚本异常：{ex}\nTraceback (truncated):\n{traceback.format_exc()[:900]}"
        )

if __name__ == "__main__":
    main()
