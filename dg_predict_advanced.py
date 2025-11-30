# dg_predict_advanced.py
# 预测模型 + 公共假期检测 + 随机 + 模拟 + Telegram 提醒
# 每次被 GitHub Actions 触发即可，适合全天候 24/7 运行

import os
import json
import random
import time
from datetime import datetime, timedelta, timezone
import requests

### === 配置区域，请你填入你的真实 Telegram Bot Token 与 Chat ID ===
TELEGRAM_BOT_TOKEN = "<YOUR_TELEGRAM_BOT_TOKEN>"
TELEGRAM_CHAT_ID = "<YOUR_CHAT_ID>"
### ================================================================

# 时区设为马来西亚 UTC+8
TZ = timezone(timedelta(hours=8))

# 公共假期 API URL 模板 (Nager.Date 公共假期 API for Malaysia)
HOLIDAY_API = "https://date.nager.at/api/v3/PublicHolidays/{year}/MY"

STATE_FILE = "state_advanced.json"

# 时间段 + 基础分数 (score) —— 可按经验调整
TIME_SLOTS = [
    # (start_h, start_m, end_h, end_m, base_score)
    (2,   0, 3,  0, 78),   # 凌晨
    (9,  32, 9,  52, 72),  # 09:32–09:52
    (13, 30, 13, 50, 68),
    (16,  0, 16, 20, 60),
    (23, 30, 23, 50, 75),
    # fallback /低分段 — 保留以覆盖全天
    (0,   0, 2,   0, 30),
    (3,   0, 9,  32, 35),
    (10,  0, 13, 30, 40),
    (14,  0, 16,  0, 45),
    (17,  0, 23, 30, 50),
]

THRESHOLD_STRONG = 75
THRESHOLD_MEDIUM = 50

# 放水 / 中等胜率 时段模拟的持续时间区间 (分钟)
DURATION_MIN = 12
DURATION_MAX = 35

# 提前预警时间 (分钟) —— 在结束前多少分钟发送预警
PREWARN_MIN = 5

# ------------------ STATE 管理 ------------------

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    # 初始状态
    return {
        "alert": None,
        "holidays": {}  # 缓存假期列表 per year
    }

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def fetch_holidays(year):
    url = HOLIDAY_API.format(year=year)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # 返回 date 字符串集合 e.g. "2025-12-25"
            return { item["date"] for item in data }
    except Exception as e:
        print("fetch holidays error:", e)
    return set()

def is_holiday(dt, state):
    y = dt.year
    holidays = state.get("holidays", {}).get(str(y))
    if holidays is None:
        holidays = fetch_holidays(y)
        state.setdefault("holidays", {})[str(y)] = list(holidays)
        save_state(state)
    return dt.strftime("%Y-%m-%d") in holidays

# ------------------ 时间 / 概率 模型 ------------------

def get_time_slot_score(dt, state):
    hhmm = dt.hour * 60 + dt.minute
    base = None
    for slot in TIME_SLOTS:
        sh, sm, eh, em, score = slot
        start = sh * 60 + sm
        end = eh * 60 + em
        if end <= start:
            # 跨午夜
            if hhmm >= start or hhmm < end:
                base = score
                break
        else:
            if start <= hhmm < end:
                base = score
                break
    if base is None:
        base = 30  # 默认低分
    # 周末加权
    if dt.weekday() >= 5:
        base += 8
    # 假期加权
    if is_holiday(dt, state):
        base += 12
    # cap
    return min(base, 95)

def compute_combined_score(base_score):
    # 随机浮动 +/- 8
    return base_score + random.randint(-8, 8)

# ------------------ Telegram 通知 ------------------

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=15)
    except Exception as e:
        print("Telegram send error:", e)

# ------------------ 主逻辑 ------------------

def main():
    state = load_state()
    now = datetime.now(TZ)

    base = get_time_slot_score(now, state)
    combined = compute_combined_score(base)

    alert = state.get("alert")

    # 若已有 alert，则检查是否接近结束或结束
    if alert:
        end_time = datetime.fromisoformat(alert["end_time"])
        # 结束判断
        if now >= end_time:
            start_time = datetime.fromisoformat(alert["start_time"])
            dur = int((end_time - start_time).total_seconds() / 60)
            send_telegram(f"✅ <b>DG 模型 — 放水/高胜率 结束</b>\n"
                          f"类型: {alert['type']}  理由: {alert.get('reason','')}\n"
                          f"开始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                          f"结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                          f"持续: {dur} 分钟")
            state["alert"] = None
            save_state(state)
        else:
            # 预警判断
            if (end_time - now) <= timedelta(minutes=PREWARN_MIN):
                if not alert.get("prewarn_sent"):
                    mins_left = int((end_time - now).total_seconds() / 60)
                    send_telegram(f"⚠️ <b>DG 模型 — 放水即将结束</b>\n"
                                  f"类型: {alert['type']}\n"
                                  f"预计结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                  f"剩余: {mins_left} 分钟\n"
                                  f"理由: {alert.get('reason','')}")
                    alert["prewarn_sent"] = True
                    state["alert"] = alert
                    save_state(state)
        return

    # 没有 alert，决定是否开启一个新的放水/中上胜率期
    if combined >= THRESHOLD_STRONG:
        level = "强放水🔥🔥"
    elif combined >= THRESHOLD_MEDIUM:
        level = "中等胜率🟡"
    else:
        # 回避时段，不提醒
        return

    # 随机确定持续时长
    duration = random.randint(DURATION_MIN, DURATION_MAX)
    end_time = now + timedelta(minutes=duration)
    prob = min(99, math.floor((combined / 120.0) * 100))

    alert = {
        "type": level,
        "start_time": now.isoformat(),
        "end_time": end_time.isoformat(),
        "reason": f"combined_score={combined}, base={base}",
        "prewarn_sent": False
    }
    state["alert"] = alert
    save_state(state)

    send_telegram(f"{level} 已开始\n"
                  f"📈 胜率概率估计: {prob}%\n"
                  f"🕒 预计结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"⏳ 预计持续: {duration} 分钟\n"
                  f"📍 类型: {level}\n"
                  f"说明: 基于时间段 + 公共假期 + 随机 + 模型预测")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 若脚本发生异常，也发通知
        try:
            send_telegram(f"⚠️ DG 模型监测脚本 出错: {e}")
        except:
            pass
        raise
