import time
import datetime
import requests
import math

# ---------------- 用户配置 ----------------
TELEGRAM_TOKEN = "你的TelegramBotToken"
CHAT_ID = "你的ChatID"
CHECK_INTERVAL = 300  # 每 5 分钟检查一次
HISTORICAL_DATA = "模拟历史胜率数据"  # 可以替换为你实际数据源
# 放水时间段配置（根据周一至周五、周末、公共假期）
TIME_WINDOWS = {
    "weekday": [("10:00", "12:00"), ("14:00", "16:00"), ("20:00", "22:00")],
    "weekend": [("11:00", "13:00"), ("15:00", "17:00"), ("21:00", "23:00")],
    "holiday": [("09:00", "11:00"), ("13:00", "15:00"), ("19:00", "21:00")]
}

# 模拟桌子数据结构
TABLES = [
    {"name": "桌1", "data": []},
    {"name": "桌2", "data": []},
    {"name": "桌3", "data": []},
]

# ---------------- 工具函数 ----------------
def is_holiday(date):
    # 可扩展公共假期逻辑
    return date.weekday() >= 5  # 暂时周六日算假期

def is_time_in_window(start, end):
    now = datetime.datetime.now().time()
    start_time = datetime.datetime.strptime(start, "%H:%M").time()
    end_time = datetime.datetime.strptime(end, "%H:%M").time()
    return start_time <= now <= end_time

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram发送失败: {e}")

def analyze_tables():
    # 模拟按照你的策略判断
    result = []
    for table in TABLES:
        data = table["data"]
        # 这里用随机数据/概率模拟
        combined = 60  # 假设计算胜率的历史数据指标
        prob = min(99, math.floor((combined / 120.0) * 100))
        # 判断放水强弱
        if prob >= 80:
            emoji = "🔥🔥"
        elif prob >= 60:
            emoji = "🔥"
        else:
            emoji = ""
        result.append(f"{table['name']} 胜率: {prob}% {emoji}")
    return "\n".join(result)

# ---------------- 主循环 ----------------
def main():
    while True:
        now = datetime.datetime.now()
        weekday_type = "holiday" if is_holiday(now) else ("weekend" if now.weekday() >=5 else "weekday")
        windows = TIME_WINDOWS[weekday_type]

        for start, end in windows:
            if is_time_in_window(start, end):
                message = f"🎯 放水预测时间段: {start}-{end}\n{analyze_tables()}"
                send_telegram(message)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
