import time
import math
import datetime
import requests
import json

# =======================
# 用户配置区
# =======================
TELEGRAM_TOKEN = "你的TelegramBotToken"
CHAT_ID = "你的ChatID"

# 放水强度标记 🔥 🔥 / 🔥
STRONG_SIGNAL = "🔥 🔥"
NORMAL_SIGNAL = "🔥"

# 周一至周五 / 周末 / 公共假期时间段（示例，可调整）
TIME_WINDOWS = {
    "weekday": [("09:00", "12:00"), ("14:00", "17:00"), ("20:00", "23:00")],
    "weekend": [("10:00", "13:00"), ("15:00", "18:00"), ("21:00", "23:30")],
    "holiday": [("10:00", "14:00"), ("16:00", "20:00")],
}

# 高峰期标记
HIGH_PEAK_HOURS = [("20:00", "23:00")]

# =======================
# 辅助函数
# =======================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram发送失败: {e}")

def is_time_in_window(window_start, window_end, now=None):
    if not now:
        now = datetime.datetime.now().time()
    start = datetime.datetime.strptime(window_start, "%H:%M").time()
    end = datetime.datetime.strptime(window_end, "%H:%M").time()
    return start <= now <= end

def get_current_period():
    today = datetime.datetime.today()
    weekday = today.weekday()
    is_holiday = False  # 假设你有方式判断是否为公共假期，可扩展
    if is_holiday:
        return "holiday"
    elif weekday < 5:
        return "weekday"
    else:
        return "weekend"

# =======================
# 历史胜率/放水概率计算
# =======================
def predict_water_period():
    """
    根据历史胜率数据预测放水时段概率。
    返回：
        signal_strength (str): 强/普通放水 🔥 🔥 / 🔥
        probability (int): 放水概率百分比
    """
    # 这里依赖历史数据和概率预测
    # 示例：随机示范逻辑，可替换为真实历史统计
    import random
    p = random.randint(60, 99)  # 模拟概率
    signal = STRONG_SIGNAL if p >= 85 else NORMAL_SIGNAL
    return signal, p

# =======================
# 入场策略判定函数
# =======================
def evaluate_table(table_data):
    """
    table_data: list of rounds, 例如 ["B", "B", "P", "P", "P", "B", "P"]
    返回 True/False 是否入场
    """
    long_streak = 0
    previous = None
    for idx, outcome in enumerate(table_data):
        if previous == outcome:
            long_streak += 1
        else:
            long_streak = 1
        previous = outcome

        # 检查断连开单
        if long_streak == 1 and idx >= 1:
            # 前面长连断开后，连续单跳
            if table_data[idx-1] != outcome:
                # 满足断连开单条件，离开此台桌
                return False

        # 多连、长连等策略可在此扩展

    return True

# =======================
# 主循环
# =======================
def main():
    period_type = get_current_period()
    for window in TIME_WINDOWS[period_type]:
        start, end = window
        while is_time_in_window(start, end):
            # 预测放水
            signal, probability = predict_water_period()

            message = f"放水时段预测: {signal} 概率: {probability}%"
            send_telegram(message)

            # 模拟桌面数据检查
            tables = [
                ["B", "B", "B", "B", "P", "P", "B"],  # 示例桌面
                ["P", "P", "B", "P", "P", "P", "B"]
            ]

            for idx, table in enumerate(tables):
                can_enter = evaluate_table(table)
                if not can_enter:
                    send_telegram(f"桌{idx+1} 出现断连开单, 请离开此台桌。")
                else:
                    send_telegram(f"桌{idx+1} 满足入场策略，可以考虑入场。")

            # 提前提醒 & 持续提醒机制
            time.sleep(300)  # 每5分钟提醒一次

if __name__ == "__main__":
    main()
