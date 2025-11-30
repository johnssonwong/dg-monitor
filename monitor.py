import os
import datetime
import random
import requests

# Telegram 配置
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# 放水时段配置（历史数据概率预测）
# 示例：你可以自行根据历史数据调整时间段和强度
# 时间格式：('开始时间', '结束时间', '强度')，强度 'high' = 2🔥，'medium' = 1🔥
WEEKDAY_PERIODS = [
    ('10:00', '12:00', 'high'),
    ('14:00', '16:00', 'medium'),
    ('20:00', '22:00', 'high')
]

WEEKEND_PERIODS = [
    ('11:00', '13:00', 'high'),
    ('15:00', '17:00', 'medium'),
    ('21:00', '23:00', 'high')
]

HOLIDAY_PERIODS = [
    ('10:00', '12:00', 'high'),
    ('14:00', '16:00', 'high'),
    ('20:00', '22:00', 'high')
]

# 模拟平台历史胜率 / 放水概率
def get_platform_win_rate():
    # 高峰期概率低一点，低峰期概率高一点
    hour = datetime.datetime.now().hour
    if 11 <= hour <= 14 or 20 <= hour <= 22:
        return random.uniform(0.7, 0.85)  # 强放水
    else:
        return random.uniform(0.55, 0.7)   # 中等放水

# 判断是否在放水时间段
def is_in_period(periods):
    now = datetime.datetime.now().time()
    for start_str, end_str, strength in periods:
        start = datetime.datetime.strptime(start_str, "%H:%M").time()
        end = datetime.datetime.strptime(end_str, "%H:%M").time()
        if start <= now <= end:
            return strength
    return None

# 判断今天属于哪类日子
def get_today_periods():
    today = datetime.datetime.today()
    weekday = today.weekday()
    # 可根据你自己设置的节假日名单判断
    holidays = []  # 例: ['2025-12-25', '2025-01-01']
    if today.strftime("%Y-%m-%d") in holidays:
        return HOLIDAY_PERIODS
    elif weekday < 5:
        return WEEKDAY_PERIODS
    else:
        return WEEKEND_PERIODS

# 模拟判断牌桌策略（长连、多连、断连开单）
def evaluate_table_strategy():
    # 模拟结果：True=可入场，False=断连开单或不可入场
    outcome = random.choices(
        ['long_streak', 'multi_streak', 'break_single', 'empty_table'],
        weights=[0.3, 0.2, 0.3, 0.2],
        k=1
    )[0]
    return outcome

# 发送 Telegram 消息
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram 配置未设置，无法发送消息")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': msg,
        'parse_mode': 'HTML'
    }
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print("Telegram 发送异常:", e)

def main():
    periods = get_today_periods()
    strength = is_in_period(periods)
    if strength:
        # 平台胜率预测
        win_rate = get_platform_win_rate()
        table_status = evaluate_table_strategy()
        emoji = '🔥🔥' if strength == 'high' else '🔥'
        msg = f"💰 放水预测开始\n时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"强度: {strength} {emoji}\n"
        msg += f"平台胜率参考: {win_rate:.2f}\n"
        msg += f"入场策略判断: {table_status}\n"
        if table_status == 'break_single':
            msg += "⚠️ 当前桌断连开单，请寻找下一桌"
        send_telegram(msg)
        print(msg)
    else:
        print(f"{datetime.datetime.now()}: 当前不在放水时段，无需提醒。")

if __name__ == "__main__":
    main()
