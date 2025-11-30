import requests
import pytz
from datetime import datetime, timedelta
import random
import holidays

# -----------------------------
# CONFIG
# -----------------------------
TZ = pytz.timezone("Asia/Kuala_Lumpur")
MY_HOLIDAYS = holidays.MY()
TELEGRAM_BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"

# -----------------------------
# Telegram 发送函数
# -----------------------------
def send(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

# -----------------------------
# 模拟 23 桌 DG 真人桌面
# -----------------------------
def simulate_dg_tables():
    tables = []
    for i in range(23):
        streak_len = random.randint(0, 12)
        seq = [random.choice([0,1]) for _ in range(streak_len)]  # 0=闲,1=庄
        tables.append(seq)
    return tables

# -----------------------------
# 分析桌面
# -----------------------------
def analyze_tables(tables):
    long_streak = 0
    multi_streak = 0
    single_jump = 0
    for seq in tables:
        if len(seq)>=4 and all(x==seq[0] for x in seq):
            long_streak += 1
        for i in range(len(seq)-2):
            if seq[i]==seq[i+1]==seq[i+2]:
                multi_streak += 1
                break
        for i in range(len(seq)-3):
            if seq[i]!=seq[i+1]!=seq[i+2]!=seq[i+3]:
                single_jump += 1
                break
    return long_streak, multi_streak, single_jump

# -----------------------------
# 动态放水时段及概率
# -----------------------------
def get_expected_periods(dt):
    wd = dt.weekday()
    is_holiday = dt.date() in MY_HOLIDAYS
    base_periods = []

    if wd in [0,1,2,3,4]:
        base_periods += [
            ("06:58","07:23",0.82),
            ("09:31","09:59",0.85),
            ("11:45","12:20",0.78),
            ("14:02","14:28",0.65),
            ("17:35","18:15",0.73),
            ("21:10","21:47",0.88),
            ("23:56","00:19",0.80)
        ]
    if wd in [5,6]:
        base_periods += [
            ("08:05","08:42",0.88),
            ("10:26","11:03",0.92),
            ("13:40","14:15",0.80),
            ("16:55","17:32",0.87),
            ("22:18","23:00",0.91)
        ]
    if is_holiday:
        base_periods += [
            ("07:33","08:40",0.90),
            ("10:10","10:58",0.95),
            ("15:18","15:59",0.87),
            ("20:35","21:25",0.93)
        ]

    final_periods=[]
    for start,end,prob in base_periods:
        s_h, s_m=map(int,start.split(":"))
        e_h, e_m=map(int,end.split(":"))
        drift_s=random.randint(-2,2)
        drift_e=random.randint(-2,2)
        start_dt=dt.replace(hour=s_h,minute=s_m,second=0)+timedelta(minutes=drift_s)
        end_dt=dt.replace(hour=e_h,minute=e_m,second=0)+timedelta(minutes=drift_e)
        final_periods.append((start_dt,end_dt,prob))
    return final_periods

# -----------------------------
# 信心指数计算
# -----------------------------
def calculate_confidence(long_cnt,multi_cnt,single_filter):
    score = long_cnt*5 + multi_cnt*3 - single_filter*2
    score = max(0,min(score,100))
    return score

# -----------------------------
# 主程序
# -----------------------------
def main():
    now=datetime.now(TZ)
    periods=get_expected_periods(now)

    in_fangshui=False
    current_prob=0
    end_time=None

    for start,end,prob in periods:
        if start <= now <= end:
            in_fangshui=True
            current_prob=prob
            end_time=end
            break

    if in_fangshui:
        tables=simulate_dg_tables()
        long_cnt,multi_cnt,single_filter=analyze_tables(tables)
        confidence=calculate_confidence(long_cnt,multi_cnt,single_filter)

        if current_prob>=0.80:
            if current_prob>0.88:
                intensity="🔥 强"
            else:
                intensity="✨ 中"
        else:
            intensity="⚠️ 弱"

        remaining=int((end_time-now).total_seconds()/60)
        next_start = end_time + timedelta(minutes=random.randint(15,60))
        next_end = next_start + timedelta(minutes=random.randint(20,40))

        msg=(
            f"{intensity} *DG 放水高胜率提醒*\n"
            f"📅 当前时间：{now.strftime('%H:%M')}\n"
            f"🕒 放水预计结束：{end_time.strftime('%H:%M')}（剩余 {remaining} 分钟）\n"
            f"🎯 放水等级概率：{int(current_prob*100)}%\n"
            f"📊 信心指数：{confidence}%\n"
            f"📊 长龙桌数：{long_cnt} 张\n"
            f"📊 多连桌数：{multi_cnt} 张\n"
            f"⚠ 单跳过滤桌数：{single_filter} 张\n"
            f"💡 建议：符合放水规则，可手动入场追龙\n"
            f"⏱ 预计下一放水窗口：{next_start.strftime('%H:%M')} ~ {next_end.strftime('%H:%M')}"
        )
        send(msg)

if __name__ == "__main__":
    main()
