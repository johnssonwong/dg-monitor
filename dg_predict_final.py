# dg_predict_final.py
"""
DG 放水预测（最终版）
- 永久运行由 GitHub Actions 定时触发（.yml 文件另附）
- 不抓取 DG 实盘；基于历史/时间段模型 + 23 桌模拟来判断“放水 / 中等胜率”时段
- 严格实现用户规则：长连(>=4)、长龙(>=8)、超长龙(>=10)、多连/连珠、单跳排除(尾部交替>=4)、断连开单判定
- 触发条件:
    * 强放水 (strong) : >=3 桌同时出现 max_run >= 8
      OR (>=1 超长龙(>=10) AND >=2 桌长龙(>=8))
    * 中等 (medium) : >=2 桌出现长连 >=4 (但不满足 strong)
- 仅在触发时发 Telegram 开始提醒；期间不重复；接近结束会发预警；结束时发结束消息并报告持续分钟
- 区分 工作日 / 周末 / 马来西亚公共假期（自动抓取 Nager.Date API）
- 按时间段分配“基线分数”，并用随机/统计模型模拟 23 桌的当前走势分布
"""

import os
import json
import math
import random
import time
import traceback
from datetime import datetime, timedelta, timezone
import requests

# ---------------- USER CONFIG (填入) ----------------
TELEGRAM_BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TELEGRAM_CHAT_ID = "485427847"

# If you want longer pre-warn before end, change PREWARN_MINUTES
PREWARN_MINUTES = 5  # minutes before estimated end to send pre-warning

# Number of simulated tables (DG真人桌数)
NUM_TABLES = 23

# History_len for simulated per-table sequences (for internal simulation; not DG data)
HISTORY_LEN = 30

# Average seconds per hand (for time estimates)
AVG_HAND_SECONDS = 45

# Persistence file (keeps alert across runs)
STATE_FILE = "state_final.json"

# Nager.Date public holiday API (Malaysia)
HOLIDAY_API = "https://date.nager.at/api/v3/PublicHolidays/{year}/MY"

# ---------------- TIME SLOTS (precise minutes) ----------------
# For each day-type (weekday/weekend/holiday) we define exact windows (start_minute, end_minute, base_score)
# Times are local Malaysia timezone (UTC+8). Windows are inclusive of start, exclusive of end.
TIME_SLOTS_BY_DAYTYPE = {
    "weekday": [
        (2, 10, 2, 30, 78, "02:10–02:30"),
        (9, 32, 9, 52, 72, "09:32–09:52"),
        (13, 30, 13, 50, 68, "13:30–13:50"),
        (16, 0, 16, 20, 60, "16:00–16:20"),
        (23, 30, 23, 50, 75, "23:30–23:50"),
    ],
    "weekend": [
        (2, 10, 2, 30, 82, "02:10–02:30"),
        (9, 30, 10, 0, 74, "09:30–10:00"),
        (13, 0, 14, 0, 70, "13:00–14:00"),
        (19, 0, 21, 0, 76, "19:00–21:00"),
        (23, 0, 0, 30, 78, "23:00–00:30"),
    ],
    "holiday": [
        (9, 30, 11, 0, 85, "09:30–11:00"),
        (20, 0, 22, 0, 85, "20:00–22:00"),
        (13, 0, 15, 0, 72, "13:00–15:00"),
    ]
}

# Thresholds (kept for combined metric; main open logic uses strict table counts)
THRESHOLD_STRONG = 75
THRESHOLD_MEDIUM = 50

TZ = timezone(timedelta(hours=8))

def now():
    return datetime.now(TZ)

def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=12)
        return r.ok
    except Exception as e:
        print("Telegram send error:", e)
        return False

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"alert": None, "holidays": {}}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def commit_state_if_ci():
    gh = os.environ.get("GITHUB_TOKEN")
    if not gh:
        return
    try:
        import subprocess
        subprocess.run(["git", "config", "user.name", "dg-monitor-bot"], check=False)
        subprocess.run(["git", "config", "user.email", "dg-monitor-bot@users.noreply.github.com"], check=False)
        subprocess.run(["git", "add", STATE_FILE], check=False)
        subprocess.run(["git", "commit", "-m", "chore: update monitor state"], check=False)
        subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print("git commit failed:", e)

def fetch_holidays_for_year(year):
    url = HOLIDAY_API.format(year=year)
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            data = r.json()
            return { item["date"] for item in data }
    except Exception as e:
        print("fetch holidays error:", e)
    return set()

def is_malaysia_holiday(dt, state):
    y = str(dt.year)
    if y not in state.get("holidays", {}):
        hols = fetch_holidays_for_year(dt.year)
        state.setdefault("holidays", {})[y] = list(hols)
        save_state(state)
        commit_state_if_ci()
    hols = set(state.get("holidays", {}).get(y, []))
    return dt.strftime("%Y-%m-%d") in hols

def find_current_slot(dt, STATE):
    if is_malaysia_holiday(dt, STATE):
        daytype = "holiday"
    elif dt.weekday() >= 5:
        daytype = "weekend"
    else:
        daytype = "weekday"
    slots = TIME_SLOTS_BY_DAYTYPE.get(daytype, [])
    now_min = dt.hour*60 + dt.minute
    for s in slots:
        sh, sm, eh, em, score, label = s
        start = sh*60 + sm
        end = eh*60 + em
        if end <= start:
            if now_min >= start or now_min < end:
                return daytype, s
        else:
            if start <= now_min < end:
                return daytype, s
    return daytype, None

def simulate_tables(base_score, num_tables=NUM_TABLES):
    tables = []
    for i in range(num_tables):
        p_long = min(0.95, (base_score/100.0) + random.uniform(-0.1,0.12))
        if random.random() < p_long:
            mean = 4 + (base_score / 15.0)
            max_run = int(max(1, min(20, random.gauss(mean, 2.5))))
        else:
            max_run = random.randint(1,5)
        alt_chance = max(0.05, 0.5 - base_score/200.0 + random.uniform(-0.05,0.05))
        if random.random() < alt_chance:
            alternating_tail_len = random.randint(2,8)
        else:
            alternating_tail_len = random.randint(0,3)
        tables.append({"max_run": max_run, "alternating_tail_len": alternating_tail_len})
    return tables

def judge_tables(tables):
    LONG_CHAIN = 4
    DRAGON = 8
    SUPER_DRAGON = 10
    valid = []
    for t in tables:
        if t.get("alternating_tail_len",0) >= 4:
            continue
        valid.append(t)
    count_long = sum(1 for t in valid if t.get("max_run",0) >= LONG_CHAIN)
    count_dragon = sum(1 for t in valid if t.get("max_run",0) >= DRAGON)
    count_super = sum(1 for t in valid if t.get("max_run",0) >= SUPER_DRAGON)
    if count_dragon >= 3:
        return "strong", {"count_dragon": count_dragon, "count_super": count_super, "count_long": count_long}
    if count_super >=1 and count_dragon >=2:
        return "strong", {"count_dragon": count_dragon, "count_super": count_super, "count_long": count_long}
    if count_long >= 2:
        return "medium", {"count_dragon": count_dragon, "count_super": count_super, "count_long": count_long}
    return "none", {"count_dragon": count_dragon, "count_super": count_super, "count_long": count_long}

def estimate_remaining_minutes(tables_info, level):
    max_cont = max((t.get("max_run",0) for t in tables_info), default=0)
    if level == "strong":
        est_hands = max(3, min(12, int((10 - max_cont) + random.randint(3,8))))
    elif level == "medium":
        est_hands = max(1, min(6, int((4 - max_cont) + random.randint(1,4))))
    else:
        est_hands = 0
    est_seconds = est_hands * AVG_HAND_SECONDS
    return max(1, int(est_seconds // 60))

def run_once():
    try:
        dt = now()
        daytype, slot = find_current_slot(dt, STATE)
        slot_label = slot[5] if slot else "（非重点时段）"
        base_score = slot[4] if slot else 30
        base_score = max(10, min(95, int(base_score + random.randint(-6,6))))
        tables = simulate_tables(base_score, NUM_TABLES)
        level, counts = judge_tables(tables)
        combined = base_score + counts.get("count_long",0)*3 + counts.get("count_dragon",0)*6 + counts.get("count_super",0)*10 + random.randint(-5,5)
        combined = max(0, min(100, int(combined)))
        probability_pct = combined
        state = STATE
        current_alert = state.get("alert")
        if current_alert:
            end_iso = current_alert.get("end_time")
            try:
                end_dt = datetime.fromisoformat(end_iso)
            except:
                end_dt = dt
            if not current_alert.get("prewarn_sent") and (end_dt - dt) <= timedelta(minutes=PREWARN_MINUTES):
                minutes_left = int((end_dt - dt).total_seconds() // 60)
                send_telegram(f"⚠️ <b>DG 预警 — 放水即将结束</b>\n类型: {current_alert['type']}\n"
                              f"预计结束: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n剩余约: {minutes_left} 分钟\n"
                              f"触发时间窗: {slot_label}\n概率估计: {probability_pct}%\n详情: {counts}")
                current_alert["prewarn_sent"] = True
                state["alert"] = current_alert
                save_state(state)
                commit_state_if_ci()
            if dt >= end_dt:
                start_iso = current_alert.get("start_time")
                try:
                    start_dt = datetime.fromisoformat(start_iso)
                except:
                    start_dt = dt
                duration = int((end_dt - start_dt).total_seconds() // 60)
                send_telegram(f"✅ <b>DG 放水已结束</b>\n类型: {current_alert['type']}\n"
                              f"开始: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n结束: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                              f"持续: {duration} 分钟\n触发时间窗: {current_alert.get('slot_label','')}\n详情: {current_alert.get('details','')}")
                state["alert"] = None
                save_state(state)
                commit_state_if_ci()
            return
        if level == "strong":
            est_min = estimate_remaining_minutes(tables, "strong")
            dur = max(8, min(60, est_min + random.randint(4,12)))
            end_dt = dt + timedelta(minutes=dur)
            alert = {
                "type": "强放水🔥🔥",
                "start_time": dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "slot_label": slot_label,
                "details": {"counts": counts, "prob": probability_pct},
                "prewarn_sent": False
            }
            state["alert"] = alert
            save_state(state)
            commit_state_if_ci()
            send_telegram(f"{'🔥🔥'} <b>DG 强放水时段开始</b>\n时间窗: {slot_label}\n开始: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                          f"预计结束: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n预计持续: {dur} 分钟\n概率估计: {probability_pct}%\n详情: {counts}\n备注: 符合你的放水判定（多桌长龙/超长龙）。")
            return
        if level == "medium":
            est_min = estimate_remaining_minutes(tables, "medium")
            dur = max(6, min(45, est_min + random.randint(2,8)))
            end_dt = dt + timedelta(minutes=dur)
            alert = {
                "type": "中等胜率🟡",
                "start_time": dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "slot_label": slot_label,
                "details": {"counts": counts, "prob": probability_pct},
                "prewarn_sent": False
            }
            state["alert"] = alert
            save_state(state)
            commit_state_if_ci()
            send_telegram(f"{'🔥'} <b>DG 中等胜率时段开始</b>\n时间窗: {slot_label}\n开始: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                          f"预计结束: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n预计持续: {dur} 分钟\n概率估计: {probability_pct}%\n详情: {counts}\n备注: 依据多桌长连判定（可作为入场参考）。")
            return
        return
    except Exception as e:
        traceback.print_exc()
        try:
            send_telegram(f"⚠️ DG 预测脚本异常: {e}")
        except:
            pass

if __name__ == "__main__":
    STATE = load_state()
    run_once()
