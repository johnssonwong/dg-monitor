# telegram_reminder.py
import requests
from datetime import datetime, timedelta
import pytz
import json
import os
import traceback

# -----------------------
# 你提供的配置（请勿改动，已按你给的填入）
BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"
REPO_NAME = "dg-monitor"   # 仅供记录，实际不用于 API
TIMEZONE = "Asia/Kuala_Lumpur"
# -----------------------

# -----------------------
# A (稳健) 与 B (进攻) 时间窗（GMT+8）
# A 必备（更严格）
A_PERIODS = [
    ("09:28", "10:05"),
    ("15:26", "16:10"),
    ("20:33", "21:22"),
]

# B 补充（进攻）
B_PERIODS = [
    ("10:47", "11:33"),
    ("13:42", "14:18"),
    ("17:55", "18:40"),
    ("23:12", "23:58"),
    ("00:00", "00:40"),
]

# Weekday/Weekend/Holiday split:
# We'll map weekday vs weekend and holidays to sets:
WEEKDAY_PERIODS = A_PERIODS + B_PERIODS
WEEKEND_PERIODS = [
    # weekend tuned windows: we keep both A/B but weekend can emphasize different windows
    ("10:00","10:40"),
    ("13:42","14:18"),
    ("17:55","18:40"),
    ("23:12","23:58"),
]
HOLIDAY_PERIODS = [
    # Slightly expanded windows on holidays (as discussed)
    ("09:58","10:48"),
    ("14:20","15:05"),
    ("19:32","20:22"),
    ("22:40","23:55"),
]

# -----------------------
STATE_PATH = "state.json"

# Robust load/save
def load_state():
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    # default structure:
    return {
        # keys: "YYYY-MM-DD|HH:MM-HH:MM" -> {"status":"started"/"finished", "start_at":"HH:MM"}
    }

def save_state(state):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass  # do not raise

# Telegram helper (safe)
def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception:
        # swallow errors - do not allow script to crash
        pass

# holiday detection via Nager.Date (public holidays API)
def is_malaysia_holiday(dt):
    try:
        year = dt.year
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/MY"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            holidays = resp.json()
            today_str = dt.strftime("%Y-%m-%d")
            for h in holidays:
                if h.get("date") == today_str:
                    return True
    except Exception:
        # network or API error -> fallback False
        pass
    return False

# time helpers
def parse_time(hm_str, ref_date):
    return datetime.strptime(hm_str, "%H:%M").replace(year=ref_date.year, month=ref_date.month, day=ref_date.day)

def is_now_in_period(now_dt, start_str, end_str):
    start_dt = parse_time(start_str, now_dt)
    end_dt = parse_time(end_str, now_dt)
    if end_dt < start_dt:
        # spans midnight
        end_dt += timedelta(days=1)
    return start_dt <= now_dt <= end_dt, start_dt, end_dt

def main():
    try:
        tz = pytz.timezone(TIMEZONE)
        now = datetime.now(tz)
        today_key = now.strftime("%Y-%m-%d")
        weekday = now.weekday()  # 0..6

        state = load_state()

        # choose which periods apply today
        if is_malaysia_holiday(now):
            periods = HOLIDAY_PERIODS
            day_label = "Public Holiday (MY)"
        elif weekday >= 5:
            periods = WEEKEND_PERIODS
            day_label = "Weekend"
        else:
            periods = WEEKDAY_PERIODS
            day_label = "Weekday"

        # For traceability, send a light heartbeat only every hour (not every run)
        # We'll send once at minute 0 of each hour to confirm system alive
        if now.minute == 0:
            send_telegram(f"⚡ DG Monitor heartbeat — {now.strftime('%Y-%m-%d %H:%M')} ({day_label})")

        # loop periods, decide start/finish
        for start_str, end_str in periods:
            key = f"{today_key}|{start_str}-{end_str}"
            in_period, start_dt, end_dt = is_now_in_period(now, start_str, end_str)
            if in_period:
                # If not started yet -> send start message and store start time
                if state.get(key) != "started":
                    remaining_min = int((end_dt - now).total_seconds() // 60)
                    # choose emoji level by whether it is in strict A or in B
                    level = "🔥🔥🔥" if (start_str, end_str) in A_PERIODS else "🔥🔥"
                    text = (
                        f"🎊 DG 放水检测提醒 {level}\n"
                        f"🕒 时段：{start_str} - {end_str} (GMT+8)\n"
                        f"⏳ 预计结束：{end_dt.strftime('%H:%M')}\n"
                        f"🔥 剩余约 {remaining_min} 分钟\n"
                        f"📌 建议：按你的【追连/断连开单】策略入场（监测所有桌）"
                    )
                    send_telegram(text)
                    # record start minute string
                    state[key] = {"status": "started", "start_at": now.strftime("%H:%M")}
                    save_state(state)
                else:
                    # already started — but we also can update remaining every X minutes to help realtime
                    # to avoid message spam, only send update if remaining changed significantly:
                    # store last_remain in state to compare
                    last_remain = state.get(key, {}).get("last_remain")
                    remaining_min = int((end_dt - now).total_seconds() // 60)
                    if last_remain is None or remaining_min != last_remain:
                        # send lightweight update every time remaining decreases by at least 1 minute
                        # but guard: only update if remaining_min % 5 == 0 or remaining_min <= 3
                        if remaining_min % 5 == 0 or remaining_min <= 3:
                            update_text = f"⏱ 放水进行中 — 剩余约 {remaining_min} 分钟 (时段 {start_str}-{end_str})"
                            send_telegram(update_text)
                        # write last_remain
                        state[key]["last_remain"] = remaining_min
                        save_state(state)
            else:
                # not in period
                if state.get(key, {}).get("status") == "started":
                    # we've just transitioned out — send end notification with real duration
                    start_at = state[key].get("start_at")
                    try:
                        start_dt_local = parse_time(start_at, now)
                    except Exception:
                        # fallback: start at scheduled start time
                        start_dt_local = parse_time(start_str, now)
                    # duration minutes: end_dt - start_dt_local (clamped)
                    # end_dt variable computed with now date; recalc end_dt for period
                    _, scheduled_start, scheduled_end = is_now_in_period(now - timedelta(minutes=1), start_str, end_str)
                    duration = int((scheduled_end - datetime.strptime(state[key].get("start_at", scheduled_start.strftime("%H:%M")), "%H:%M").replace(year=now.year, month=now.month, day=now.day)).total_seconds() // 60)
                    end_text = (
                        f"✅ DG 放水时段结束\n"
                        f"🕒 时段：{start_str} - {end_str}\n"
                        f"⏱ 持续约 {duration} 分钟\n"
                    )
                    send_telegram(end_text)
                    # mark finished
                    state[key]["status"] = "finished"
                    save_state(state)

        # successful finish
    except Exception as ex:
        # Never fail the run; notify error (but do not raise)
        try:
            send_telegram(f"❗ DG Monitor 脚本捕获异常，已忽略并继续：{str(ex)}")
            # optional: include traceback trimmed
            tb = traceback.format_exc()
            if tb:
                # send truncated traceback if small
                send_telegram(f"Traceback (truncated):\n{tb[:1000]}")
        except Exception:
            pass
    finally:
        # ensure state is saved (best-effort)
        try:
            save_state(load_state())  # reload current safe state and save (no-op likely)
        except:
            pass

if __name__ == "__main__":
    main()
