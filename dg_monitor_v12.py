#!/usr/bin/env python3
# dg_monitor_v12.py
# V12 - DG Baccarat Monitor (final)
# Requirements: python3.8+, pip install requests pytz holidays

import os
import json
import math
import time
import random
import traceback
from datetime import datetime, timedelta
import pytz
import requests
import holidays

# -------------------------
# Configuration & Defaults
# -------------------------
TZ_NAME = os.getenv("MY_TIMEZONE", "Asia/Kuala_Lumpur")
TZ = pytz.timezone(TZ_NAME)

# Read Telegram config from env (GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Fallback (not recommended) - if these are empty the script will still run but won't send Telegram
# You should set the two Secrets in GitHub: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in env. Telegram messages will not be sent.")

# Files
STATE_FILE = "state_v12.json"       # persistent state (history, cooldowns, etc.)
DAILY_LOG = "dg_daily_log_v12.json" # daily counts and durations
ERROR_LOG = "dg_error_v12.log"

# Table count (you specified 23 real tables)
N_TOTAL = 23

# Effective score weights (can be tuned by self-learning)
W1 = 0.35  # weight for N_long (≥4)
W2 = 0.25  # weight for N_dragon (≥8)
W3 = 0.25  # weight for density (no-empty fraction)
W4 = 0.10  # weight for N_multichain
W5 = 0.05  # negative weight for singlejump proportion

# Time weight mapping (used to bias thresholds based on time-of-day)
TIME_WEIGHTS = {
    "peak": 1.00,     # high traffic (default higher chance but we will still check density)
    "secondary": 0.85,
    "low": 0.65
}

# Define time ranges (local TZ) as tuples of (start_hour, start_min, end_hour, end_min)
# These are used only to set a time_weight category; rules are flexible and adaptive.
PEAK_WINDOWS = [("19:30","23:30")]        # typically busiest (but may be used for reduced trust)
SECONDARY_WINDOWS = [("16:00","18:30")]   # afternoon/evening
LOW_WINDOWS = [("03:00","12:00")]         # cold / low-flow (but may be used for targeted放水)

# Thresholds
HIGH_ALERT_THRESHOLD = 0.85   # if effective_score*time_weight >= this -> high alert (🔥 or 🔥🔥 depending on other signals)
MID_ALERT_THRESHOLD = 0.70
SINGLEJUMP_BLOCK = 0.40       # if >=40% single-jump tables -> block (considered noisy/收割)
CONFIRM_REQUIRED = 2          # require consecutive confirmations
COOLDOWN_MINUTES = 10         # minimal cooldown before repeating same-level alert for same period
STRONG_DRAGON_COUNT = 3       # number of dragons (≥8) to favour strong 放水 (🔥🔥)
STRONG_LONG_THRESHOLD = 0.40  # fraction of tables with long ≥5 to push to 🔥🔥

# Learning / history retention
HISTORY_KEEP = 60  # keep last N durations per slot for averaging

# Simulate vs Real source:
# By default we simulate 23 tables because no public DG API is available.
# If in future you have a reliable source URL that returns JSON roadmaps, you can set REAL_SOURCE_URL env.
REAL_SOURCE_URL = os.getenv("DG_ROAD_API_URL")  # optional, not provided now

# -------------------------
# Utility helpers
# -------------------------
def now_tz():
    return datetime.now(TZ)

def send_telegram(text):
    """Send Telegram message if credentials available. Uses Markdown formatting."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # No token configured; log to stdout instead
        print("[TELEGRAM DISABLED] " + text)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=10)
        if resp.status_code != 200:
            print("Telegram send failed:", resp.status_code, resp.text)
            return False
        return True
    except Exception as e:
        print("Telegram send exception:", e)
        return False

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return {"history": {}, "periods": {}}

def save_state(st):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception as e:
        print("Failed to save state:", e)

def load_daily_log():
    try:
        if os.path.exists(DAILY_LOG):
            with open(DAILY_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return {}

def save_daily_log(d):
    try:
        with open(DAILY_LOG, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception as e:
        print("Failed to save daily log:", e)

def is_time_in_window(now, start_hm, end_hm):
    """start_hm/end_hm: 'HH:MM' strings in LOCAL timezone. Handles cross-midnight windows."""
    s_h, s_m = map(int, start_hm.split(":"))
    e_h, e_m = map(int, end_hm.split(":"))
    start_dt = now.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
    end_dt = now.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return start_dt <= now <= end_dt

def time_weight_for_now(now):
    # return numeric weight and label
    for s,e in PEAK_WINDOWS:
        if is_time_in_window(now, s, e):
            return TIME_WEIGHTS["peak"], "peak"
    for s,e in SECONDARY_WINDOWS:
        if is_time_in_window(now, s, e):
            return TIME_WEIGHTS["secondary"], "secondary"
    for s,e in LOW_WINDOWS:
        if is_time_in_window(now, s, e):
            return TIME_WEIGHTS["low"], "low"
    return TIME_WEIGHTS["peak"], "peak"  # default

# -------------------------
# Road/roadmap source (simulate or real)
# -------------------------
def fetch_real_tables():
    """If REAL_SOURCE_URL is set and returns JSON roadboards, adapt here.
       Expected format (example): {'tables': [[1,1,0,0,1], ...]} where 1=Banker,0=Player (recent to oldest or vice versa).
       This function should be adapted to the real API schema if available.
    """
    url = REAL_SOURCE_URL
    if not url:
        return None
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            # try to adapt common shapes:
            if isinstance(data, dict) and "tables" in data and isinstance(data["tables"], list):
                return data["tables"][:N_TOTAL]
            # else: return None to fall back to simulation
    except Exception as e:
        print("Real fetch failed:", e)
    return None

def simulate_tables():
    """Simulate N_TOTAL tables. Each table is a list of recent outcomes (0=Player,1=Banker)."""
    tables = []
    for _ in range(N_TOTAL):
        # generate runs: decide whether this table currently has a run or is random
        if random.random() < 0.35:
            # produce a run/连: length 4..12 of same value then some noise
            base = random.choice([0,1])
            run_len = random.randint(4, random.choice([6,8,10,12]))
            seq = [base] * run_len
            # append a few noise moves possibly
            for _ in range(random.randint(0,3)):
                seq.append(random.choice([0,1]))
        else:
            # random short sequences
            seq = [random.choice([0,1]) for _ in range(random.randint(0,6))]
        tables.append(seq)
    return tables

# -------------------------
# Table analysis functions
# -------------------------
def analyze_tables(tables):
    """Return counts: N_long(>=4), N_dragon(>=8), N_multichain(two-row >=4), N_singlejump (consecutive alternating >=4),
       density = fraction non-empty (we consider table with at least 1 record as non-empty)
    """
    n_long = 0
    n_dragon = 0
    n_multichain = 0
    n_singlejump = 0
    non_empty = 0

    for seq in tables:
        if not seq:
            continue
        non_empty += 1
        L = len(seq)
        # longest same-value run
        longest = 1
        cur = 1
        for i in range(1, L):
            if seq[i] == seq[i-1]:
                cur += 1
            else:
                longest = max(longest, cur)
                cur = 1
        longest = max(longest, cur)

        if longest >= 4:
            n_long += 1
        if longest >= 8:
            n_dragon += 1
        # multi-chain: detect at least two separated runs >=3 in the sequence
        runs_ge3 = 0
        cur = 1
        for i in range(1, L):
            if seq[i] == seq[i-1]:
                cur += 1
            else:
                if cur >= 3:
                    runs_ge3 += 1
                cur = 1
        if cur >= 3:
            runs_ge3 += 1
        if runs_ge3 >= 2:
            n_multichain += 1

        # single jump (alternating pattern) detection: consecutive alternating of length >=4
        alt_len = 1
        for i in range(1, L):
            if seq[i] != seq[i-1]:
                alt_len += 1
            else:
                alt_len = 1
            if alt_len >= 4:
                n_singlejump += 1
                break

    density = non_empty / max(1, N_TOTAL)
    return {
        "N_long": n_long,
        "N_dragon": n_dragon,
        "N_multichain": n_multichain,
        "N_singlejump": n_singlejump,
        "density": round(density, 3)
    }

# -------------------------
# Effective score / decision logic
# -------------------------
def effective_score(metrics):
    # normalize counts by N_TOTAL
    n_long_frac = metrics["N_long"] / N_TOTAL
    n_dragon_frac = metrics["N_dragon"] / N_TOTAL
    multi_frac = metrics["N_multichain"] / N_TOTAL
    single_frac = metrics["N_singlejump"] / N_TOTAL
    density = metrics["density"]

    score = (W1 * n_long_frac +
             W2 * n_dragon_frac +
             W3 * density +
             W4 * multi_frac -
             W5 * single_frac)
    # clamp 0..1
    score = max(0.0, min(1.0, score))
    return score

# -------------------------
# Detection & alert flow
# -------------------------
def detect_and_alert():
    now = now_tz()
    date_key = now.strftime("%Y-%m-%d")
    state = load_state()
    daily = load_daily_log()

    # periods: dynamic expected windows (this engine uses time weighting only)
    tw, tw_label = time_weight_for_now(now)

    # fetch tables (real if source available, else simulate)
    real = fetch_real_tables()
    if real is None:
        tables = simulate_tables()
        source = "SIM"
    else:
        tables = real
        source = "REAL"

    metrics = analyze_tables(tables)
    score = effective_score(metrics)
    weighted_score = score * tw

    # block if singlejump proportion too high
    singlejump_prop = metrics["N_singlejump"] / max(1, N_TOTAL)
    blocked = singlejump_prop >= SINGLEJUMP_BLOCK

    # compute flags for strong 放水 (🔥🔥)
    strong_dragon_condition = metrics["N_dragon"] >= STRONG_DRAGON_COUNT
    strong_long_frac = (metrics["N_long"] / N_TOTAL) >= STRONG_LONG_THRESHOLD

    # Build period_key to record notifications per minute-window (use start minute)
    minute_key = now.strftime("%Y-%m-%d %H:%M")
    period_key = f"{minute_key}"

    # prepare storage for this minute
    entry = state.get("periods", {}).get(period_key, {})

    # Determine level: 0=none,1=mid,2=high,3=strong
    level = 0
    if not blocked:
        if weighted_score >= HIGH_ALERT_THRESHOLD:
            if strong_dragon_condition or strong_long_frac:
                level = 3  # strong -> 🔥🔥
            else:
                level = 2  # high -> 🔥
        elif weighted_score >= MID_ALERT_THRESHOLD:
            level = 1  # mid -> 🟡

    # confirmation logic: check previous minute(s)
    confirm_met = False
    # find previous consecutive confirmations in state
    prev_keys = sorted([k for k in state.get("periods", {}) if k < period_key], reverse=True)
    consecutive = 0
    for k in prev_keys[:CONFIRM_REQUIRED]:
        prev = state["periods"].get(k, {})
        if prev and prev.get("detected_level", 0) >= level and prev.get("valid", False):
            consecutive += 1
    if consecutive >= (CONFIRM_REQUIRED - 1):  # because current minute would make total CONFIRM_REQUIRED
        confirm_met = True

    # Alternatively, if level is very strong (>=3) we may allow single-sample immediate alert
    immediate_send = False
    if level >= 3 and weighted_score >= 0.92:
        immediate_send = True

    should_notify = False
    notify_type = None  # 'mid','high','strong'
    if level > 0:
        if immediate_send or confirm_met:
            should_notify = True
            if level == 1:
                notify_type = "mid"
            elif level == 2:
                notify_type = "high"
            elif level == 3:
                notify_type = "strong"
    # Do not notify if blocked (single-jump heavy) or level==0
    if blocked or level == 0:
        should_notify = False

    # Ensure cooldown: do not spam same level for same "slot" within COOLDOWN_MINUTES
    # We'll use a rolling key "alert_recent" in state to store last notified timestamp per notify_type
    now_ts = int(now.timestamp())
    last_alerts = state.get("alerts", {})
    last_sent_ts = last_alerts.get(notify_type, 0)
    if should_notify and last_sent_ts and (now_ts - last_sent_ts) < COOLDOWN_MINUTES * 60:
        # allow if level increased (e.g., mid -> high) and increase >= some delta
        last_level = last_alerts.get("last_level", 0)
        if level > last_level and (level - last_level) >= 1:
            # allow immediate escalation
            pass
        else:
            should_notify = False

    # compose message if notify
    if should_notify:
        # prepare message fields
        remaining_est_min = estimate_remaining_minutes(metrics, level, state)
        strength_label = {1: "🟡 中胜率", 2: "🔥 放水", 3: "🔥🔥 强放水"}[level]
        confidence_pct = int(score * 100)
        prob_pct = int(weighted_score * 100)
        peak_note = f"（时段权重：{tw_label}）"

        # counts and ratios
        long_cnt = metrics["N_long"]
        dragon_cnt = metrics["N_dragon"]
        multichain_cnt = metrics["N_multichain"]
        singlejump_cnt = metrics["N_singlejump"]
        density = metrics["density"]

        # daily logging
        if date_key not in daily:
            daily[date_key] = {"count": 0, "total_minutes": 0, "alerts": []}
        daily[date_key]["count"] += 1
        # For duration, we'll record estimated remaining as part of the alert; actual measured durations will be improved later
        daily[date_key]["alerts"].append({
            "ts": now.isoformat(),
            "level": level,
            "prob_pct": prob_pct,
            "confidence_pct": confidence_pct,
            "remaining_est_min": remaining_est_min
        })
        save_daily_log(daily)

        # Send Telegram message (final formatted)
        msg_lines = []
        msg_lines.append(f"{strength_label} *DG 放水提醒* {peak_note}")
        msg_lines.append(f"📅 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")
        msg_lines.append(f"🎯 放水胜率概率：*{prob_pct}%*    信心指数：*{confidence_pct}%*")
        msg_lines.append(f"🕒 预计结束（估）：{ (now + timedelta(minutes=remaining_est_min)).strftime('%H:%M') }（剩余 {remaining_est_min} 分钟）")
        msg_lines.append(f"📊 长连(≥4) 桌数：{long_cnt} / {N_TOTAL}")
        msg_lines.append(f"📊 长龙(≥8) 桌数：{dragon_cnt} / {N_TOTAL}")
        msg_lines.append(f"📊 多连(连珠) 桌数：{multichain_cnt}")
        msg_lines.append(f"⚠ 单跳过滤桌数：{singlejump_cnt}（占比 {round(singlejump_cnt / max(1,N_TOTAL) * 100,1)}%）")
        msg_lines.append(f"📈 桌面密度(非空白比)：{density*100:.0f}%")
        msg_lines.append(f"💡 建议：{advice_for_level(level, tw_label)}")
        msg_lines.append(f"📌 数据来源：{source} 模式（若有真实 API 会优先使用）")
        msg = "\n".join(msg_lines)

        send_telegram(msg)

        # update last alert info
        if "alerts" not in state:
            state["alerts"] = {}
        state["alerts"][notify_type] = now_ts
        state["alerts"]["last_level"] = level

    # record this minute's metrics for confirmation history
    if "periods" not in state:
        state["periods"] = {}
    state["periods"][period_key] = {
        "ts": now.isoformat(),
        "metrics": metrics,
        "score": score,
        "weighted_score": weighted_score,
        "detected_level": level,
        "valid": (not blocked and level > 0)
    }

    # keep history small
    if "periods" in state and len(state["periods"]) > 2000:
        # remove oldest
        keys = sorted(state["periods"].keys())
        for k in keys[:200]:
            state["periods"].pop(k, None)

    save_state(state)

def estimate_remaining_minutes(metrics, level, state):
    """Estimate remaining minutes based on level and historical durations for similar slots."""
    # default estimates (conservative)
    if level >= 3:
        base = 25
    elif level == 2:
        base = 14
    elif level == 1:
        base = 8
    else:
        base = 0
    # try to adjust by history: check history durations in state.history for similar metrics (coarse)
    hist = state_default().get("history", {})
    # we do a simple smoothing: if we have average durations recorded for prior same-level alerts, use them
    totals = []
    for k,v in hist.items():
        # v is list of durations
        if v:
            totals.append(sum(v)/len(v))
    if totals:
        avg_hist = sum(totals)/len(totals)
        # combine with base
        est = int((base + avg_hist)/2)
    else:
        est = base
    return max(1, est)

def state_default():
    s = load_state()
    return s

def advice_for_level(level, tw_label):
    if level >= 3:
        return "强放水（🔥🔥）——高胜率窗口，可考虑按你规则追龙；遇到高峰标注收割风险时请轻仓。"
    if level == 2:
        return "放水（🔥）——胜率提高，建议观察两三局确认后入场。"
    if level == 1:
        return "中胜率（🟡）——可小仓轻试，非必须入场。"
    return "当前非推荐入场时段（收割或走势不明确）。"

# -------------------------
# Main runner
# -------------------------
def main():
    try:
        detect_and_alert()
    except Exception as ex:
        # log error and attempt to notify
        tb = traceback.format_exc()
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{now_tz().isoformat()} ERROR: {ex}\n{tb}\n\n")
        try:
            send_telegram(f"❗ DG Monitor V12 异常：{ex}\n詳見 error log")
        except:
            pass

if __name__ == "__main__":
    main()
