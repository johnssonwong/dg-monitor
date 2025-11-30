# monitor_predict.py
# Predictive DG monitor (model-based). Minimal deps: requests
# - 23-table simulation
# - Uses time-slot weights for weekdays / weekends / Malaysia public holidays
# - Implements your rules: 长连>=4, 龙>=8, 超龙>=10, 连珠/多连, 排除连续单跳>=4
# - Sends Telegram only on: START, PRE-WARN (near end), END
# - START is sent immediately when model opens a window (not tail-only)
# - State persisted in state_predict.json
# - Commit state back to repo if GITHUB_TOKEN present (for continuity)
#
# SECURITY: Your Telegram token/chat_id are prefilled as given. Keep repo private or rotate token if leaked.

import os, json, random, time, traceback, subprocess
from datetime import datetime, timedelta, timezone
import requests

# ---------------- CONFIG ----------------
TELEGRAM_BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TELEGRAM_CHAT_ID = "485427847"

STATE_FILE = "state_predict.json"
NUM_TABLES = 23
HISTORY_LEN = 30
AVG_HAND_SECONDS = 45

# Time slots: (start_h, start_m, end_h, end_m, base_score_weekday, base_score_weekend, base_score_holiday)
TIME_SLOTS = [
    # You specified these windows as important -> give them higher base scores
    (2, 0, 3, 0, 80, 86, 90),        # 02:00-03:00
    (9, 32, 9, 52, 75, 78, 82),      # 09:32-09:52 特别窗口
    (13, 30, 13, 50, 70, 74, 78),    # 13:30-13:50
    (16, 0, 16, 20, 62, 67, 72),     # 16:00-16:20
    (23, 30, 23, 50, 76, 80, 85),    # 23:30-23:50
]
# default low base for non-listed windows
DEFAULT_BASE_WEEKDAY = 30
DEFAULT_BASE_WEEKEND = 36
DEFAULT_BASE_HOLIDAY = 42

# thresholds
THRESH_STRONG = 75
THRESH_MEDIUM = 50

# chain rules
LONG_CHAIN = 4
DRAGON = 8
SUPER_DRAGON = 10
SINGLE_JUMP_EXCLUDE = 4

# duration range for predicted window (minutes)
DUR_MIN = 12
DUR_MAX = 36

# pre-warning lead time (minutes)
PREWARN_MIN = 5

# public holiday API (Nager.Date)
NAGER_API = "https://date.nager.at/api/v3/PublicHolidays/{year}/MY"

# emojis
EMOJI_STRONG = "🔥🔥"
EMOJI_MEDIUM = "🔥"
EMOJI_END = "✅"
EMOJI_WARN = "⚠️"

# HTTP
HTTP_TIMEOUT = 12
RETRIES = 1

TZ = timezone(timedelta(hours=8))

# ------------- helpers ----------------
def now(): return datetime.now(TZ)
def now_str(): return now().strftime("%Y-%m-%d %H:%M:%S %Z")
def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML"}, timeout=HTTP_TIMEOUT)
        return r.ok
    except Exception as e:
        print("tg send fail", e)
        return False

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE, "r", encoding="utf-8"))
        except:
            pass
    return {"alert": None, "tables": {}, "holidays": {}}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def commit_state():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    try:
        subprocess.run(["git", "config", "user.name", "dg-monitor-bot"], check=False)
        subprocess.run(["git", "config", "user.email", "dg-monitor-bot@users.noreply.github.com"], check=False)
        subprocess.run(["git", "add", STATE_FILE], check=False)
        subprocess.run(["git", "commit", "-m", "update state_predict.json"], check=False)
        subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print("git commit failed", e)

# public holiday fetch
def fetch_holidays(year):
    url = NAGER_API.format(year=year)
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code==200:
            data = r.json()
            return {d["date"] for d in data}
    except:
        pass
    return set()

def is_holiday(dt, state):
    yr = str(dt.year)
    if yr not in state.get("holidays", {}):
        holidays = list(fetch_holidays(dt.year))
        state.setdefault("holidays", {})[yr] = holidays
        save_state(state)
        commit_state()
    holidays = set(state.get("holidays", {}).get(yr, []))
    return dt.strftime("%Y-%m-%d") in holidays

# init tables
def init_tables(state):
    tables = state.get("tables", {})
    for i in range(1, NUM_TABLES+1):
        k = f"t{i}"
        if k not in tables:
            tables[k] = simulate_initial()
    state["tables"] = tables
    return tables

def simulate_initial():
    seq=[]
    while len(seq)<HISTORY_LEN:
        typ = random.choices(["run","alt","rand"], weights=[0.5,0.25,0.25])[0]
        if typ=="run":
            side = random.choice(["Z","X"])
            l = random.randint(2,7)
            seq += [side]*l
        elif typ=="alt":
            l = random.randint(2,6)
            for j in range(l):
                seq.append("Z" if j%2==0 else "X")
        else:
            seq.append(random.choice(["Z","X"]))
    return seq[-HISTORY_LEN:]

def simulate_next(seq, time_score):
    last = seq[-1]
    # current run
    run=1
    for s in reversed(seq[:-1]):
        if s==last: run+=1
        else: break
    base = 0.38 + (time_score/220.0) + min(0.35, run*0.03)
    base = max(0.05, min(0.95, base))
    return last if random.random() < base else ("Z" if last=="X" else "X")

# analyze sequence
def analyze(seq):
    if not seq: return {}
    last=seq[-1]
    cont=1
    for s in reversed(seq[:-1]):
        if s==last: cont+=1
        else: break
    max_run=1; cur=1
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1]:
            cur+=1
            if cur>max_run: max_run=cur
        else:
            cur=1
    alt=1
    for L in range(2, min(len(seq),12)+1):
        tail=seq[-L:]; ok=True
        for i in range(1,len(tail)):
            if tail[i]==tail[i-1]:
                ok=False; break
        if ok: alt=L
        else: break
    return {"latest":last,"current_continuous":cont,"max_run":max_run,"alternating_tail_len":alt,"seq":seq[-HISTORY_LEN:]}

def judge_all(infos):
    valid=[]
    for t in infos:
        if t.get("alternating_tail_len",0) >= SINGLE_JUMP_EXCLUDE:
            continue
        valid.append(t)
    count_long=sum(1 for t in valid if t.get("max_run",0) >= LONG_CHAIN)
    count_dragon=sum(1 for t in valid if t.get("max_run",0) >= DRAGON)
    count_super=sum(1 for t in valid if t.get("max_run",0) >= SUPER_DRAGON)
    if count_dragon >=3: return {"level":"strong","reason":f"{count_dragon}桌 ≥{DRAGON} (长龙)"}
    if count_super>=1 and count_dragon>=2: return {"level":"strong","reason":f"{count_super} 超龙 + {count_dragon} 龙"}
    if count_long>=2: return {"level":"medium","reason":f"{count_long}桌 ≥{LONG_CHAIN} (长连)"}
    return {"level":"none","reason":"无足够长连"}

# get base time score
def time_slot_score(dt, state):
    hhmm = dt.hour*60 + dt.minute
    base=None
    for s in TIME_SLOTS:
        sh,sm,eh,em, bw, ew, hw = s
        start=sh*60+sm; end=eh*60+em
        if end<=start:
            if hhmm>=start or hhmm<end: base=(bw,ew,hw); break
        else:
            if start<=hhmm<end: base=(bw,ew,hw); break
    if base is None:
        # default
        if dt.weekday()>=5: return DEFAULT_BASE_WEEKEND
        if is_holiday(dt, state): return DEFAULT_BASE_HOLIDAY
        return DEFAULT_BASE_WEEKDAY
    # pick based on day type
    if is_holiday(dt, state):
        return base[2]
    if dt.weekday()>=5:
        return base[1]
    return base[0]

def estimate_remaining_minutes(infos, level):
    if level=="none": return 0
    maxc = max((t.get("current_continuous",0) for t in infos), default=0)
    if level=="strong":
        est_hands = max(3, min(12, int((SUPER_DRAGON - maxc)/1.0) + 4))
    else:
        est_hands = max(2, min(8, int((LONG_CHAIN - maxc)/1.0) + 2))
    return max(1, int((est_hands*AVG_HAND_SECONDS)//60))

# main run
def run_once():
    state = load_state()
    init_tables(state)
    tables = state["tables"]
    dt = now()
    tscore = time_slot_score(dt, state)

    infos=[]
    for k,seq in tables.items():
        nxt = simulate_next(seq, tscore)
        seq.append(nxt)
        if len(seq)>HISTORY_LEN: seq = seq[-HISTORY_LEN:]
        tables[k]=seq
        infos.append(analyze(seq))

    # compute combined score
    valid_long = sum(1 for t in infos if t.get("max_run",0) >= LONG_CHAIN and t.get("alternating_tail_len",0) < SINGLE_JUMP_EXCLUDE)
    valid_dragon = sum(1 for t in infos if t.get("max_run",0) >= DRAGON and t.get("alternating_tail_len",0) < SINGLE_JUMP_EXCLUDE)
    combined = tscore + random.randint(-8,8) + valid_long*3 + valid_dragon*6
    combined = max(0, min(200, combined))
    prob = min(99, int((combined/150.0)*100))

    judgment = judge_all(infos)
    level = judgment["level"]
    reason = judgment["reason"]

    alert = state.get("alert")

    # If alert exists, check prewarn/ end
    if alert:
        end = datetime.fromisoformat(alert["end_time"]).astimezone(TZ)
        if now() >= end:
            start = datetime.fromisoformat(alert["start_time"]).astimezone(TZ)
            mins = int((end - start).total_seconds()//60)
            send_telegram(f"{EMOJI_END} <b>DG 模型 — 放水已结束</b>\n类型: {alert['type']}\n开始: {start.strftime('%Y-%m-%d %H:%M:%S')}\n结束: {end.strftime('%Y-%m-%d %H:%M:%S')}\n持续: {mins} 分钟\n说明: {alert.get('detail','')}")
            state["alert"] = None
            save_state(state); commit_state()
        else:
            if not alert.get("prewarn_sent") and (end - now()) <= timedelta(minutes=PREWARN_MIN):
                left = int((end - now()).total_seconds()//60)
                send_telegram(f"{EMOJI_WARN} <b>DG 预警 — 放水即将结束</b>\n类型: {alert['type']}\n预计结束: {end.strftime('%Y-%m-%d %H:%M:%S')}\n剩余: {left} 分钟\n说明: {alert.get('detail','')}")
                alert["prewarn_sent"] = True
                state["alert"] = alert
                save_state(state); commit_state()
    else:
        # decide opening
        if combined >= THRESH_STRONG:
            est_min = estimate_remaining_minutes(infos, "strong")
            dur = max(DUR_MIN, min(DUR_MAX, est_min + random.randint(3,10)))
            endt = now() + timedelta(minutes=dur)
            alert = {"type":"strong","start_time":now().isoformat(),"end_time":endt.isoformat(),
                     "detail":f"{reason}; combined={combined}; prob≈{prob}%", "prewarn_sent":False}
            state["alert"]=alert; save_state(state); commit_state()
            send_telegram(f"{EMOJI_STRONG} <b>DG 模型 — 强放水 (HIGH) 开始</b>\n概率估计: {prob}%\n理由: {reason}\n开始: {now_str()}\n预计结束: {endt.strftime('%Y-%m-%d %H:%M:%S')}\n预计持续: {dur} 分钟\n说明: 如果你看到“断连开单/同排连开”则为最佳入场点。")
        elif combined >= THRESH_MEDIUM:
            est_min = estimate_remaining_minutes(infos, "medium")
            dur = max(DUR_MIN, min(DUR_MAX, est_min + random.randint(1,5)))
            endt = now() + timedelta(minutes=dur)
            alert = {"type":"medium","start_time":now().isoformat(),"end_time":endt.isoformat(),
                     "detail":f"{reason}; combined={combined}; prob≈{prob}%", "prewarn_sent":False}
            state["alert"]=alert; save_state(state); commit_state()
            send_telegram(f"{EMOJI_MEDIUM} <b>DG 模型 — 中等胜率 (MEDIUM) 开始</b>\n概率估计: {prob}%\n理由: {reason}\n开始: {now_str()}\n预计结束: {endt.strftime('%Y-%m-%d %H:%M:%S')}\n预计持续: {dur} 分钟\n说明: 若你观察到“同排连开/长连”可考虑入场。")

    # persist tables & state always
    state["tables"] = tables
    save_state(state)
    commit_state()

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        print("exception", e); traceback.print_exc()
        try:
            send_telegram(f"⚠️ DG monitor exception: {str(e)}")
        except:
            pass
    # exit 0 always
