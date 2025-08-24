# scripts/monitor_dg_core.py
# Core DG monitor logic: fetch DG page, screenshot, analyze, classify, Telegram alerting, state/history.
import os, json, math, time, subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------- Config (可调) ----------
BOT_TOKEN = os.getenv("DG_BOT_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8")
CHAT_ID = os.getenv("DG_CHAT_ID", "485427847")
DG_URLS = [os.getenv("DG_URL1","https://dg18.co/wap/"), os.getenv("DG_URL2","https://dg18.co/")]
LOCAL_TZ_OFFSET = os.getenv("TZ_OFFSET", "+08:00")  # e.g. +08:00

# 判定阈值（请按需微调）
MIN_BOARDS_FOR_PUTTING_WATER = 3     # 放水判定：至少 3 张有长龙/超长龙
MIN_LONG_BOARDS_FOR_MID = 3         # 中等胜率中对“长龙”要求（你的要求：至少 3 张桌子有长龙）
MIN_MULTICHAN_BOARDS = 3            # 中等胜率要求：至少 3 张桌子具有 >=3 个 run >=4（多连/连珠）
LONGISH_LEN = 4                     # 长连
LONG_CHAIN_LEN = 8                  # 龙
SUPER_LONG_CHAIN_LEN = 10           # 超长龙

MIN_BLOB_SIZE = 5
RESIZE_WIDTH = 1200
CELL = 80
AUTO_CELL_THRESHOLD = 20

STATE_FILE = ".dg_state.json"
HISTORY_MAX = 100  # 存储历史事件数上限

# 冷却（若发送一次开始提醒后，过短时间不重复发送）
COOLDOWN_MINUTES = 10

# ---------- 时间工具 ----------
def now_utc():
    return datetime.now(timezone.utc)

def to_local(dt_utc):
    # convert UTC dt to local using LOCAL_TZ_OFFSET like '+08:00'
    sign = 1 if LOCAL_TZ_OFFSET[0] == "+" else -1
    hh = int(LOCAL_TZ_OFFSET[1:3])
    mm = int(LOCAL_TZ_OFFSET[4:6]) if len(LOCAL_TZ_OFFSET) >= 6 else 0
    tzlocal = timezone(timedelta(hours=sign*hh, minutes=sign*mm))
    return dt_utc.astimezone(tzlocal)

def local_str(dt_utc):
    return to_local(dt_utc).strftime("%Y-%m-%d %H:%M:%S")

# ---------- state persistence ----------
def load_state():
    p = Path(STATE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except:
            return {}
    return {}

def save_state(state):
    Path(STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    # commit back to repo for next runs to read persistent history
    try:
        subprocess.run(["git", "config", "--global", "user.email", "dg-monitor@example.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "dg-monitor-bot"], check=True)
        subprocess.run(["git", "add", STATE_FILE], check=True)
        subprocess.run(["git", "commit", "-m", "dg: update state"], check=True)
        subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print("Warning: push failed:", e)

# ---------- Telegram ----------
def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=30)
        j = r.json()
        if not j.get("ok"):
            print("Telegram send failed:", j)
            return False
        return True
    except Exception as e:
        print("Telegram error:", e)
        return False

# ---------- image helpers ----------
def find_color_blobs(img_bgr, min_size=MIN_BLOB_SIZE):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # red ranges
    lower1 = np.array([0, 80, 60]); upper1 = np.array([8, 255, 255])
    lower2 = np.array([170, 80, 60]); upper2 = np.array([180,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue
    lowerb = np.array([95, 70, 60]); upperb = np.array([135, 255, 255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    blobs = []
    for mask, color in [(mask_r, 'B'), (mask_b, 'P')]:
        num_labels, labels = cv2.connectedComponents(mask)
        for lab in range(1, num_labels):
            pts = np.where(labels == lab)
            cnt = len(pts[0])
            if cnt < min_size:
                continue
            cx = int(np.mean(pts[1])); cy = int(np.mean(pts[0]))
            blobs.append({"cx":cx, "cy":cy, "count":cnt, "color":('B' if color=='B' else 'P')})
    return blobs

def auto_detect_regions(img):
    H,W = img.shape[:2]
    cell = CELL
    cols = math.ceil(W/cell); rows = math.ceil(H/cell)
    counts = np.zeros((rows, cols), dtype=np.int32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            h = hsv[y,x,0]; s = hsv[y,x,1]; v = hsv[y,x,2]
            if ((h<=8 or h>=170) and s>70 and v>50) or (95<=h<=135 and s>70 and v>50):
                cx = x//cell; cy = y//cell
                if 0 <= cy < rows and 0 <= cx < cols:
                    counts[cy,cx] += 1
    thr = AUTO_CELL_THRESHOLD
    hits = []
    for ry in range(rows):
        for rx in range(cols):
            if counts[ry,rx] >= thr:
                hits.append((rx,ry))
    if not hits:
        thr2 = max(4, thr//2)
        for ry in range(rows):
            for rx in range(cols):
                if counts[ry,rx] >= thr2:
                    hits.append((rx,ry))
    groups = []
    for rx,ry in hits:
        x = rx*cell; y = ry*cell; w = cell; h = cell
        merged=False
        for g in groups:
            if not (x > g['x']+g['w']+cell or x+w < g['x']-cell or y > g['y']+g['h']+cell or y+h < g['y']-cell):
                g['x'] = min(g['x'], x); g['y'] = min(g['y'], y)
                g['w'] = max(g['w'], x+w - g['x']); g['h'] = max(g['h'], y+h - g['y'])
                merged=True
                break
        if not merged:
            groups.append({'x':x,'y':y,'w':w,'h':h})
    regs = []
    for g in groups:
        x = max(0, g['x']-8); y = max(0, g['y']-8)
        w = min(W - x, g['w'] + 16); h = min(H - y, g['h'] + 16)
        regs.append((x,y,w,h))
    return regs

def analyze_region(img, rect):
    x,y,w,h = rect
    sub = img[y:y+h, x:x+w].copy()
    blobs = find_color_blobs(sub, min_size=MIN_BLOB_SIZE)
    if not blobs:
        return {"total":0, "runs":[], "flattened":[]}
    # cluster x => columns
    xs = sorted([b['cx'] for b in blobs])
    groups = [[xs[0]]]
    for v in xs[1:]:
        if v - groups[-1][-1] <= 16:
            groups[-1].append(v)
        else:
            groups.append([v])
    centers = [int(sum(g)/len(g)) for g in groups]
    columns = {c:[] for c in centers}
    for b in blobs:
        nearest = min(centers, key=lambda c: abs(c - b['cx']))
        columns[nearest].append(b)
    sequences = []
    for c in sorted(columns.keys()):
        col = columns[c]
        col.sort(key=lambda it: it['cy'])
        seq = [it['color'] for it in col]
        sequences.append(seq)
    flattened = []
    maxlen = max((len(s) for s in sequences), default=0)
    for r in range(maxlen):
        for c in range(len(sequences)):
            if r < len(sequences[c]):
                flattened.append(sequences[c][r])
    # runs
    runs = []
    if flattened:
        cur = flattened[0]; ln = 1
        for ch in flattened[1:]:
            if ch == cur:
                ln += 1
            else:
                runs.append({"color":cur, "len":ln})
                cur = ch; ln = 1
        runs.append({"color":cur, "len":ln})
    return {"total": len(flattened), "runs": runs, "flattened": flattened}

def detect_from_screenshot_bytes(screenshot_bytes):
    nparr = np.frombuffer(screenshot_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    H,W = img.shape[:2]
    scale = RESIZE_WIDTH / W if W > RESIZE_WIDTH else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    regs = auto_detect_regions(img)
    if not regs:
        regs = [(0,0,img.shape[1], img.shape[0])]
    summaries = []
    for r in regs:
        s = analyze_region(img, r)
        maxrun = max((rr['len'] for rr in s['runs']), default=0)
        long_runs_count = sum(1 for rr in s['runs'] if rr['len'] >= LONGISH_LEN)
        runs_ge4 = sum(1 for rr in s['runs'] if rr['len'] >= LONGISH_LEN)
        # classify
        cat = 'other'
        if maxrun >= SUPER_LONG_CHAIN_LEN:
            cat = 'super_long'
        elif maxrun >= LONG_CHAIN_LEN:
            cat = 'long'
        elif maxrun >= LONGISH_LEN:
            cat = 'longish'
        summaries.append({
            "rect": r,
            "total": s['total'],
            "maxrun": maxrun,
            "runs": s['runs'],
            "long_runs_count": long_runs_count,
            "category": cat
        })
    return summaries

def classify_overall(summaries):
    # compute counts
    dragon_boards = sum(1 for s in summaries if s['maxrun'] >= LONG_CHAIN_LEN)
    super_boards = sum(1 for s in summaries if s['maxrun'] >= SUPER_LONG_CHAIN_LEN)
    multichain_boards = sum(1 for s in summaries if s['long_runs_count'] >= 3)  # >=3 runs >=4
    longish_boards = sum(1 for s in summaries if s['maxrun'] >= LONGISH_LEN)
    total_boards = len(summaries)
    sparse_boards = sum(1 for s in summaries if s['total'] < 6)
    # 1) 放水判定：若长龙/超长龙桌数 >= MIN_BOARDS_FOR_PUTTING_WATER => 放水
    if dragon_boards >= MIN_BOARDS_FOR_PUTTING_WATER:
        return "放水时段（提高胜率）", {"dragon_boards": dragon_boards, "super_boards": super_boards, "multichain_boards": multichain_boards}
    # 2) 中等胜率（中上）：你的要求：至少有 3 张桌子有 >=3 runs>=4 && >=3 张桌子有 龙/超龙
    if multichain_boards >= MIN_MULTICHAN_BOARDS and dragon_boards >= MIN_LONG_BOARDS_FOR_MID:
        return "中等胜率（中上）", {"dragon_boards": dragon_boards, "super_boards": super_boards, "multichain_boards": multichain_boards}
    # 3) 胜率调低（收割）：多数桌 sparse
    if total_boards > 0 and sparse_boards >= total_boards * 0.6:
        return "胜率调低（收割时段）", {"dragon_boards": dragon_boards, "super_boards": super_boards, "multichain_boards": multichain_boards}
    # 4) 否则胜率中等
    return "胜率中等（平台收割中等时段）", {"dragon_boards": dragon_boards, "super_boards": super_boards, "multichain_boards": multichain_boards}

# ---------- Playwright fetch with retries and slider attempts ----------
def fetch_screenshot_from_dg(max_attempts=4):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1600,"height":900}, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        page = context.new_page()
        for url in DG_URLS:
            for attempt in range(1, max_attempts+1):
                try:
                    page.goto(url, timeout=30000)
                    time.sleep(2)
                    # Try to find and click "Free" or "免费试玩"
                    clicked = False
                    texts = ["Free", "免费试玩", "免费", "试玩"]
                    for txt in texts:
                        try:
                            btn = page.query_selector(f'xpath=//*[contains(text(), "{txt}") or contains(@alt, "{txt}") or contains(@aria-label, "{txt}")]')
                            if btn:
                                try:
                                    btn.click(timeout=3000)
                                except:
                                    page.evaluate("(el)=>el.click()", btn)
                                time.sleep(1)
                                clicked = True
                                break
                        except:
                            continue
                    # attempt slider / drag
                    try:
                        possible = page.query_selector_all("input[type=range], .slider, .geetest_slider_button, .drag, .captcha, .slider-button")
                        if possible:
                            for el in possible:
                                try:
                                    box = el.bounding_box()
                                    if box:
                                        sx = box['x'] + box['width']/2; sy = box['y'] + box['height']/2
                                        page.mouse.move(sx, sy)
                                        page.mouse.down()
                                        page.mouse.move(sx + box['width']*2, sy, steps=20)
                                        page.mouse.up()
                                        time.sleep(1)
                                except:
                                    pass
                    except:
                        pass
                    # wait for board area (heuristic)
                    # Wait for some element that tends to exist in the DG page like canvas or table or specific class
                    try:
                        page.wait_for_timeout(1500)
                        # try waiting for images or elements
                        # (fallback: just wait a bit)
                        page.wait_for_timeout(1500)
                    except:
                        pass
                    screenshot = page.screenshot(full_page=True)
                    if screenshot and len(screenshot) > 5000:
                        browser.close()
                        return screenshot
                except PWTimeout:
                    print("Playwright timeout on", url, "attempt", attempt)
                except Exception as e:
                    print("Playwright error:", e, "on", url, "attempt", attempt)
                time.sleep(2)
        browser.close()
    return None

# ---------- history/estimate helpers ----------
def median(lst):
    if not lst: return None
    s = sorted(lst)
    n = len(s)
    if n%2==1:
        return s[n//2]
    else:
        return (s[n//2-1] + s[n//2]) / 2.0

# ---------- main run_once ----------
def run_once():
    print("[run_once] start", datetime.utcnow().isoformat(), "UTC")
    state = load_state()
    last_status = state.get("status")
    start_ts = state.get("start_ts")  # stored as epoch
    history = state.get("history", [])  # list of durations in seconds
    last_alert_ts = state.get("last_alert_ts", 0)
    cooldown_until = state.get("cooldown_until", 0)

    # 1) fetch screenshot
    shot = fetch_screenshot_from_dg()
    if not shot:
        print("[run_once] failed to fetch screenshot")
        return

    # 2) detect
    summaries = detect_from_screenshot_bytes(shot)
    overall, stats = classify_overall(summaries)
    print("[run_once] overall:", overall, stats)

    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    # if in alert state previously and still in alert (or changed) we handle transitions
    need_alert_now = overall in ("放水时段（提高胜率）","中等胜率（中上）")
    # handle start
    if need_alert_now:
        # if not already in active alert
        if last_status not in ("放水时段（提高胜率）","中等胜率（中上）"):
            # only send if not cooldown
            if now_ts >= cooldown_until:
                # compute estimated end time from history median
                median_sec = median(history)
                estimated_info = ""
                if median_sec:
                    elapsed = 0
                    # start not set yet so elapsed = 0
                    est_end_dt = now + timedelta(seconds=median_sec)
                    remaining_min = int(median_sec/60)
                    estimated_info = f"估计结束：{local_str(est_end_dt)}（估计还剩 {remaining_min} 分钟，基于历史中位数）"
                else:
                    estimated_info = "估计结束：未知（历史数据不足，结束后脚本会记录真实持续时间）"

                # build message
                emoji = "🟢" if overall.startswith("放水") else "🟡"
                msg = f"{emoji} [DG提醒] {overall} 已开始\n开始（本地）：{local_str(now)}\n{estimated_info}\n详情：长龙桌数={stats.get('dragon_boards',0)}，超长龙={stats.get('super_boards',0)}，多连桌数(≥3 runs≥4)={stats.get('multichain_boards',0)}"
                ok = send_telegram(msg)
                if ok:
                    print("[run_once] sent start alert")
                    # update state
                    state['status'] = overall
                    state['start_ts'] = now_ts
                    state['last_alert_ts'] = now_ts
                    state['cooldown_until'] = now_ts + COOLDOWN_MINUTES*60
                    save_state(state)
                else:
                    print("[run_once] failed to send telegram start")
            else:
                print("[run_once] would alert but in cooldown")
        else:
            # already in alert state; do not re-send start alert
            print("[run_once] already in alert state; continue monitoring for end")
    else:
        # Not currently need_alert_now
        if last_status in ("放水时段（提高胜率）","中等胜率（中上）"):
            # previously in alert, now ended -> send end msg and record history
            prev_start_ts = state.get("start_ts")
            if prev_start_ts:
                start_dt = datetime.fromtimestamp(prev_start_ts, tz=timezone.utc)
                dur = now - start_dt
                dur_seconds = int(dur.total_seconds())
                mins = dur_seconds // 60
                secs = dur_seconds % 60
                msg = f"🔴 [DG结束] {last_status} 已结束\n开始（本地）：{local_str(start_dt)}\n结束（本地）：{local_str(now)}\n持续：{mins} 分 {secs} 秒\n本次长龙桌数={stats.get('dragon_boards',0)}，多连桌数={stats.get('multichain_boards',0)}"
                send_telegram(msg)
                # record history
                history = state.get("history", [])
                history.append(dur_seconds)
                if len(history) > HISTORY_MAX:
                    history = history[-HISTORY_MAX:]
                state['history'] = history
                # update status
                state['status'] = overall
                state['start_ts'] = None
                state['cooldown_until'] = 0
                save_state(state)
                print("[run_once] sent end alert and saved history")
            else:
                # no start recorded; just update state
                state['status'] = overall
                state['start_ts'] = None
                state['cooldown_until'] = 0
                save_state(state)
                print("[run_once] alert ended but no start_ts recorded; updated state")
        else:
            # not alert before and not alert now -> nothing to do
            print("[run_once] no alert and nothing to do")

# If invoked as module
def run_once_entrypoint():
    run_once()

if __name__ == "__main__":
    run_once()
