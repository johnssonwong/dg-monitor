# monitor.py
# DG Baccarat monitoring -> screenshot -> simple color-cluster analysis -> Telegram alerts
# NOTE: this is an heuristic implementation. Tune clustering thresholds if required.

import os, sys, time, json, math, subprocess
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import cv2
from PIL import Image
from playwright.sync_api import sync_playwright

# ----------------------------
# CONFIGURATION (edit if you want)
# ----------------------------
# Default token & chat id (you already provided — but prefer using GitHub Secrets)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN') or "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID') or "485427847"

DG_URLS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

# detection thresholds (may need tuning)
CLUSTER_DIST_PX = 220    # max pixel distance to group colored blobs into same table
Y_ROW_BUCKET = 28        # bucket height to group blobs into same "row" in a table
MIN_BLOB_AREA = 30       # minimal pixel area to consider a colored blob

# decision thresholds (based on your rules)
# for "full盤長連局勢型" detection counts:
MIN_MATCH_TABLES_20_TOTAL = 8
MIN_MATCH_TABLES_10_TOTAL = 4

# for "中等勝率 (中上)" thresholds (20 tables -> >=6 , 10 tables -> >=3)
MIN_MATCH_TABLES_20_FOR_MID = 6
MIN_MATCH_TABLES_10_FOR_MID = 3

STATE_FILE = "state.json"

# ----------------------------
# Helper: Telegram
# ----------------------------
BASE_TELEGRAM = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_telegram_message(text, parse_mode="HTML"):
    url = f"{BASE_TELEGRAM}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode}
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.ok
    except Exception as e:
        print("Telegram send error:", e)
        return False

def send_telegram_photo(photo_path, caption=None):
    url = f"{BASE_TELEGRAM}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID}
    if caption:
        data["caption"] = caption
    try:
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            r = requests.post(url, data=data, files=files, timeout=30)
            return r.ok
    except Exception as e:
        print("Telegram photo send error:", e)
        return False

# ----------------------------
# Helper: state management in repo (read/write & commit)
# ----------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE,"r",encoding="utf-8"))
        except:
            return {}
    return {}

def save_state(state):
    json.dump(state, open(STATE_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def git_commit_state(msg="update state"):
    # commit & push state.json (uses actions' checkout credentials)
    try:
        subprocess.run(["git","config","user.email","action@github.com"], check=False)
        subprocess.run(["git","config","user.name","github-actions[bot]"], check=False)
        subprocess.run(["git","add", STATE_FILE], check=True)
        subprocess.run(["git","commit","-m", msg], check=True)
        subprocess.run(["git","push"], check=True)
    except subprocess.CalledProcessError as e:
        print("git commit/push failed (likely no changes or no push permission):", e)

# ----------------------------
# Image processing: detect red/blue blobs, cluster into tables, detect runs
# ----------------------------
def find_color_centers(img_bgr):
    # returns list of (x,y,color) where color in {'R','B'}
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red mask (two ranges)
    lower_r1 = np.array([0,100,60]); upper_r1 = np.array([10,255,255])
    lower_r2 = np.array([160,100,60]); upper_r2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)

    # Blue mask (approx)
    lower_b = np.array([90,80,50]); upper_b = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lower_b, upper_b)

    # clean
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)

    centers = []
    # red contours
    cnts, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA: continue
        x,y,w,h = cv2.boundingRect(c)
        centers.append((x + w//2, y + h//2, 'R'))
    # blue contours
    cnts, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA: continue
        x,y,w,h = cv2.boundingRect(c)
        centers.append((x + w//2, y + h//2, 'B'))

    return centers

def cluster_centers_to_tables(centers):
    clusters = []
    for (x,y,c) in centers:
        placed = False
        for cl in clusters:
            cx,cy = cl['centroid']
            if math.hypot(cx - x, cy - y) < CLUSTER_DIST_PX:
                cl['points'].append((x,y,c))
                # update centroid
                xs = [p[0] for p in cl['points']]; ys = [p[1] for p in cl['points']]
                cl['centroid'] = (sum(xs)/len(xs), sum(ys)/len(ys))
                placed = True
                break
        if not placed:
            clusters.append({'points':[(x,y,c)], 'centroid':(x,y)})
    return clusters

def analyze_cluster(cluster):
    # group points by approximate row (y)
    pts = cluster['points']
    rows = {}
    for (x,y,c) in pts:
        key = int(round(y / Y_ROW_BUCKET) * Y_ROW_BUCKET)
        rows.setdefault(key, []).append((x,c))
    # for each row, sort by x and find longest same-color consecutive run (by adjacency)
    max_run = 0
    runs = []
    for k, arr in rows.items():
        arr_sorted = sorted(arr, key=lambda t:t[0])
        cur_color = None; cur_len = 0
        for (x,col) in arr_sorted:
            if col == cur_color:
                cur_len += 1
            else:
                if cur_len>0:
                    runs.append((cur_color, cur_len))
                cur_color = col
                cur_len = 1
        if cur_len>0:
            runs.append((cur_color, cur_len))
        # update max_run
        for _,rl in runs:
            if rl > max_run: max_run = rl
    # classify
    cls = 'none'
    if max_run >= 10:
        cls = 'superdragon'
    elif max_run >= 8:
        cls = 'dragon'
    elif max_run >= 4:
        cls = 'long'
    return {'max_run': max_run, 'class': cls, 'rows_count': len(rows)}

# ----------------------------
# Decision logic based on your rules
# ----------------------------
def evaluate_picture(img_path):
    img = cv2.imread(img_path)
    centers = find_color_centers(img)
    # no centers => probably blank / shuffle screen
    if not centers:
        return {'total_tables': 0, 'matched': [], 'summary': 'no_tokens_detected'}
    clusters = cluster_centers_to_tables(centers)
    results = []
    for cl in clusters:
        res = analyze_cluster(cl)
        res['cluster_centroid'] = cl['centroid']
        results.append(res)

    total_tables = len(results)
    # count categories
    long_cnt = sum(1 for r in results if r['class'] == 'long')
    dragon_cnt = sum(1 for r in results if r['class'] == 'dragon')
    super_cnt = sum(1 for r in results if r['class'] == 'superdragon')

    # determine "full盘长连局势型" (we say a table qualifies if it has class long/dragon/super)
    qualifying_tables = sum(1 for r in results if r['class'] in ('long','dragon','superdragon'))

    classification = 'no_signal'
    # check "超长龙触发型": >=1 super and >=2 dragon (total >=3)
    if super_cnt >=1 and dragon_cnt >=2 and (super_cnt + dragon_cnt >=3):
        classification = 'full_super_trigger'
    else:
        # check full-plate long chain
        if total_tables >= 20:
            if qualifying_tables >= MIN_MATCH_TABLES_20_TOTAL:
                classification = 'full_plate_long'
        elif total_tables >= 10:
            if qualifying_tables >= MIN_MATCH_TABLES_10_TOTAL:
                classification = 'full_plate_long'
        # also consider mid-level (中上) thresholds
        if classification == 'no_signal':
            if total_tables >= 20 and qualifying_tables >= MIN_MATCH_TABLES_20_FOR_MID:
                classification = 'mid_high'
            elif total_tables >= 10 and qualifying_tables >= MIN_MATCH_TABLES_10_FOR_MID:
                classification = 'mid_high'

    # Compose summary
    summary = {
        'total_tables': total_tables,
        'qualifying_tables': qualifying_tables,
        'long_cnt': long_cnt,
        'dragon_cnt': dragon_cnt,
        'super_cnt': super_cnt,
        'classification': classification,
        'raw_clusters': results
    }
    return summary

# ----------------------------
# Playwright navigation & screenshot
# ----------------------------
def capture_dg_screenshot(out_path="screenshot.png", max_wait=20_000):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
        context = browser.new_context(viewport={"width":1366,"height":768})
        page = context.new_page()
        # try each url until one loads
        loaded = False
        for url in DG_URLS:
            try:
                page.goto(url, timeout=20000)
                loaded = True
                break
            except Exception as e:
                print("Failed load", url, e)
                continue
        if not loaded:
            print("Failed to load any DG URL")
            browser.close()
            return False

        # Try to click "Free" or "免费试玩"
        try:
            page.wait_for_timeout(1200)
            # attempts in order (text might vary)
            for sel in ["text=Free", "text=免费试玩", "text=Free Play", "text=免费"]:
                try:
                    page.click(sel, timeout=3500)
                    print("Clicked", sel)
                    page.wait_for_timeout(1200)
                    break
                except Exception:
                    pass
            # handle slider: attempt to find an element with role 'slider' or class containing 'drag'
            # Try multiple strategies; this is heuristic and may fail depending on site's DOM
            try:
                slider = page.query_selector("div[role=slider]")
                if slider:
                    box = slider.bounding_box()
                    if box:
                        x0 = box["x"] + 2
                        y0 = box["y"] + box["height"]/2
                        x1 = box["x"] + box["width"] - 2
                        page.mouse.move(x0, y0)
                        page.mouse.down()
                        page.mouse.move(x1, y0, steps=15)
                        page.mouse.up()
                        page.wait_for_timeout(1200)
                else:
                    # fallback: search for input[type=range]
                    inp = page.query_selector("input[type=range]")
                    if inp:
                        box = inp.bounding_box()
                        if box:
                            x0 = box["x"] + 2
                            y0 = box["y"] + box["height"]/2
                            x1 = box["x"] + box["width"] - 2
                            page.mouse.move(x0,y0)
                            page.mouse.down()
                            page.mouse.move(x1,y0, steps=15)
                            page.mouse.up()
                            page.wait_for_timeout(1200)
            except Exception as e:
                print("slider attempt error:", e)
            # wait a bit for game area to render
            page.wait_for_timeout(2500)
        except Exception as e:
            print("click-free step error:", e)

        # final wait and screenshot
        try:
            page.wait_for_timeout(1400)
            page.screenshot(path=out_path, full_page=True)
            print("Saved screenshot", out_path)
        except Exception as e:
            print("screenshot error:", e)
            browser.close()
            return False

        browser.close()
        return True

# ----------------------------
# Estimate end-time heuristic (very rough)
# ----------------------------
def estimate_end_time(summary):
    # heuristic: base on max chain
    clusters = summary.get('raw_clusters', [])
    max_chain = 0
    for r in clusters:
        if r.get('max_run',0) > max_chain:
            max_chain = r['max_run']
    # simple heuristic mapping:
    if max_chain >= 12:
        mins = 20
    elif max_chain >= 10:
        mins = 12
    elif max_chain >= 8:
        mins = 8
    elif max_chain >= 4:
        mins = 5
    else:
        mins = 3
    est_end = datetime.now(timezone.utc) + timedelta(minutes=mins)
    return est_end.astimezone(tz=timezone(timedelta(hours=8))), mins  # convert to MYT

# ----------------------------
# Main run flow
# ----------------------------
def main():
    timestamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    print("Run at (MYT):", timestamp.isoformat())

    # capture screenshot
    ss_path = "screenshot.png"
    ok = capture_dg_screenshot(ss_path)
    if not ok:
        print("Failed to capture screenshot.")
        return

    summary = evaluate_picture(ss_path)
    print("Summary:", summary.get('classification'), " tables:", summary.get('total_tables'))

    state = load_state()
    active = state.get('active', False)
    classification = summary.get('classification', 'no_signal')

    # Decide: if classification indicates "放水"/"中等" then send start message (if not active)
    if classification in ('full_super_trigger','full_plate_long','mid_high'):
        # map labels
        label = "放水时段（提高胜率）" if classification in ('full_super_trigger','full_plate_long') else "中等胜率（中上）"
        # if not already active, start event
        if not active:
            est_time, est_mins = estimate_end_time(summary)
            start_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
            msg = (f"🟢 <b>{label}</b> 侦测到！\n"
                   f"时间 (MYT): {start_time}\n"
                   f"匹配桌数: {summary.get('qualifying_tables')}/{summary.get('total_tables')}\n"
                   f"长龙数: {summary.get('dragon_cnt')}，超长龙数: {summary.get('super_cnt')}\n"
                   f"预计放水结束时间（估计）: {est_time.strftime('%Y-%m-%d %H:%M:%S')} （约 {est_mins} 分钟）\n"
                   f"请立即查看并手动入场。")
            send_telegram_message(msg)
            send_telegram_photo(ss_path, caption="侦测到的桌面截图（供参考）")
            # set state
            state = {'active': True, 'classification': classification, 'start_time': start_time}
            save_state(state)
            git_commit_state(msg=f"start event {classification} at {start_time}")
        else:
            # already active: do nothing (avoid spamming)
            print("Already active - no new start message sent.")
    else:
        # no signal: if previously active then close event and notify
        if active:
            start_time = state.get('start_time')
            end_time_obj = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
            # compute duration
            try:
                st = datetime.fromisoformat(start_time)
            except Exception:
                st = None
            duration_minutes = None
            if st:
                duration_minutes = int((end_time_obj - st).total_seconds() / 60)
            # send end message
            end_msg = (f"🔴 放水/中等胜率 已结束（检测到条件不再成立）。\n"
                       f"开始时间: {start_time}\n"
                       f"结束时间: {end_time_obj.isoformat()}\n"
                       f"持续时长 (分钟): {duration_minutes if duration_minutes is not None else '未知'}\n"
                       f"最后一次匹配情况: matching {summary.get('qualifying_tables')}/{summary.get('total_tables')}")
            send_telegram_message(end_msg)
            send_telegram_photo(ss_path, caption="当前桌面截图（放水结束时）")
            # update state
            state = {'active': False}
            save_state(state)
            git_commit_state(msg="end event")
        else:
            print("No active event and no trigger.")

if __name__ == "__main__":
    main()
