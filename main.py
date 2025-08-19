#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DG Auto Monitor for Telegram
- Uses Playwright to open DG page, simulate entering Free demo, handle scroll slider,
  take screenshot of full page.
- Uses OpenCV to detect red/blue beads, cluster into boards, compute runs,
  and classify overall state following your rules (放水 / 中等中上 / 胜率中等 / 收割).
- Maintains small state (start_ts, last_ts, historical durations) in .dg_state/state.json
  persisted via actions/cache in GitHub Actions workflow.
- Sends Telegram messages when entering 放水/中等 中上 and when that period ends.
"""

import os, sys, json, time, math, statistics, traceback
from datetime import datetime, timezone
import requests
from pathlib import Path

# Image processing libs
import cv2
import numpy as np

# Playwright
from playwright.sync_api import sync_playwright

# config (can be overridden by env)
TG_TOKEN = os.environ.get("TG_TOKEN") or os.environ.get("TG_TOKEN_RAW")
TG_CHAT = os.environ.get("TG_CHAT_ID") or os.environ.get("TG_CHAT")
DG_URL1 = os.environ.get("DG_URL1", "https://dg18.co/wap/")
DG_URL2 = os.environ.get("DG_URL2", "https://dg18.co/")
STATE_DIR = Path(".dg_state")
STATE_FILE = STATE_DIR / "state.json"

# detection parameters (may need tuning based on screenshot)
MIN_BLOB = 8  # min pixels for a bead blob
COLUMN_GAP = 18  # pixel gap to separate columns when clustering beads within a board
BOARD_CLUSTER_GAP = 120  # when merging bead centers into boards (in pixels)
MIN_BOARDS_FOR_POW = int(os.environ.get("MIN_BOARDS_FOR_POW", "3"))  # 放水至少满足桌数
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))  # 中等(中上)要求 >=2 长龙
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))

# helper to persist small state
def load_state():
    if not STATE_FILE.exists():
        return {"periods": [], "current": None, "last_notify": 0}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except:
        return {"periods": [], "current": None, "last_notify": 0}

def save_state(s):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s), encoding="utf-8")

# Telegram send
def send_telegram(message: str):
    if not TG_TOKEN or not TG_CHAT:
        print("[WARN] TG token/chat not set. Message not sent:", message)
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=20)
        j = r.json()
        if not j.get("ok"):
            print("Telegram error:", j)
            return False
        return True
    except Exception as e:
        print("Telegram send failed:", e)
        return False

# Open page, navigate to DG, click Free, try to handle slider, return screenshot bytes (PNG)
def capture_dg_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1366, "height": 900})
        page = context.new_page()
        # try both links
        for entry in [DG_URL1, DG_URL2]:
            try:
                page.goto(entry, timeout=60000)
                time.sleep(2)
                # try clicking a Free button text
                for sel_text in ["Free", "免费试玩", "免费", "free"]:
                    try:
                        # case-insensitive: use locator with text
                        btn = page.locator(f"text=/.*{sel_text}.*/i")
                        if btn.count() > 0:
                            btn.first.click(timeout=5000)
                            time.sleep(1)
                    except:
                        pass
                # After clicking, a new pop-up/page may appear. Wait a bit.
                time.sleep(2)
                # Try to handle the "scroll security bar" by attempting to find known slider elements,
                # otherwise simulate a drag near bottom center.
                try:
                    # try to find possible slider by input[type=range] or class names
                    slider = page.query_selector("input[type=range]")
                    if slider:
                        # set value by JS
                        page.evaluate("(el)=>el.value = el.max || 100", slider)
                        time.sleep(0.8)
                    else:
                        # fallback: drag from left to right near bottom center
                        box = page.viewport_size
                        w = box["width"]
                        h = box["height"]
                        y = int(h * 0.85)
                        x1 = int(w * 0.12); x2 = int(w * 0.88)
                        page.mouse.move(x1, y)
                        page.mouse.down()
                        page.mouse.move(x2, y, steps=20)
                        page.mouse.up()
                        time.sleep(1)
                except Exception as e:
                    print("slider attempt failed:", e)

                # give page time to load fully
                time.sleep(3)
                # capture full page screenshot
                img_bytes = page.screenshot(full_page=True)
                browser.close()
                return img_bytes
            except Exception as e:
                print("entry", entry, "failed:", e)
                continue
        # if both failed:
        browser.close()
        raise RuntimeError("Unable to reach DG pages with given entries.")

# Image processing: detect red and blue bead centers
def detect_beads(img_bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # thresholds for red (two ranges) and blue - may need tuning
    # red lower range
    lower1 = np.array([0, 70, 50]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50]); upper2 = np.array([180, 255, 255])
    blue_l = np.array([95, 70, 50]); blue_u = np.array([135, 255, 255])

    mask_r1 = cv2.inRange(hsv, lower1, upper1)
    mask_r2 = cv2.inRange(hsv, lower2, upper2)
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    mask_b = cv2.inRange(hsv, blue_l, blue_u)

    # morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel)

    # find contours for both masks
    centers = []
    def extract_centers(mask, color_label):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_BLOB: continue
            M = cv2.moments(c)
            if M["m00"]==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            centers.append({"x":cx, "y":cy, "color": color_label})
    extract_centers(mask_r, "B")  # B = banker/red (we use 'B' to align earlier notation)
    extract_centers(mask_b, "P")  # P = player/blue
    return centers, img

# cluster bead centers into boards by spatial proximity (simple box clustering)
def cluster_boards(centers):
    pts = [(c["x"], c["y"]) for c in centers]
    used = [False]*len(pts)
    boards = []
    for i,(x,y) in enumerate(pts):
        if used[i]: continue
        # start new group
        bx1,by1,bx2,by2 = x,y,x,y
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed=False
            for j,(xx,yy) in enumerate(pts):
                if used[j]: continue
                # if point lies near current box expand
                if xx >= bx1-BOARD_CLUSTER_GAP and xx <= bx2+BOARD_CLUSTER_GAP and yy >= by1-BOARD_CLUSTER_GAP and yy <= by2+BOARD_CLUSTER_GAP:
                    used[j]=True
                    group.append(j)
                    bx1=min(bx1,xx); by1=min(by1,yy); bx2=max(bx2,xx); by2=max(by2,yy)
                    changed=True
        # record board with its bead indices
        boards.append({"bbox":(bx1,by1,bx2,by2), "indices": group})
    # convert boards to a structured form
    board_list = []
    for b in boards:
        beads = [centers[i] for i in b["indices"]]
        board_list.append({"bbox": b["bbox"], "beads": beads})
    return board_list

# Given bead positions for a board, cluster by X positions into columns and read top->bottom per column
def read_board_sequence(board):
    beads = board["beads"]
    if len(beads) == 0:
        return []
    xs = sorted(set([b["x"] for b in beads]))
    # cluster xs by gap
    xs_sorted = sorted(xs)
    clusters = []
    cur = [xs_sorted[0]]
    for v in xs_sorted[1:]:
        if v - cur[-1] <= COLUMN_GAP:
            cur.append(v)
        else:
            clusters.append(cur)
            cur=[v]
    clusters.append(cur)
    # cluster centers per column (approx by x)
    cols = []
    for c in clusters:
        # median x for cluster
        mx = int(np.median(c))
        # pick beads closest to mx
        col_beads = [b for b in beads if abs(b["x"]-mx) <= (COLUMN_GAP+4)]
        # sort top->bottom (y ascending)
        col_beads_sorted = sorted(col_beads, key=lambda z: z["y"])
        cols.append(col_beads_sorted)
    # build flattened sequence in bead-plate reading: for row index 0..maxlen-1 take each column
    maxlen = max(len(c) for c in cols)
    seq = []
    for row in range(maxlen):
        for col in cols:
            if row < len(col):
                seq.append(col[row]["color"])
    return seq

# compute runs in flattened sequence
def compute_runs(seq):
    if not seq: return []
    runs = []
    cur = {"color": seq[0], "len": 1}
    for s in seq[1:]:
        if s == cur["color"]:
            cur["len"] += 1
        else:
            runs.append(cur)
            cur = {"color": s, "len": 1}
    runs.append(cur)
    return runs

# classify overall state given boards info
def classify_overall(board_summaries):
    # board_summaries: list of dicts: {maxRun, category}
    longCount = sum(1 for b in board_summaries if b["category"] in ("long","super_long"))
    superCount = sum(1 for b in board_summaries if b["category"] == "super_long")
    longishCount = sum(1 for b in board_summaries if b["category"] == "longish")
    total_boards = len(board_summaries)
    sparse = sum(1 for b in board_summaries if b["total"] < 6)
    # rules:
    if longCount >= MIN_BOARDS_FOR_POW:
        return "放水时段（提高胜率）", {"longCount":longCount,"superCount":superCount,"longishCount":longishCount}
    elif longCount >= MID_LONG_REQ and longishCount > 0:
        return "中等胜率（中上）", {"longCount":longCount,"superCount":superCount, "longishCount":longishCount}
    else:
        if total_boards>0 and sparse >= total_boards*0.6:
            return "胜率调低 / 收割时段", {"longCount":longCount,"superCount":superCount,"sparse":sparse}
        else:
            return "胜率中等（平台收割中等时段）", {"longCount":longCount,"superCount":superCount,"sparse":sparse}

def main():
    print("DG monitor starting", datetime.now().isoformat())
    state = load_state()
    try:
        img_bytes = capture_dg_page()
    except Exception as e:
        print("capture failed:", e)
        traceback.print_exc()
        return

    centers, img = detect_beads(img_bytes)
    print(f"Detected {len(centers)} bead centers")
    boards = cluster_boards(centers)
    print(f"Clustered into {len(boards)} boards (candidates)")

    # build board summaries
    board_summaries = []
    for b in boards:
        seq = read_board_sequence(b)
        runs = compute_runs(seq)
        maxrun = 0
        if runs:
            maxrun = max(r["len"] for r in runs)
        if maxrun >= 10:
            cat = "super_long"
        elif maxrun >= 8:
            cat = "long"
        elif maxrun >= 4:
            cat = "longish"
        else:
            cat = "other"
        board_summaries.append({"total":len(seq), "maxRun":maxrun, "category":cat, "runs":runs})

    overall, meta = classify_overall(board_summaries)
    print("Overall:", overall, meta)

    # now manage notifications with state + cooldown
    now_ts = int(time.time()*1000)
    last_notify = state.get("last_notify", 0)
    cooldown_ms = COOLDOWN_MINUTES * 60 * 1000

    current = state.get("current", None)  # structure when in period: {"type":"放水...", "start":ts, "last":ts}
    if overall in ("放水时段（提高勝率）","放水时段（提高胜率）"): # handling minor spelling difference
        overall = "放水时段（提高胜率）"
    if current and current.get("type") in ("放水时段（提高胜率）","中等胜率（中上）") and overall not in ("放水时段（提高胜率）","中等胜率（中上）"):
        # period ended
        start = current.get("start")
        end = now_ts
        duration_min = (end - start) / 60000.0
        print("Period ended. Duration min:", duration_min)
        # save period into history
        state.setdefault("periods", []).append({"type": current.get("type"), "start": start, "end": end, "duration_min": duration_min})
        # send telegram end message
        msg = f"[DG提醒] 放水已结束（{current.get('type')}）\\n开始: {datetime.fromtimestamp(start/1000, tz=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}\\n结束: {datetime.fromtimestamp(end/1000, tz=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}\\n共持续: {duration_min:.1f} 分钟"
        send_telegram(msg)
        state["current"] = None
        state["last_notify"] = now_ts
        save_state(state)
    elif overall in ("放水时段（提高胜率）","中等胜率（中上）"):
        # enter or continue period
        if not current or current.get("type") != overall:
            # new period
            if now_ts - last_notify < cooldown_ms:
                print("In cooldown, skipping notify")
            else:
                # send start notify with some stats and estimate
                # compute historical average duration if any
                periods = state.get("periods", [])
                durations = [p["duration_min"] for p in periods if p.get("duration_min") and p.get("type")==overall]
                avg_dur = statistics.mean(durations) if durations else None
                start_ts = now_ts
                # set state current
                state["current"] = {"type": overall, "start": start_ts, "last": start_ts}
                state["last_notify"] = now_ts
                save_state(state)
                # craft message
                est_txt = "暂无历史估算"
                if avg_dur:
                    est_txt = f"历史同类平均持续约 {avg_dur:.1f} 分钟，估计剩余 = {max(0, avg_dur):.1f} 分钟"
                msg = f"[DG提醒] 检测到 <{overall}>\\n匹配长/超长龙数: {meta.get('longCount',0)}, 超长龙: {meta.get('superCount',0)}\\n{est_txt}\\n现在: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                send_telegram(msg)
        else:
            # continuing period: update last
            state["current"]["last"] = now_ts
            save_state(state)
            # optionally send small heartbeat? We'll not spam - only start & end notifications.
    else:
        # not in target states -> do nothing, but we may update state.current if previously in a period (handled earlier)
        print("No active period detected; nothing to send.")
        save_state(state)

    # final save
    save_state(state)
    print("Done.")

if __name__ == "__main__":
    main()
