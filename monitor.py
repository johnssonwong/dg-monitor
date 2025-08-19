# monitor.py
# DG 自动监测脚本（用于 GitHub Actions）
# 已内置：Telegram Token, Chat ID, DG links, timezone (UTC+8)
# 要求：GitHub Actions runner 会执行此脚本，每5分钟一轮
# 作用：打开 DG 网站、进入 Free 模式、截图、用 OpenCV 检测局势、按规则判定并用 Telegram 通知。并将状态写入 .state.json（用于跨次运行持久化）

import os, sys, json, math, time, statistics, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

# -------------- 用户常量（已按你要求自动填入） --------------
TG_BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TG_CHAT_ID    = "485427847"
DG_LINK_1     = "https://dg18.co/wap/"
DG_LINK_2     = "https://dg18.co/"
LOCAL_TZ      = timezone(timedelta(hours=8))   # Malaysia UTC+8
# ----------------- 规则阈值（可按需调整） -----------------
MIN_BOARDS_FOR_PAWATER = 3   # 满足放水：至少3张桌为 长龙/超长龙（可在 1-5 之间调整）
MID_LONG_REQ = 2             # 中等胜率：至少 >=2 张长龙或超长龙
COOLDOWN_MINUTES = 10        # 若已触发提醒后，等待 cooldown 后才允许下一次开始提醒；用于防止重复
STATE_FILE = ".state.json"
# --------------------------------------------------------

# Dependencies: playwright, pillow, numpy, opencv-python-headless
# The GitHub Actions workflow will install these prior to running.

def now_ts():
    return int(time.time())

def now_dt():
    return datetime.now(LOCAL_TZ)

def load_state():
    p = Path(STATE_FILE)
    if not p.exists(): 
        return {"mode":"idle","alert_start":None,"alert_type":None,"durations":[]}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"mode":"idle","alert_start":None,"alert_type":None,"durations":[]}

def save_state(state):
    Path(STATE_FILE).write_text(json.dumps(state, indent=2, ensure_ascii=False))

# send telegram helper
def send_telegram(text, image_bytes=None):
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id":TG_CHAT_ID, "text": text})
        ok = r.ok
        if image_bytes is not None:
            # try send photo if provided
            try:
                urlp = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
                files = {'photo': ('screenshot.jpg', image_bytes, 'image/jpeg')}
                data = {'chat_id': TG_CHAT_ID, 'caption': text}
                rp = requests.post(urlp, data=data, files=files, timeout=30)
                if rp.ok:
                    return True
            except Exception as e:
                print("send photo failed", e)
        return ok
    except Exception as e:
        print("send telegram failed", e)
        return False

# ----------------------------------------------
# Playwright automation + screenshot
# ----------------------------------------------
def capture_dg_screenshot(tmp_path="/tmp/dg_shot.png", headless=True):
    """
    Use Playwright to open DG, click Free, attempt to slide slider, wait for game area, and take full page screenshot.
    Returns path to saved PNG.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    shot_path = tmp_path
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--no-sandbox","--disable-setuid-sandbox"])
        context = browser.new_context(java_script_enabled=True, viewport={"width":1366,"height":800})
        page = context.new_page()
        # Some DG entry points may redirect. Try both.
        tried = []
        for entry in [DG_LINK_1, DG_LINK_2]:
            try:
                page.goto(entry, timeout=30000)
                tried.append(entry)
                time.sleep(1)
                # try to find "Free" or "免费试玩" link/button
                found=False
                for sel in ["text=Free","text=免费试玩","text=免费","a:has-text('Free')","button:has-text('Free')","text=TRY FOR FREE"]:
                    try:
                        el = page.query_selector(sel)
                        if el:
                            try:
                                el.click(timeout=5000)
                                found=True
                                break
                            except Exception:
                                pass
                    except Exception:
                        pass
                # After clicking Free a new page may open or a slider appears.
                # Wait for navigation or slider.
                time.sleep(2)
                # Attempt to handle the slider (common patterns)
                # Try several common slider selectors
                slider_selectors = [
                    ".geetest_slider_button", ".sliderButton",".slider","div[class*='slider']",
                    "div[class*='geetest']", "div.geetest_slider_button", ".rc-slider-handle"
                ]
                # Wait a bit for slider to appear
                for s in slider_selectors:
                    try:
                        el = page.query_selector(s)
                        if el:
                            box = el.bounding_box()
                            # attempt a drag
                            if box:
                                x = box["x"] + box["width"]/2
                                y = box["y"] + box["height"]/2
                                # attempt drag to right in several steps
                                page.mouse.move(x,y)
                                page.mouse.down()
                                page.mouse.move(x+250, y, steps=30)
                                page.mouse.up()
                                time.sleep(2)
                    except PWTimeout:
                        pass
                    except Exception:
                        pass
                # fallback: try to click "I agree" or "进入"
                for sel2 in ["text=进入","text=Start","text=Enter","text=I agree","text=同意"]:
                    try:
                        el = page.query_selector(sel2)
                        if el:
                            el.click()
                            time.sleep(2)
                    except Exception:
                        pass
                # Wait until page shows many SVG/canvas elements or game area, else just capture screenshot
                try:
                    page.wait_for_timeout(1500)
                except Exception:
                    pass
                # take full page screenshot
                page.screenshot(path=shot_path, full_page=True)
                browser.close()
                return shot_path
            except Exception as e:
                print("entry failed", entry, e)
                # try next entry
                try:
                    page.close()
                except:
                    pass
        browser.close()
    raise RuntimeError("All DG entry attempts failed: tried " + ",".join(tried))

# ----------------------------------------------
# Image analysis: detect red/blue beads and compute runs
# ----------------------------------------------
def analyze_screenshot(path):
    """
    Return a summary list of board stats: for each detected board region, compute flattened bead sequence and runs.
    Will attempt to detect high-density red/blue areas and split into regions (heuristic).
    """
    import cv2
    import numpy as np
    from PIL import Image
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback to PIL open
        img_p = Image.open(path).convert("RGB")
        arr = np.array(img_p)[:,:,::-1].copy()
        img = arr
    h,w = img.shape[:2]

    # downscale for faster processing if huge
    scale = 1.0
    if w > 1600:
        scale = 1600.0 / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        h,w = img.shape[:2]

    # color masks for red and blue (BGR space)
    # red mask
    lower_red1 = np.array([0,0,120])
    upper_red1 = np.array([80,80,255])
    # blue mask
    lower_blue = np.array([120,0,0])
    upper_blue = np.array([255,120,120])

    mask_r = cv2.inRange(img, lower_red1, upper_red1)
    mask_b = cv2.inRange(img, lower_blue, upper_blue)
    mask_any = cv2.bitwise_or(mask_r, mask_b)

    # morphological to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_any = cv2.morphologyEx(mask_any, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_any = cv2.morphologyEx(mask_any, cv2.MORPH_DILATE, kernel, iterations=1)

    # find contours of high-density areas -> candidate board regions
    contours, _ = cv2.findContours(mask_any, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        if ww*hh < 2000: continue
        # expand a bit
        pad = 8
        x = max(0, x-pad); y = max(0,y-pad)
        ww = min(w-x, ww+pad*2); hh = min(h-y, hh+pad*2)
        regions.append((x,y,ww,hh))
    # if no regions found, fallback to splitting by grid (assume multi-board layout)
    if not regions:
        cols = 4
        rows = 4
        cw = w//cols; ch = h//rows
        for r in range(rows):
            for c in range(cols):
                regions.append((c*cw, r*ch, cw, ch))

    boards = []
    for (x,y,ww,hh) in regions:
        sub = img[y:y+hh, x:x+ww]
        mr = cv2.inRange(sub, lower_red1, upper_red1)
        mb = cv2.inRange(sub, lower_blue, upper_blue)
        # detect blobs
        cnts_r,_ = cv2.findContours(mr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_b,_ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for c in cnts_r:
            x2,y2,w2,h2 = cv2.boundingRect(c)
            if w2*h2 < 30: continue
            centers.append(("B", x2 + w2//2, y2 + h2//2))
        for c in cnts_b:
            x2,y2,w2,h2 = cv2.boundingRect(c)
            if w2*h2 < 30: continue
            centers.append(("P", x2 + w2//2, y2 + h2//2))
        # sort centers by x then y to approximate bead-plate reading
        centers.sort(key=lambda t:(t[1], t[2]))
        # group by approximate columns using x gaps
        xs = [c[1] for c in centers]
        cols = []
        if xs:
            curcol=[centers[0]]
            for c in centers[1:]:
                if abs(c[1]-curcol[-1][1]) > 30:  # threshold for new column
                    cols.append(curcol)
                    curcol=[c]
                else:
                    curcol.append(c)
            cols.append(curcol)
        # construct flattened sequence top-to-bottom per column left-to-right
        flattened=[]
        for col in cols:
            col_sorted = sorted(col, key=lambda z:z[2])  # by y
            for bead in col_sorted:
                flattened.append(bead[0])
        # compute runs
        runs=[]
        if flattened:
            cur = {'color': flattened[0], 'len':1}
            for ch in flattened[1:]:
                if ch == cur['color']:
                    cur['len'] += 1
                else:
                    runs.append(cur)
                    cur = {'color':ch,'len':1}
            runs.append(cur)
        boards.append({
            "bbox":(int(x/scale), int(y/scale), int(ww/scale), int(hh/scale)),
            "beads": len(flattened),
            "flattened": flattened,
            "runs": runs,
            "max_run": max([r['len'] for r in runs]) if runs else 0
        })

    return boards

# classify overall using user's saved rules
def classify_overall(boards):
    # counts
    long_count = sum(1 for b in boards if b['max_run'] >= 8)
    super_count = sum(1 for b in boards if b['max_run'] >= 10)
    longish_count = sum(1 for b in boards if b['max_run'] >= 4)
    sparse_count = sum(1 for b in boards if b['beads'] < 6)
    total = len(boards)
    # apply rules
    if long_count >= MIN_BOARDS_FOR_PAWATER:
        return "放水时段（提高胜率）", long_count, super_count
    if long_count >= MID_LONG_REQ and longish_count>0:
        return "中等胜率（中上）", long_count, super_count
    if sparse_count >= total * 0.6:
        return "胜率调低 / 收割时段", long_count, super_count
    return "胜率中等（平台收割中等时段）", long_count, super_count

# persist state by committing to repo (when running in GH Actions with GITHUB_TOKEN)
def git_commit_state(msg):
    try:
        # use git to add/commit/push state file if git environment exists
        os.system("git config user.email 'action@github.com' || true")
        os.system("git config user.name 'github-action' || true")
        os.system("git add "+STATE_FILE+" || true")
        os.system("git commit -m "+repr(msg)+" || true")
        # push using provided token (GITHUB_TOKEN is automatically provided in Actions)
        # Use origin URL as already configured by checkout action.
        os.system("git push origin HEAD:main || git push || true")
    except Exception as e:
        print("git commit failed", e)

def main():
    print("DG monitor starting at", now_dt().isoformat())
    state = load_state()
    try:
        shot = capture_dg_screenshot(tmp_path="/tmp/dg_shot.png", headless=True)
    except Exception as e:
        tb = traceback.format_exc()
        print("capture failed:", e)
        send_telegram("[DG监测] 无法访问 DG 或捕获截图，错误: " + str(e) + "\n" + tb)
        return

    boards = analyze_screenshot(shot)
    overall, long_count, super_count = classify_overall(boards)
    print("Detected overall:", overall, "long_count:", long_count, "super_count:", super_count)
    # prepare readable snippet
    summary_lines = [f"检测时间: {now_dt().strftime('%Y-%m-%d %H:%M:%S')}", f"判定: {overall}", f"长龙/超长龙: {long_count}/{super_count}", f"总检测桌数: {len(boards)}"]
    text_summary = "\n".join(summary_lines)

    # load state
    now = now_ts()
    if state.get("mode") == "idle":
        # only trigger start alert if we are in a "good" state
        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            # check cooldown: ensure last alert_end + cooldown passed or never alerted.
            last_start = state.get("alert_start")
            last_alert_time = int(last_start) if last_start else 0
            # If we previously had a start recorded but no end (unexpected), allow new start if enough time
            cooldown = COOLDOWN_MINUTES*60
            # if enough time since last alert_end (we store end into durations?) for simplicity we allow
            # Start alert
            state["mode"] = "alert"
            state["alert_start"] = now
            state["alert_type"] = overall
            save_state(state)
            git_commit_state(f"Start alert {overall} at {now}")
            # compute estimate using historical durations if available
            est_text = "尚无历史估算数据，无法可靠预测结束时间。"
            if state.get("durations"):
                avg_sec = int(statistics.mean(state["durations"]))
                est_end = datetime.now(LOCAL_TZ) + timedelta(seconds=avg_sec)
                est_text = f"历史平均放水时长约 {avg_sec//60} 分钟，估计结束时间: {est_end.strftime('%Y-%m-%d %H:%M:%S')} (约 {avg_sec//60} 分钟后)"
            # send start message with screenshot attached
            with open(shot, "rb") as f:
                img_bytes = f.read()
            start_text = f"[DG开始提醒] 判定：{overall}\n长龙/超长龙: {long_count}/{super_count}\n{est_text}\n\n实时判定依据已启用（每5分钟检测）。"
            send_telegram(start_text, image_bytes=img_bytes)
            print("Started alert and sent telegram.")
        else:
            print("Not a start condition. No alert.")
    else:
        # mode == alert (we are currently in alert). If condition ends, send end message and append duration.
        if overall not in ("放水时段（提高胜率）","中等胜率（中上）"):
            # ended
            start_ts = state.get("alert_start") or now
            dur = now - int(start_ts)
            minutes = dur//60
            # save duration to history
            state.setdefault("durations",[]).append(dur)
            # cap history length
            if len(state["durations"]) > 50:
                state["durations"] = state["durations"][-50:]
            state["mode"] = "idle"
            state["alert_start"] = None
            prev_type = state.get("alert_type")
            state["alert_type"] = None
            save_state(state)
            git_commit_state(f"End alert {prev_type} duration {dur}")
            end_text = f"[DG结束提醒] {prev_type} 已结束。持续约 {minutes} 分钟。\n检测结束时间: {now_dt().strftime('%Y-%m-%d %H:%M:%S')}\n\n后续将以历史平均时长改善估算。"
            # send final message and screenshot
            with open(shot,"rb") as f:
                img_bytes = f.read()
            send_telegram(end_text, image_bytes=img_bytes)
            print("Alert ended. Sent end telegram.")
        else:
            # still ongoing — optionally send periodic heartbeat? we will not spam; do nothing
            print("Alert still ongoing; no new telegram. (in alert state)")

    # done
    print("Run finished.")

if __name__ == "__main__":
    main()
