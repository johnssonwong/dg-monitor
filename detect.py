# detect.py
# 说明：使用 Playwright + OpenCV，从 DG 页面截屏并检测局势，按照用户规则判断并发 Telegram 提醒。
# 注意：此脚本会在 GitHub Actions 中运行（headless），并会尝试提交 state.json 用于储存临时状态与统计。

import os, sys, json, time, math
import datetime
import requests
import numpy as np
import cv2
from playwright.sync_api import sync_playwright

# =========================
# 配置区（你要我自动填入的值）
# =========================
TELEGRAM_BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TELEGRAM_CHAT_ID = "485427847"
DG_URLS = ["https://dg18.co/wap/", "https://dg18.co/"]
# 如果你把 token/chat 存在 Secrets，请把下面两项改为：
# TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
# TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# 最低满足“放水（提高胜率）”的 长龙/超长龙 桌数（你之前设定为 >=3）
MIN_BOARDS_FOR_PLAYS = 3
# 中等胜率（中上）触发条件： >=2 张长龙或超长龙 即触发小提醒
MID_LONG_REQ = 2

# 冷却时间（分钟）—— 若提醒一次后等待 cooldown 分钟才允许再次提醒
COOLDOWN_MINUTES = 10

# state 文件名（用于保存放水开始时间、历史事件等）
STATE_FILE = "state.json"
# =========================

# Helper: load/save state
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE,'r',encoding='utf-8'))
        except:
            pass
    # default
    return {
        "ongoing": False,
        "start_ts": None,
        "last_alert_ts": None,
        "cooldown_until": None,
        "episodes": []  # store past durations in seconds
    }

def save_state(s):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# Send Telegram
def telegram_send(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

# Image analysis helpers
def hsv_mask_for_red_blue(bgr_img):
    # return mask_red, mask_blue
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # red can wrap hue 0 and 180
    lower1 = np.array([0, 100, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160,100,50])
    upper2 = np.array([179,255,255])
    mask_red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue range
    lowerb = np.array([95, 100, 50])
    upperb = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lowerb, upperb)
    return mask_red, mask_blue

def find_dense_regions(mask, grid=60, thr_cells=30):
    # coarse scan to find high-density blocks; returns list of bboxes (x,y,w,h) in image coordinates
    h, w = mask.shape
    cw = grid
    ch = grid
    cells = []
    for y in range(0, h, ch):
        for x in range(0, w, cw):
            sub = mask[y:y+ch, x:x+cw]
            cnt = int(cv2.countNonZero(sub))
            cells.append(((x,y,cw,ch), cnt))
    # pick cells with cnt > some threshold
    vals = [c for (_,c) in cells]
    if len(vals)==0:
        return []
    thr = max(10, int(np.percentile(vals, 75)))  # adaptive threshold
    hits = [c[0] for c in cells if c[1] >= thr]
    # merge adjacent hits into bounding boxes
    boxes = []
    for (x,y,wc,hc) in hits:
        merged = False
        for b in boxes:
            bx,by,bw,bh = b
            # if overlapping or close
            if not (x > bx + bw + cw or x + wc < bx - cw or y > by + bh + ch or y + hc < by - ch):
                # merge
                nx = min(bx, x)
                ny = min(by, y)
                nx2 = max(bx + bw, x + wc)
                ny2 = max(by + bh, y + hc)
                b[0], b[1], b[2], b[3] = nx, ny, nx2-nx, ny2-ny
                merged = True
                break
        if not merged:
            boxes.append([x,y,wc,hc])
    # convert to tuple list
    return [(int(b[0]),int(b[1]),int(b[2]),int(b[3])) for b in boxes]

def detect_beads_and_runs(img):
    # img: full BGR image (numpy)
    mask_r, mask_b = hsv_mask_for_red_blue(img)
    # combine to find busy areas
    combined = cv2.bitwise_or(mask_r, mask_b)
    boxes = find_dense_regions(combined, grid=60)
    # for each box, find bead centers and colors
    results = []
    for (x,y,w,h) in boxes:
        sub = img[y:y+h, x:x+w]
        mr, mb = hsv_mask_for_red_blue(sub)
        # morphological clean
        kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mr = cv2.morphologyEx(mr, cv2.MORPH_OPEN, kr)
        mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, kr)
        # find contours for red and blue
        contours_r, _ = cv2.findContours(mr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_b, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours_r:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            (cx,cy), radius = cv2.minEnclosingCircle(cnt)
            centers.append((int(cx), int(cy), 'B'))  # B = 庄 (red)
        for cnt in contours_b:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            (cx,cy), radius = cv2.minEnclosingCircle(cnt)
            centers.append((int(cx), int(cy), 'P'))  # P = 闲 (blue)
        if len(centers) == 0:
            continue
        # cluster by x to columns
        centers_sorted = sorted(centers, key=lambda v: v[0])
        xs = [c[0] for c in centers_sorted]
        # cluster threshold depends on width
        thr = max(12, int(w/20))
        groups = []
        cur = [centers_sorted[0]]
        for c in centers_sorted[1:]:
            if c[0] - cur[-1][0] <= thr:
                cur.append(c)
            else:
                groups.append(cur)
                cur = [c]
        groups.append(cur)
        # for each column, sort by y and output sequence top->bottom
        sequences = []
        for g in groups:
            col = sorted(g, key=lambda v: v[1])
            seq = [p[2] for p in col]
            sequences.append(seq)
        # flatten by reading row-wise across columns: first items of col1, col2..., then second rows...
        maxlen = max((len(s) for s in sequences), default=0)
        flattened = []
        for r in range(maxlen):
            for col in sequences:
                if r < len(col):
                    flattened.append(col[r])
        # compute runs
        runs = []
        if flattened:
            curc = flattened[0]; L = 1
            for c in flattened[1:]:
                if c == curc:
                    L += 1
                else:
                    runs.append((curc, L))
                    curc = c; L = 1
            runs.append((curc, L))
        results.append({
            "box": (x,y,w,h),
            "bead_count": len(flattened),
            "flattened": flattened,
            "runs": runs,
            "max_run": max((r[1] for r in runs), default=0)
        })
    return results

# 判定整体局势
def classify_overall(board_results):
    # board_results: list of per-board dicts with max_run
    long_count = sum(1 for b in board_results if b['max_run'] >= 8)  # 长龙 >=8
    super_count = sum(1 for b in board_results if b['max_run'] >= 10) # 超长龙 >=10
    longish_count = sum(1 for b in board_results if 4 <= b['max_run'] < 8) # 长连 >=4 <8
    # Determine:
    if long_count + super_count >= MIN_BOARDS_FOR_PLAYS:
        return "放水时段（提高胜率）", long_count + super_count, super_count
    elif (long_count + super_count) >= MID_LONG_REQ and longish_count > 0:
        return "中等胜率（中上）", long_count + super_count, super_count
    else:
        # sparse check:
        sparse = sum(1 for b in board_results if b['bead_count'] < 6)
        if len(board_results)>0 and sparse >= len(board_results)*0.6:
            return "胜率调低 / 收割时段", long_count + super_count, super_count
        else:
            return "胜率中等（平台收割中等时段）", long_count + super_count, super_count

# Main run
def run_once():
    state = load_state()
    now_ts = int(time.time())
    # check cooldown
    if state.get("cooldown_until") and now_ts < state["cooldown_until"]:
        # still in cooldown --- we still want to check for end of 放水 to send end alert, so continue
        pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":1366,"height":768})
        success = False
        # try each DG url
        for u in DG_URLS:
            try:
                page.goto(u, timeout=30000)
                time.sleep(2)
                # try to click Free / 免费试玩 链接
                # attempt multiple selectors
                for sel in ["text=Free", "text=免费试玩", "text=Free Play", "a:has-text('Free')", "a:has-text('免费')"]:
                    try:
                        page.click(sel, timeout=3000)
                        time.sleep(2)
                        break
                    except:
                        pass
                # try to handle scroll slider (best-effort)
                # find potential slider element by common attributes
                try:
                    # attempt a generic drag: locate any element with role slider or input range
                    # fallback: try to drag near coordinates where slider often appears
                    page.mouse.move(300, 500)
                    page.mouse.down()
                    page.mouse.move(1000, 500, steps=10)
                    page.mouse.up()
                    time.sleep(2)
                except:
                    pass
                # WAIT for the table UI to load
                time.sleep(2)
                # final screenshot
                screenshot = page.screenshot(full_page=True)
                success = True
                break
            except Exception as e:
                # try next url
                print("访问 DG URL 失败：", u, e)
                continue
        browser.close()

    if not success:
        print("无法访问 DG 页面或截图失败")
        return

    # save raw screenshot
    with open("screen.png", "wb") as f:
        f.write(screenshot)

    # load with OpenCV
    arr = np.frombuffer(screenshot, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # resize to max width to speed up if needed
    maxw = 1400
    h,w = img.shape[:2]
    if w > maxw:
        scale = maxw / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    boards = detect_beads_and_runs(img)
    overall, longcount, supercount = classify_overall(boards)
    print("判定：", overall, "长龙数:", longcount, " 超长龙:", supercount)
    # Logging summary:
    summary_lines = []
    for i,b in enumerate(boards, start=1):
        summary_lines.append(f"桌{i}: beads={b['bead_count']}, max_run={b['max_run']}, runs={b['runs'][:6]}")
    # handle state transitions and telegram notifications
    now = datetime.datetime.utcnow().timestamp()
    cd_until = state.get("cooldown_until") or 0
    # Detect start:
    if overall in ("放水时段（提高胜率）", "中等胜率（中上）"):
        # If previously not ongoing, start new episode and send start alert (if not in cooldown)
        if not state.get("ongoing", False):
            # check cooldown
            if now_ts < cd_until:
                print("在冷却期内，检测到放水但暂不提醒")
            else:
                # start episode
                state["ongoing"] = True
                state["start_ts"] = now_ts
                state["last_alert_ts"] = now_ts
                state["cooldown_until"] = now_ts + COOLDOWN_MINUTES*60
                save_state(state)
                # build message
                msg = f"[DG提醒] 局势判定：{overall}\n长龙数: {longcount}, 超长龙: {supercount}\nTime: {datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}\n说明：请手动入场并按照策略执行。"
                ok, resp = telegram_send(msg)
                print("发送提醒：", ok, resp)
        else:
            # already ongoing; optionally update last_alert_ts but skip sending if in cooldown
            print("已有放水进行中，继续观察。")
    else:
        # not a play period
        if state.get("ongoing", False):
            # previously ongoing but now ended -> send end message with duration & update episodes
            start_ts = state.get("start_ts")
            if start_ts:
                dur = now_ts - start_ts
                state["ongoing"] = False
                state["start_ts"] = None
                state["last_alert_ts"] = now_ts
                state["episodes"].append(int(dur))
                state["cooldown_until"] = now_ts + COOLDOWN_MINUTES*60
                save_state(state)
                minutes = int(dur/60)
                msg = f"[DG提醒] 放水已结束。\n持续时长: {minutes} 分钟 ({dur} 秒)\n结束时间: {datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
                ok, resp = telegram_send(msg)
                print("发送结束提醒：", ok, resp)
        else:
            print("目前不属于放水或中上时段，不提醒。")

    # save a local summary file for debugging
    with open("summary.log","a",encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} 判定={overall} 长龙={longcount} 超={supercount} boards={len(boards)}\n")
    save_state(state)

if __name__ == "__main__":
    run_once()
