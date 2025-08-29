# main.py
# DG 实盘检测 + 历史回退提醒（用于 GitHub Actions）
# 说明: TG token/chat 通过环境变量注入 (TG_BOT_TOKEN / TG_CHAT_ID)
# state/history files: state.json, history_db.json, history_stats.json, last_summary.json

import os, sys, time, json, math, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------------- config ----------------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN_ENV = "TG_BOT_TOKEN"
TG_CHAT_ENV = "TG_CHAT_ID"

MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))
HISTORY_LOOKBACK_DAYS = int(os.environ.get("HISTORY_LOOKBACK_DAYS", "28"))
HISTORY_PROB_THRESHOLD = float(os.environ.get("HISTORY_PROB_THRESHOLD", "0.35"))  # fallback trigger threshold
TZ = timezone(timedelta(hours=8))

STATE_FILE = "state.json"
HISTORY_DB = "history_db.json"
HISTORY_STATS = "history_stats.json"
LAST_SUMMARY = "last_summary.json"

# ---------------- helper ----------------
def nowstr():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def send_telegram(text):
    token = os.environ.get(TG_TOKEN_ENV, "").strip()
    chat = os.environ.get(TG_CHAT_ENV, "").strip()
    if not token or not chat:
        log("Telegram token/chat 未配置，跳过发送。")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat, "text": text}, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功")
            return True
        else:
            log(f"Telegram 返回错误: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 失败: {e}")
        return False

def load_json(p, default):
    try:
        if not os.path.exists(p): return default
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        log(f"加载 {p} 出错: {e}")
        return default

def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------------- image utilities ----------------
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_color_points(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0,90,60]), np.array([10,255,255])
    lower2, upper2 = np.array([160,90,60]), np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb, upperb = np.array([85,60,40]), np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)
    points = []
    for mask, lab in [(mask_r,'B'), (mask_b,'P')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            M = cv2.moments(cnt)
            if M["m00"]==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,lab))
    return points

def cluster_points_to_boards(points, shape):
    h,w = shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)
    for x,y,_ in points:
        if 0<=y<h and 0<=x<w: mask[y,x] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
    big = cv2.dilate(mask, kernel, iterations=1)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(big, connectivity=8)
    rects = []
    for i in range(1, num):
        x0,y0,w0,h0 = stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP], stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]
        if w0 < 60 or h0 < 40: continue
        pad = 8
        x1 = max(0, x0-pad); y1 = max(0, y0-pad)
        x2 = min(w-1, x0+w0+pad); y2 = min(h-1, y0+h0+pad)
        rects.append((x1,y1, x2-x1, y2-y1))
    if not rects:
        cols = max(3, w//300); rows = max(2, h//180)
        cw = w//cols; ch = h//rows
        for r in range(rows):
            for c in range(cols):
                rects.append((c*cw, r*ch, cw, ch))
    return rects

def analyze_board(bgr, rect):
    x,y,w,h = rect
    crop = bgr[y:y+h, x:x+w]
    pts = detect_color_points(crop)
    pts_local = [(px,py,c) for (px,py,c) in pts]
    if not pts_local:
        return {"total":0, "maxRun":0, "category":"empty", "columns":[], "runs":[]}
    xs = [p[0] for p in pts_local]
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups=[]
    for idx in sorted_idx:
        xv = xs[idx]
        placed=False
        for g in col_groups:
            gxs = [pts_local[i][0] for i in g]
            if abs(np.mean(gxs)-xv) <= max(10, w//40):
                g.append(idx); placed=True; break
        if not placed:
            col_groups.append([idx])
    columns=[]
    for g in col_groups:
        col_pts = sorted([pts_local[i] for i in g], key=lambda t: t[1])
        seq = [p[2] for p in col_pts]
        columns.append(seq)
    flattened=[]
    maxlen = max((len(c) for c in columns), default=0)
    for r in range(maxlen):
        for col in columns:
            if r < len(col):
                flattened.append(col[r])
    runs=[]
    if flattened:
        cur={"color":flattened[0], "len":1}
        for k in range(1, len(flattened)):
            if flattened[k]==cur["color"]:
                cur["len"]+=1
            else:
                runs.append(cur); cur={"color":flattened[k], "len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    cat="other"
    if maxRun>=10: cat="super_long"
    elif maxRun>=8: cat="long"
    elif maxRun>=4: cat="longish"
    elif maxRun==1: cat="single"
    return {"total":len(flattened), "maxRun":maxRun, "category":cat, "columns":columns, "runs":runs}

def classify_overall(board_infos):
    longCount = sum(1 for b in board_infos if b["category"] in ("long","super_long"))
    superCount = sum(1 for b in board_infos if b["category"]=="super_long")
    longishCount = sum(1 for b in board_infos if b["category"]=="longish")
    def board_has_3consec_multicolumn(columns):
        col_runlens=[]
        for col in columns:
            if not col:
                col_runlens.append(0); continue
            ccur=col[0]; clen=1; maxc=1
            for t in col[1:]:
                if t==ccur: clen+=1
                else:
                    if clen>maxc: maxc=clen
                    ccur=t; clen=1
            if clen>maxc: maxc=clen
            col_runlens.append(maxc)
        for i in range(len(col_runlens)-2):
            if col_runlens[i]>=4 and col_runlens[i+1]>=4 and col_runlens[i+2]>=4:
                return True
        return False
    boards_with_multicol = sum(1 for b in board_infos if board_has_3consec_multicolumn(b["columns"]))
    boards_with_long = sum(1 for b in board_infos if b["maxRun"]>=8)
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount
    if boards_with_multicol >= 3 and boards_with_long >= 2:
        return "中等胜率（中上）", boards_with_long, sum(1 for b in board_infos if b["category"]=="super_long")
    totals = [b["total"] for b in board_infos]
    if board_infos and sum(1 for t in totals if t < 6) >= len(board_infos)*0.6:
        return "胜率调低 / 收割时段", sum(1 for b in board_infos if b["maxRun"]>=8), sum(1 for b in board_infos if b["category"]=="super_long")
    return "胜率中等（平台收割中等时段）", sum(1 for b in board_infos if b["maxRun"]>=8), sum(1 for b in board_infos if b["category"]=="super_long")

# ---------------- Playwright site capture (click Free + slide) ----------------
def capture_dg_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
        context = browser.new_context(viewport={"width":1280,"height":900})
        page = context.new_page()
        screenshot_bytes = None
        for url in DG_LINKS:
            try:
                log(f"打开: {url}")
                page.goto(url, timeout=35000)
                time.sleep(1.0)
                # try text buttons
                btn_texts = ["Free", "免费试玩", "免费", "Play Free", "试玩", "Free Play"]
                clicked=False
                for t in btn_texts:
                    try:
                        loc = page.locator(f"text={t}")
                        if loc.count()>0:
                            loc.first.click(timeout=4000); clicked=True; log(f"点击文本按钮: {t}"); break
                    except Exception:
                        continue
                # fallback try elements scanning
                if not clicked:
                    try:
                        els = page.locator("a,button")
                        c = min(80, els.count())
                        for i in range(c):
                            try:
                                txt = els.nth(i).inner_text().strip()
                                if "free" in txt.lower() or "试玩" in txt or "免费" in txt:
                                    els.nth(i).click(timeout=3000); clicked=True; log(f"点击 a/button 索引{i} 文本: {txt}"); break
                            except: continue
                    except: pass
                time.sleep(1.2)
                # try find slider
                slider_found=False
                try:
                    slider_selectors = ["input[type=range]","div[role=slider]","div[class*=slider]","div[class*=captcha]","div[class*=slide]"]
                    for sel in slider_selectors:
                        els = page.query_selector_all(sel)
                        if els and len(els)>0:
                            elem = els[0]; box = elem.bounding_box()
                            if box:
                                slider_found=True
                                x0 = box["x"]+2; y0 = box["y"] + box["height"]/2
                                x1 = box["x"] + box["width"] - 4
                                page.mouse.move(x0,y0); page.mouse.down(); page.mouse.move(x1,y0, steps=30); page.mouse.up()
                                log(f"对 selector {sel} 执行滑动"); time.sleep(1.5)
                                break
                except Exception as e:
                    log(f"查找滑动元素异常: {e}")
                # fallback image-based slider
                if not slider_found:
                    try:
                        ss = page.screenshot(full_page=True)
                        img = pil_from_bytes(ss); bgr = cv_from_pil(img)
                        hh, ww = bgr.shape[:2]
                        lower = bgr[int(hh*0.25):int(hh*0.85), int(ww*0.08):int(ww*0.92)]
                        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
                        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        best=None; best_area=0
                        for cnt in contours:
                            x,y,ww_,hh_ = cv2.boundingRect(cnt); area = ww_*hh_
                            if area>best_area and ww_>hh_*3 and ww_>40:
                                best=(x,y,ww_,hh_); best_area=area
                        if best:
                            bx,by,bw,bh = best
                            px = int(ww*0.08)+bx; py = int(hh*0.25)+by
                            x0 = px+4; y0 = py + bh//2; x1 = px + bw - 4
                            page.mouse.move(x0,y0); page.mouse.down(); page.mouse.move(x1,y0, steps=30); page.mouse.up()
                            log("执行滑动（图像辅助）"); time.sleep(1.2)
                    except Exception as e:
                        log(f"图像辅助滑动异常: {e}")
                # try several retries to detect many dots
                for attempt in range(6):
                    try:
                        ss = page.screenshot(full_page=True)
                        pil = pil_from_bytes(ss); bgr = cv_from_pil(pil)
                        pts = detect_color_points(bgr)
                        log(f"截图尝试 {attempt+1}: 点数 {len(pts)}")
                        if len(pts) >= 40:
                            screenshot_bytes = ss; log("进入实盘判定成功（点数充足）"); break
                        else:
                            time.sleep(2.0 + attempt)
                    except Exception as e:
                        log(f"截图分析异常: {e}")
                        time.sleep(2.0)
                if screenshot_bytes: break
            except PWTimeout as e:
                log(f"访问超时: {e}"); continue
            except Exception as e:
                log(f"访问/交互异常: {e}"); continue
        try: context.close()
        except: pass
        try: browser.close()
        except: pass
        return screenshot_bytes

# ---------------- historical fallback functions ----------------
def update_history_db_with_event(start_iso, duration_minutes, kind):
    db = load_json(HISTORY_DB, [])
    # append and keep last 60 events
    db.append({"start": start_iso, "duration_minutes": int(duration_minutes), "kind": kind})
    # keep only last 120 events
    db = db[-120:]
    save_json(HISTORY_DB, db)
    return db

def load_history_stats():
    return load_json(HISTORY_STATS, {})

def fallback_check_now_and_notify():
    stats = load_history_stats()
    if not stats or "counts" not in stats:
        log("无历史统计数据，无法基于历史做回退预判。")
        return False
    # compute minute-of-week index for now (0..10079)
    now = datetime.now(TZ)
    minute_of_week = now.weekday()*1440 + now.hour*60 + now.minute
    counts = stats["counts"]  # dict str(minute) -> occurrences across weeks
    weeks = max(1, stats.get("weeks",1))
    cnt = counts.get(str(minute_of_week), 0)
    prob = cnt / weeks
    log(f"历史回退：minute_of_week={minute_of_week}, count={cnt}, weeks={weeks}, prob={prob:.3f}")
    if prob >= HISTORY_PROB_THRESHOLD:
        # estimate average duration at this minute
        avg_dur = stats.get("avg_duration_minutes_by_minute", {}).get(str(minute_of_week), None)
        est_min = round(avg_dur) if avg_dur else 10
        est_end = (now + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
        msg = f"🔔 [DG历史推断提醒] 当前时间基于过去{HISTORY_LOOKBACK_DAYS}天的市场数据，为高概率放水窗口 (prob={prob:.2f})。\n估计结束: {est_end}（约 {est_min} 分钟，基于历史）\n说明: 此通知为“替补”历史推断（实时抓取失败），请谨慎验证。"
        send_telegram(msg)
        return True
    else:
        log("历史概率不足，回退不触发提醒。")
        return False

# ---------------- main ----------------
def main():
    log("单次检测开始")
    state = load_json(STATE_FILE, {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]})
    screenshot = None
    try:
        screenshot = capture_dg_page()
    except Exception as e:
        log(f"capture_dg_page 异常: {e}\n{traceback.format_exc()}")
    if not screenshot:
        log("未能获取实盘截图，尝试历史回退判定...")
        did = fallback_check_now_and_notify()
        # save state and exit
        save_json(STATE_FILE, state)
        return
    # analyze screenshot
    pil = pil_from_bytes(screenshot); bgr = cv_from_pil(pil)
    pts = detect_color_points(bgr)
    log(f"整页点数: {len(pts)}")
    if len(pts) < 20:
        log("点数过少，可能未真正进入实盘；尝试历史回退判定...")
        did = fallback_check_now_and_notify()
        save_json(STATE_FILE, state)
        return
    rects = cluster_points_to_boards(pts, bgr.shape)
    log(f"聚类出桌子: {len(rects)}")
    boards = []
    for r in rects:
        info = analyze_board(bgr, r)
        boards.append(info)
    overall, longCount, superCount = classify_overall(boards)
    log(f"局势判定 -> {overall} (长龙/超龙={longCount}, 超龙={superCount})")
    now_iso = datetime.now(TZ).isoformat()
    was_active = state.get("active", False)
    is_active_now = overall in ("放水时段（提高胜率）","中等胜率（中上）")
    if is_active_now and not was_active:
        # start event
        state = {"active": True, "kind": overall, "start_time": now_iso, "last_seen": now_iso, "history": state.get("history", [])}
        # send msg with historical estimate
        hist = load_json(HISTORY_DB, [])
        durations = [h["duration_minutes"] for h in hist if h.get("duration_minutes",0)>0]
        est_min = round(sum(durations)/len(durations)) if durations else 10
        est_end = (datetime.now(TZ) + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
        msg = f"🔔 [DG提醒] {overall} 開始\n時間: {now_iso}\n长龙/超龙 桌數: {longCount} (超龙:{superCount})\n估計結束: {est_end}（約 {est_min} 分鐘，基於歷史）"
        send_telegram(msg)
        log("发送开始通知")
        save_json(STATE_FILE, state)
        # also append to history_db as placeholder start (duration unknown yet)
        db = load_json(HISTORY_DB, [])
        db.append({"start": now_iso, "duration_minutes": None, "kind": overall})
        db = db[-120:]
        save_json(HISTORY_DB, db)
    elif is_active_now and was_active:
        state["last_seen"] = now_iso
        state["kind"] = overall
        save_json(STATE_FILE, state)
        log("事件仍在进行，更新 last_seen")
    elif not is_active_now and was_active:
        # ended
        start = datetime.fromisoformat(state.get("start_time"))
        end = datetime.now(TZ)
        duration = round((end - start).total_seconds() / 60.0)
        hist = state.get("history", [])
        hist.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration})
        hist = hist[-200:]
        new_state = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": hist}
        save_json(STATE_FILE, new_state)
        # update history_db: find last event with None duration and set duration
        db = load_json(HISTORY_DB, [])
        for i in range(len(db)-1, -1, -1):
            if db[i].get("duration_minutes") in (None, 0):
                db[i]["duration_minutes"] = duration
                break
        save_json(HISTORY_DB, db)
        msg = f"✅ [DG提醒] {state.get('kind')} 已結束\n開始: {state.get('start_time')}\n結束: {end.isoformat()}\n實際持續: {duration} 分鐘"
        send_telegram(msg)
        log("发送结束通知并记录历史")
    else:
        save_json(STATE_FILE, state)
        log("当前非放水/非中上时段，不发送通知")
    # save debug summary
    summary = {"ts": now_iso, "overall": overall, "longCount": longCount, "superCount": superCount, "boards": boards[:30]}
    save_json(LAST_SUMMARY, summary)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"主程序异常: {e}\n{traceback.format_exc()}")
        sys.exit(1)
