# main.py  --- 改进版（Playwright + OpenCV）
# 功能：尽最大努力自动进入 DG 实盘（点击 Free -> 滑动安全条 -> 截图）；
#       若进入成功则对局面做视觉判定（放水 / 中等胜率（中上）等）并发送 Telegram；
#       若无法进入实盘，启用历史替补预测（若历史数据足够），否则记录并开始收集历史。
#
# 注意：请将 TG_BOT_TOKEN 与 TG_CHAT_ID 存为 GitHub Secrets，并在 workflow 中注入为环境变量。
#       我无法保证 100% 成功——若失败，请把 Actions 的完整日志和 repo 中 last_summary.json 发给我，我会给你修正版。

import os, sys, time, json, math, random, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------- CONFIG ----------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN_ENV = "TG_BOT_TOKEN"
TG_CHAT_ENV = "TG_CHAT_ID"

MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))
STATE_FILE = "state.json"
SUMMARY_FILE = "last_summary.json"
LAST_SCREENSHOT = "last_screenshot.png"

TZ = timezone(timedelta(hours=8))  # Malaysia UTC+8

# Historical prediction params
MIN_HISTORY_EVENTS_FOR_PRED = 3
PRED_BUCKET_MINUTES = 15
PRED_LEAD_MINUTES = 10

# Image detection tuning (can tweak if detection misses)
MIN_POINTS_FOR_REAL = 40  # if detected points < this, consider not in real-play page
DILATE_KERNEL_SIZE = 40   # used to group board regions

# ---------- UTIL ----------
def now_tz(): return datetime.now(TZ)
def nowstr(): return now_tz().strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{nowstr()}] {s}", flush=True)

# ---------- TELEGRAM ----------
def send_telegram(text):
    token = os.environ.get(TG_TOKEN_ENV, "").strip()
    chat = os.environ.get(TG_CHAT_ENV, "").strip()
    if not token or not chat:
        log("Telegram not configured (TG_BOT_TOKEN / TG_CHAT_ID missing).")
        return False
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             data={"chat_id": chat, "text": text}, timeout=20)
        j = resp.json()
        if not j.get("ok"):
            log(f"Telegram response not ok: {j}")
            return False
        log("Telegram message sent.")
        return True
    except Exception as e:
        log(f"Error sending Telegram: {e}")
        return False

# ---------- STATE ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}
    try:
        with open(STATE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(s,f,ensure_ascii=False,indent=2)

# ---------- IMAGE HELPERS ----------
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_color_points(bgr):
    """Return list of (x,y,label) where label 'B' for red(Banker) and 'P' for blue(Player)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # red range
    lower1, upper1 = np.array([0,90,60]), np.array([10,255,255])
    lower2, upper2 = np.array([160,90,60]), np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue range
    lowerb, upperb = np.array([85,60,40]), np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)

    points=[]
    for mask,label in [(mask_r,'B'),(mask_b,'P')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            M = cv2.moments(cnt)
            if M['m00']==0: continue
            cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
            points.append((cx,cy,label))
    return points, mask_r, mask_b

def cluster_points_to_boards(points, img_shape):
    h,w = img_shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)
    for x,y,_ in points:
        if 0<=y<h and 0<=x<w:
            mask[y,x] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE,DILATE_KERNEL_SIZE))
    big = cv2.dilate(mask, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(big, connectivity=8)
    rects=[]
    for i in range(1,num_labels):
        x,y,w_,h_ = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w_ < 60 or h_ < 40: continue
        pad = 8
        x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w-1,x+w_+pad); y1=min(h-1,y+h_+pad)
        rects.append((x0,y0,x1-x0,y1-y0))
    if not rects:
        # fallback grid split
        cols = max(3,w//300); rows = max(2,h//200)
        cw = w//cols; ch = h//rows
        for r in range(rows):
            for c in range(cols):
                rects.append((c*cw, r*ch, cw, ch))
    return rects

def analyze_board(bgr, rect):
    x,y,w,h = rect
    crop = bgr[y:y+h, x:x+w]
    pts,_,_ = detect_color_points(crop)
    pts_local = [(px,py,c) for (px,py,c) in pts]
    if not pts_local:
        return {"total":0,"maxRun":0,"category":"empty","columns":[],"runs":[]}
    xs = [p[0] for p in pts_local]
    idx_sorted = sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups=[]
    for idx in idx_sorted:
        xv = xs[idx]
        placed=False
        for g in col_groups:
            gxs=[pts_local[i][0] for i in g]
            if abs(np.mean(gxs)-xv) <= max(10, w//40):
                g.append(idx); placed=True; break
        if not placed:
            col_groups.append([idx])
    columns=[]
    for g in col_groups:
        col_pts = sorted([pts_local[i] for i in g], key=lambda t: t[1])
        seq=[p[2] for p in col_pts]
        columns.append(seq)
    flattened=[]
    maxlen = max((len(c) for c in columns), default=0)
    for r in range(maxlen):
        for col in columns:
            if r < len(col):
                flattened.append(col[r])
    runs=[]
    if flattened:
        cur={'color':flattened[0],'len':1}
        for k in range(1,len(flattened)):
            if flattened[k]==cur['color']:
                cur['len']+=1
            else:
                runs.append(cur)
                cur={'color':flattened[k],'len':1}
        runs.append(cur)
    maxRun = max((r['len'] for r in runs), default=0)
    cat='other'
    if maxRun >= 10: cat='super_long'
    elif maxRun >=8: cat='long'
    elif maxRun >=4: cat='longish'
    elif maxRun ==1: cat='single'
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"columns":columns,"runs":runs}

# ---------- Classification ----------
def classify_overall(board_infos):
    longCount = sum(1 for b in board_infos if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_infos if b['category']=='super_long')
    longishCount = sum(1 for b in board_infos if b['category']=='longish')
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount

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

    boards_with_multicol = sum(1 for b in board_infos if board_has_3consec_multicolumn(b['columns']))
    boards_with_long = sum(1 for b in board_infos if b['maxRun'] >= 8)

    if boards_with_multicol >= 3 and boards_with_long >= 2:
        return "中等胜率（中上）", boards_with_long, sum(1 for b in board_infos if b['category']=='super_long')

    totals = [b['total'] for b in board_infos]
    if board_infos and sum(1 for t in totals if t < 6) >= len(board_infos)*0.6:
        return "胜率调低 / 收割时段", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')

    return "胜率中等（平台收割中等时段）", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')

# ---------- Playwright helpers (robust attempts) ----------
# Add anti-headless shims
def apply_stealth(page):
    # Make navigator.webdriver false, set languages, plugin, chrome runtime, etc.
    page.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => false});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
    Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4]});
    window.chrome = { runtime: {} };
    """)
    # override permissions query
    page.add_init_script("""window.__phantomOverride = true;""")
    return

def human_like_drag(page, start_x, start_y, end_x, end_y, steps=30):
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    for i in range(1, steps+1):
        nx = start_x + (end_x - start_x) * (i/steps) + random.uniform(-2,2)
        ny = start_y + (end_y - start_y) * (i/steps) + random.uniform(-1,1)
        page.mouse.move(nx, ny, steps=1)
        time.sleep(random.uniform(0.01,0.04))
    page.mouse.up()

def try_solve_slider(page):
    # multiple strategies, return True if looks successful
    try:
        # Strategy A: find input[type=range] or role=slider
        selectors = ["input[type=range]","div[role=slider]","div[class*=slider]","div[class*=captcha]","div[class*=slide]"]
        for sel in selectors:
            try:
                els = page.query_selector_all(sel)
                if els and len(els)>0:
                    elem = els[0]
                    box = elem.bounding_box()
                    if box:
                        x0 = box['x']+2; y0 = box['y'] + box['height']/2
                        x1 = box['x'] + box['width'] - 6
                        human_like_drag(page, x0, y0, x1, y0, steps=30)
                        time.sleep(1.0)
                        return True
            except Exception:
                continue
        # Strategy B: try to find an obvious button/text and drag a nearby element
        # Find any element with text '滑' / 'slide' / 'drag' or containing '安全' etc
        possible = page.query_selector_all("text=/滑|slide|drag|安全|security/i")
        for p in possible:
            try:
                box = p.bounding_box()
                if box:
                    sx = box['x']; sy = box['y']; w = box['width']; h = box['height']
                    # drag a bit to the right
                    start_x = sx + max(5, w*0.1); start_y = sy + h/2
                    end_x = sx + w + 40
                    human_like_drag(page, start_x, start_y, end_x, start_y, steps=28)
                    time.sleep(1.0)
                    return True
            except Exception:
                continue
        # Strategy C: image-detect slider and drag approximated position (screenshot)
        try:
            ss = page.screenshot(full_page=True)
            img = Image.open(BytesIO(ss)).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            H,W = bgr.shape[:2]
            region = bgr[int(H*0.25):int(H*0.75), int(W*0.05):int(W*0.95)]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _,th = cv2.threshold(gray, 200,255,cv2.THRESH_BINARY)
            contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best=None; best_area=0
            for cnt in contours:
                bx,by,bw,bh = cv2.boundingRect(cnt)
                area = bw*bh
                if area > best_area and bw>40 and bw>3*bh:
                    best=(bx,by,bw,bh); best_area=area
            if best:
                bx,by,bw,bh = best
                # translate to page coords
                px = int(W*0.05) + bx; py = int(H*0.25) + by
                start_x = px + 6; start_y = py + bh//2
                end_x = px + bw - 6
                human_like_drag(page, start_x, start_y, end_x, start_y, steps=30)
                time.sleep(1.2)
                return True
        except Exception:
            pass
        # Strategy D: dispatch pointer events via JS on likely element selectors
        for sel in ["input[type=range]","div[role=slider]","div[class*=slider]","div[class*=captcha]"]:
            try:
                handle = page.query_selector(sel)
                if handle:
                    box = handle.bounding_box()
                    if box:
                        sx = box['x']+2; sy=box['y']+box['height']/2
                        ex = box['x']+box['width']-6
                        page.dispatch_event(sel, "pointerdown", {"button":0, "clientX":sx, "clientY":sy})
                        page.dispatch_event(sel, "pointermove", {"clientX":ex, "clientY":sy})
                        page.dispatch_event(sel, "pointerup", {"button":0, "clientX":ex, "clientY":sy})
                        time.sleep(1.0)
                        return True
            except Exception:
                continue
    except Exception as e:
        log(f"try_solve_slider exception: {e}")
    return False

# ---------- Capture DG page (robust) ----------
def capture_dg_page(max_total_attempts=3):
    with sync_playwright() as p:
        # try different user agents and viewports to bypass blocking
        user_agents = [
            # mobile UA
            "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            # desktop UA
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        viewports = [(390,844),(1280,900)]
        attempt=0
        for attempt in range(max_total_attempts):
            ua = random.choice(user_agents)
            vw,vh = random.choice(viewports)
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(user_agent=ua, viewport={"width":vw,"height":vh}, locale="en-US")
            page = context.new_page()
            apply_stealth(page)
            # random short wait to mimic human arrival
            time.sleep(random.uniform(0.3,1.0))
            for url in DG_LINKS:
                try:
                    log(f"Opening {url} (attempt {attempt+1}, ua len {len(ua)}, viewport {vw}x{vh})")
                    page.goto(url, timeout=35000)
                    time.sleep(1.0 + random.uniform(0,1.0))
                    # click Free-ish
                    clicked=False
                    for txt in ["Free","免费试玩","免费","Play Free","试玩","Free Play","免费体验"]:
                        try:
                            loc = page.locator(f"text={txt}")
                            if loc.count()>0:
                                loc.first.click(timeout=4000)
                                clicked=True
                                log(f"Clicked text '{txt}'")
                                break
                        except Exception:
                            continue
                    if not clicked:
                        # fallback: iterate buttons/links and click those containing free/试玩/免费
                        try:
                            els = page.query_selector_all("a,button")
                            for i in range(min(80,len(els))):
                                try:
                                    t = els[i].inner_text().strip().lower()
                                    if "free" in t or "试玩" in t or "免费" in t:
                                        els[i].click(timeout=3000)
                                        clicked=True
                                        log("Clicked candidate a/button by scanning text.")
                                        break
                                except Exception:
                                    continue
                        except Exception:
                            pass
                    time.sleep(0.8 + random.uniform(0,0.8))
                    # Try slider multiple times
                    slider_ok=False
                    for s_try in range(6):
                        log(f"slider attempt {s_try+1}")
                        got = try_solve_slider(page)
                        if got:
                            slider_ok=True
                            log("Slider attempt reported success.")
                            break
                        else:
                            # try small page scrolls and retry
                            try:
                                page.mouse.wheel(0, 300)
                                time.sleep(0.5)
                            except Exception:
                                pass
                    # Now wait for potential real-play appearance
                    for check in range(8):
                        ss = page.screenshot(full_page=True)
                        # save latest screenshot for debug
                        try:
                            with open(LAST_SCREENSHOT, "wb") as f: f.write(ss)
                        except:
                            pass
                        img = pil_from_bytes(ss)
                        bgr = cv_from_pil(img)
                        pts,_,_ = detect_color_points(bgr)
                        log(f"Check {check+1}: detected points {len(pts)}")
                        if len(pts) >= MIN_POINTS_FOR_REAL:
                            log("Seems to be in real-play page (enough points).")
                            context.close(); browser.close()
                            return ss
                        time.sleep(1.2 + random.uniform(0,0.8))
                    # not reached -> try next url
                except PWTimeout as e:
                    log(f"Timeout opening {url}: {e}")
                except Exception as e:
                    log(f"Error interacting with {url}: {e}")
                finally:
                    # continue to next url
                    pass
            try:
                context.close()
            except:
                pass
            try:
                browser.close()
            except:
                pass
            # slight random backoff before next attempt
            time.sleep(2 + random.uniform(0,2))
        log("All attempts exhausted; failed to enter DG real-play page.")
        return None

# ---------- Historical prediction helper ----------
def predict_from_history(state):
    hist = state.get("history", []) or []
    if not hist:
        log("No history.")
        return None
    now = now_tz()
    cutoff = now - timedelta(days=28)
    recent=[]
    for e in hist:
        try:
            st = datetime.fromisoformat(e["start_time"])
            st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= cutoff:
                recent.append({"kind":e.get("kind"), "start":st, "duration": e.get("duration_minutes",0)})
        except Exception:
            continue
    if not recent:
        log("No recent events in last 28 days.")
        return None
    # build buckets keyed by (kind, weekday, hour, bucket_min)
    buckets={}
    for ev in recent:
        weekday = ev["start"].weekday()
        hour = ev["start"].hour
        minute_bucket = (ev["start"].minute // PRED_BUCKET_MINUTES) * PRED_BUCKET_MINUTES
        key=(ev["kind"], weekday, hour, minute_bucket)
        if key not in buckets: buckets[key]={"count":0,"durations":[]}
        buckets[key]["count"]+=1
        buckets[key]["durations"].append(ev["duration"])
    candidates=[]
    for k,v in buckets.items():
        if v["count"] >= MIN_HISTORY_EVENTS_FOR_PRED:
            avg_dur = round(sum(v["durations"])/len(v["durations"])) if v["durations"] else 10
            candidates.append({"key":k,"count":v["count"],"avg_duration":avg_dur})
    if not candidates:
        log("No bucket passes threshold.")
        return None
    candidates.sort(key=lambda x:x["count"], reverse=True)
    best = candidates[0]
    kind, weekday, hour, bmin = best["key"]
    # next occurrence
    now = now_tz()
    days_ahead = (weekday - now.weekday()) % 7
    predicted_start = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=bmin, second=0, microsecond=0)
    if predicted_start < now - timedelta(minutes=1):
        predicted_start += timedelta(days=7)
    predicted_end = predicted_start + timedelta(minutes=best["avg_duration"])
    return {"kind":kind, "predicted_start":predicted_start, "predicted_end":predicted_end, "avg_duration":best["avg_duration"], "count":best["count"]}

# ---------- Try to fetch historical from possible API endpoints (speculative) ----------
def try_fetch_public_history():
    # Some platforms expose history endpoints; try common patterns (speculative)
    tried=[]
    for base in DG_LINKS:
        # example endpoints (speculative)
        for path in ["/api/history", "/history", "/api/v1/history", "/game/history"]:
            url = base.rstrip("/") + path
            tried.append(url)
            try:
                r = requests.get(url, timeout=8)
                if r.status_code==200:
                    try:
                        j = r.json()
                        log(f"Found JSON historical endpoint: {url}")
                        return j
                    except Exception:
                        continue
            except Exception:
                continue
    log(f"Tried speculative history endpoints: {tried}")
    return None

# ---------- Main ----------
def main():
    log("=== Run start ===")
    state = load_state()
    screenshot = None
    try:
        screenshot = capture_dg_page()
    except Exception as e:
        log(f"Exception capture: {e}\n{traceback.format_exc()}")

    if screenshot:
        # Real-time path
        try:
            with open(LAST_SCREENSHOT,"wb") as f: f.write(screenshot)
        except:
            pass
        img = pil_from_bytes(screenshot)
        bgr = cv_from_pil(img)
        pts,_,_ = detect_color_points(bgr)
        log(f"Detected points: {len(pts)}")
        if len(pts) < MIN_POINTS_FOR_REAL:
            log("Points less than threshold -> treat as failed to enter real-play.")
            # fallback to history
            fallback_history_mode(state)
            return
        rects = cluster_points_to_boards(pts, bgr.shape)
        log(f"Clustered boards: {len(rects)}")
        boards=[]
        for r in rects:
            info = analyze_board(bgr, r)
            boards.append(info)
        overall, longCount, superCount = classify_overall(boards)
        log(f"Classification: {overall} (long/超:{longCount}/{superCount})")
        now = now_tz().isoformat()
        was_active = state.get("active", False)
        is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")

        if is_active_now and not was_active:
            state = {"active":True,"kind":overall,"start_time":now,"last_seen":now,"history":state.get("history",[])}
            save_state(state)
            hist = state.get("history", []) or []
            durations = [h.get("duration_minutes",0) for h in hist if h.get("duration_minutes",0)>0]
            est_min = round(sum(durations)/len(durations)) if durations else 10
            est_end = (now_tz() + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
            msg = f"🔔 [DG提醒] {overall} 開始\n時間: {now}\n长龙/超龙 桌數: {longCount} (超:{superCount})\n估計結束: {est_end}（約 {est_min} 分鐘，基於歷史）"
            send_telegram(msg)
            save_state(state)
        elif is_active_now and was_active:
            state["last_seen"] = now
            state["kind"] = overall
            save_state(state)
            log("Event still active -> updated last_seen.")
        elif not is_active_now and was_active:
            start_time = datetime.fromisoformat(state.get("start_time"))
            end_time = now_tz()
            duration_minutes = round((end_time - start_time).total_seconds() / 60.0)
            history = state.get("history", [])
            history.append({"kind":state.get("kind"), "start_time": state.get("start_time"), "end_time": end_time.isoformat(), "duration_minutes": duration_minutes})
            history = history[-1000:]
            new_state = {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":history}
            save_state(new_state)
            msg = f"✅ [DG提醒] {state.get('kind')} 已結束\n開始: {state.get('start_time')}\n結束: {end_time.isoformat()}\n實際持續: {duration_minutes} 分鐘"
            send_telegram(msg)
            log("End notification sent and history saved.")
        else:
            save_state(state)
            log("Not active; nothing to send.")

        # save summary for debugging
        summary = {"ts": now_tz().isoformat(), "overall": overall, "longCount":longCount, "superCount":superCount, "boards": boards[:60]}
        try:
            with open(SUMMARY_FILE,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
        except Exception:
            pass
        return
    else:
        # couldn't enter real-play -> fallback
        fallback_history_mode(state)
        return

def fallback_history_mode(state):
    log("Real-play capture failed -> enter fallback/historical mode.")
    # 1) try to fetch public history endpoints (speculative)
    public_hist = try_fetch_public_history()
    if public_hist:
        # if we get JSON structure, try parse events into state.history if possible
        # The structure is unknown; attempt to find list of events with start/end/duration
        events = []
        if isinstance(public_hist, list):
            events = public_hist
        elif isinstance(public_hist, dict):
            # try common keys
            for key in ("events","history","records"):
                if key in public_hist and isinstance(public_hist[key], list):
                    events = public_hist[key]; break
        # attempt to normalize
        normalized=[]
        for ev in events:
            try:
                st = ev.get("start_time") or ev.get("start") or ev.get("ts")
                end = ev.get("end_time") or ev.get("end")
                duration = ev.get("duration_minutes") or ev.get("duration")
                if st:
                    normalized.append({"kind": ev.get("kind","放水"), "start_time": st, "end_time": end, "duration_minutes": duration or 0})
            except:
                continue
        if normalized:
            # merge into state history
            hist = state.get("history", []) or []
            hist.extend(normalized)
            state["history"] = hist[-1000:]
            save_state(state)
            log("Imported public history into state.")
    # 2) If enough history in state, generate prediction and possibly send reminder
    hist = state.get("history", []) or []
    # ensure datetime conversion
    recent_count = 0
    for h in hist:
        try:
            st = datetime.fromisoformat(h["start_time"])
            st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= now_tz() - timedelta(days=28):
                recent_count += 1
        except:
            continue
    if recent_count < MIN_HISTORY_EVENTS_FOR_PRED:
        log(f"History insufficient ({recent_count} events in last 28 days). No fallback send. Will collect history for future.")
        save_state(state)
        return
    pred = predict_from_history(state)
    if not pred:
        log("No clear historical pattern found.")
        save_state(state)
        return
    ps = pred["predicted_start"]; pe = pred["predicted_end"]
    now = now_tz()
    lead = timedelta(minutes=PRED_LEAD_MINUTES)
    if (ps - lead) <= now <= pe:
        remaining = max(0, int((pe - now).total_seconds() // 60))
        msg = f"🔔 [DG替補預測] 根據過去 4 週歷史模式偵測到可能的『{pred['kind']}』\n預測開始: {ps.strftime('%Y-%m-%d %H:%M:%S')}\n估計結束: {pe.strftime('%Y-%m-%d %H:%M:%S')}（約 {pred['avg_duration']} 分鐘）\n目前剩餘: 約 {remaining} 分鐘\n※ 此為替補（基於歷史），因未能直接進入實盤"
        send_telegram(msg)
        log("Historical fallback telegram sent.")
    else:
        log(f"Predicted next {pred['kind']} at {ps.strftime('%Y-%m-%d %H:%M:%S')} (not within lead window).")
    save_state(state)
    return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Main exception: {e}\n{traceback.format_exc()}")
        sys.exit(1)
