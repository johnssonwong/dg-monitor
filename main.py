# main.py — DG 实盘检测 + 历史替补（Wayback / DG API / 导入）+ Telegram 通知
# 说明（简短）：
#  - 首先尝试真实进入 DG 平台（点击 Free -> 滑动安全条 -> 截图并解析）
#  - 若进入失败，自动尝试从以下历史来源获取“全市场”历史：
#       1) DG 可能的公开历史 API（若存在）
#       2) Wayback Machine 最近 28 天对 DG 页面之快照（抓取并渲染快照以截图解析）
#       3) (可选) 导入本地/外部历史 CSV/JSON（用户如有可放到 repo 并命名）
#       4) 若以上都失败，则开始从现在持续收集历史，供未来预测
#  - 若历史检测判定出“放水”或“中等胜率（中上）”模式，立即发送 Telegram（并标注是否为“替补/历史”）
#
# 部署：在 GitHub Actions 中运行（每5分钟触发）。将 TG_BOT_TOKEN 与 TG_CHAT_ID 存为 repo secrets。
# 注意事项请看脚本顶部注释与运行日志。

import os, sys, time, json, math, random, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import cv2
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# -------------------- CONFIG --------------------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]  # target
TG_TOKEN_ENV = "TG_BOT_TOKEN"
TG_CHAT_ENV = "TG_CHAT_ID"

MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW","3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))
STATE_FILE = "state.json"
SUMMARY_FILE = "last_summary.json"
LAST_SCREENSHOT = "last_screenshot.png"

TZ = timezone(timedelta(hours=8))  # Malaysia UTC+8

# history / wayback params
HISTORY_LOOKBACK_DAYS = 28
MIN_HISTORY_EVENTS_FOR_PRED = 3
PRED_BUCKET_MINUTES = 15
PRED_LEAD_MINUTES = 10
WAYBACK_MAX_SNAPSHOTS = 40  # max snapshots to try (per URL) in last 28 days
WAYBACK_RATE_SLEEP = 1.2     # seconds between requests to archive

# detection tuning
MIN_POINTS_FOR_REAL = 40
DILATE_KERNEL_SIZE = 40
# -------------------------------------------------

def now_tz(): return datetime.now(TZ)
def nowstr(): return now_tz().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{nowstr()}] {msg}", flush=True)

# ---------- Telegram ----------
def send_telegram(text):
    token = os.environ.get(TG_TOKEN_ENV,"").strip()
    chat = os.environ.get(TG_CHAT_ENV,"").strip()
    if not token or not chat:
        log("Telegram credentials not set (TG_BOT_TOKEN/TG_CHAT_ID).")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id":chat,"text":text}, timeout=20)
        res = r.json()
        if res.get("ok"):
            log("Telegram sent OK.")
            return True
        else:
            log(f"Telegram error: {res}")
            return False
    except Exception as e:
        log(f"Telegram send except: {e}")
        return False

# ---------- state ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}
    try:
        with open(STATE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(s,f,ensure_ascii=False,indent=2)

# ---------- image helpers ----------
def pil_from_bytes(b): return Image.open(BytesIO(b)).convert("RGB")
def cv_from_pil(pil): return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_color_points(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1,upper1 = np.array([0,90,60]), np.array([10,255,255])
    lower2,upper2 = np.array([160,90,60]), np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    lowerb,upperb = np.array([85,60,40]), np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)
    points=[]
    for mask,label in [(mask_r,'B'),(mask_b,'P')]:
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            M = cv2.moments(cnt)
            if M.get("m00",0)==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,label))
    return points, mask_r, mask_b

def cluster_points_to_boards(points, img_shape):
    h,w = img_shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)
    for x,y,_ in points:
        if 0<=y<h and 0<=x<w: mask[y,x]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE,DILATE_KERNEL_SIZE))
    big = cv2.dilate(mask, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(big, connectivity=8)
    rects=[]
    for i in range(1,num):
        x,y,w_,h_ = stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP], stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]
        if w_<60 or h_<40: continue
        pad=8
        x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w-1,x+w_+pad); y1=min(h-1,y+h_+pad)
        rects.append((x0,y0,x1-x0,y1-y0))
    if not rects:
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
        xv = xs[idx]; placed=False
        for g in col_groups:
            gxs=[pts_local[i][0] for i in g]
            if abs(np.mean(gxs)-xv) <= max(10, w//40):
                g.append(idx); placed=True; break
        if not placed:
            col_groups.append([idx])
    columns=[]
    for g in col_groups:
        col_pts = sorted([pts_local[i] for i in g], key=lambda t:t[1])
        columns.append([p[2] for p in col_pts])
    flattened=[]
    maxlen = max((len(c) for c in columns), default=0)
    for r in range(maxlen):
        for col in columns:
            if r < len(col): flattened.append(col[r])
    runs=[]
    if flattened:
        cur={"color":flattened[0],"len":1}
        for k in range(1,len(flattened)):
            if flattened[k]==cur["color"]: cur["len"]+=1
            else:
                runs.append(cur); cur={"color":flattened[k],"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    cat="other"
    if maxRun>=10: cat="super_long"
    elif maxRun>=8: cat="long"
    elif maxRun>=4: cat="longish"
    elif maxRun==1: cat="single"
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"columns":columns,"runs":runs}

# ---------- Classification ----------
def classify_overall(board_infos):
    longCount = sum(1 for b in board_infos if b["category"] in ("long","super_long"))
    superCount = sum(1 for b in board_infos if b["category"]=="super_long")
    def board_has_3consec_multicolumn(columns):
        col_runlens=[]
        for col in columns:
            if not col: col_runlens.append(0); continue
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

# ---------- Playwright helpers ----------
def apply_stealth(page):
    page.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => false});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
    Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4]});
    window.chrome = { runtime: {} };
    """)
def human_like_drag(page, x0,y0,x1,y1,steps=30):
    page.mouse.move(x0,y0); page.mouse.down()
    for i in range(1,steps+1):
        nx = x0 + (x1-x0)*(i/steps) + random.uniform(-2,2)
        ny = y0 + (y1-y0)*(i/steps) + random.uniform(-1,1)
        page.mouse.move(nx,ny,steps=1)
        time.sleep(random.uniform(0.01,0.04))
    page.mouse.up()

def try_solve_slider(page):
    try:
        selectors = ["input[type=range]","div[role=slider]","div[class*=slider]","div[class*=captcha]","div[class*=slide]"]
        for sel in selectors:
            els = page.query_selector_all(sel)
            if els and len(els)>0:
                elem = els[0]; box = elem.bounding_box()
                if box:
                    sx = box["x"]+2; sy = box["y"]+box["height"]/2; ex = box["x"]+box["width"]-6
                    human_like_drag(page,sx,sy,ex,sy,steps=30); time.sleep(1.0); return True
        # image-based fallback
        ss = page.screenshot(full_page=True)
        img = pil_from_bytes(ss); bgr = cv_from_pil(img); H,W=bgr.shape[:2]
        region = bgr[int(H*0.25):int(H*0.75), int(W*0.05):int(W*0.95)]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _,th = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best=None;best_area=0
        for cnt in cnts:
            bx,by,bw,bh = cv2.boundingRect(cnt); area=bw*bh
            if area>best_area and bw>40 and bw>3*bh: best=(bx,by,bw,bh);best_area=area
        if best:
            bx,by,bw,bh = best
            px = int(W*0.05)+bx; py = int(H*0.25)+by
            sx = px+6; sy = py+bh//2; ex = px+bw-6
            human_like_drag(page,sx,sy,ex,sy,steps=30); time.sleep(1.0); return True
    except Exception as e:
        log(f"try_solve_slider except: {e}")
    return False

def capture_dg_page(attempts=3):
    with sync_playwright() as p:
        user_agents = [
            "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        viewports = [(390,844),(1280,900)]
        for attempt in range(attempts):
            ua = random.choice(user_agents); vw,vh = random.choice(viewports)
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(user_agent=ua, viewport={"width":vw,"height":vh}, locale="en-US")
            page = context.new_page(); apply_stealth(page)
            time.sleep(random.uniform(0.3,1.0))
            for url in DG_LINKS:
                try:
                    log(f"Opening {url} attempt {attempt+1}")
                    page.goto(url, timeout=35000); time.sleep(1.0+random.random())
                    clicked=False
                    for txt in ["Free","免费试玩","免费","Play Free","试玩","Free Play","免费体验"]:
                        try:
                            loc = page.locator(f"text={txt}")
                            if loc.count()>0:
                                loc.first.click(timeout=4000); clicked=True; log(f"Clicked '{txt}'"); break
                        except: continue
                    if not clicked:
                        try:
                            els = page.query_selector_all("a,button")
                            for i in range(min(80,len(els))):
                                try:
                                    t = els[i].inner_text().strip().lower()
                                    if "free" in t or "试玩" in t or "免费" in t:
                                        els[i].click(timeout=3000); clicked=True; log("Clicked candidate button"); break
                                except: continue
                        except: pass
                    time.sleep(0.6+random.random())
                    # try slider repeatedly
                    slider_ok = False
                    for s in range(6):
                        got = try_solve_slider(page)
                        log(f"slider try {s+1}, got={got}")
                        if got:
                            slider_ok=True; break
                        else:
                            try:
                                page.mouse.wheel(0,300); time.sleep(0.6)
                            except: pass
                    # after slider attempts, check for many points
                    for check in range(8):
                        ss = page.screenshot(full_page=True)
                        try:
                            with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
                        except: pass
                        img = pil_from_bytes(ss); bgr = cv_from_pil(img)
                        pts,_,_ = detect_color_points(bgr)
                        log(f"check {check+1}: detected points {len(pts)}")
                        if len(pts) >= MIN_POINTS_FOR_REAL:
                            context.close(); browser.close()
                            log("Entered real-play (enough points).")
                            return ss
                        time.sleep(1.0+random.random())
                except PWTimeout as e:
                    log(f"Timeout on {url}: {e}")
                except Exception as e:
                    log(f"Error on {url}: {e}")
            try:
                context.close()
            except: pass
            try:
                browser.close()
            except: pass
            time.sleep(2+random.random())
        log("Failed to enter real-play after attempts.")
        return None

# ---------- Wayback helpers ----------
def get_wayback_snapshots(url, from_date=None, to_date=None, limit=40):
    # CDX API: http://web.archive.org/cdx/search/cdx?url=example.com&from=YYYYMMDD&to=YYYYMMDD&output=json
    base = "http://web.archive.org/cdx/search/cdx"
    params = {"url":url, "output":"json", "filter":"statuscode:200", "limit":str(limit)}
    if from_date: params["from"]=from_date
    if to_date: params["to"]=to_date
    try:
        r = requests.get(base, params=params, timeout=12)
        if r.status_code != 200:
            log(f"Wayback CDX returned {r.status_code} for {url}")
            return []
        j = r.json()
        # first row is header
        rows = j[1:] if len(j)>1 else []
        # each row: [urlkey, timestamp, original, mime, status, digest, length]
        timestamps = [row[1] for row in rows if len(row)>1]
        return timestamps
    except Exception as e:
        log(f"Wayback CDX error: {e}")
        return []

def fetch_wayback_snapshot_and_screenshot(snapshot_ts, target_url):
    # Build snapshot URL: https://web.archive.org/web/{ts}/{url}
    snap = f"https://web.archive.org/web/{snapshot_ts}/{target_url}"
    try:
        # Render snapshot with Playwright to ensure dynamic content (if available) is executed
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(viewport={"width":1280,"height":900})
            page = context.new_page()
            page.goto(snap, timeout=30000)
            time.sleep(2.0)
            ss = page.screenshot(full_page=True)
            context.close(); browser.close()
            return ss
    except Exception as e:
        log(f"Fetch/render wayback snapshot {snapshot_ts} failed: {e}")
        return None

# ---------- Historical ingestion (external CSV/JSON support) ----------
def load_history_from_file(filename):
    if not os.path.exists(filename): return []
    try:
        with open(filename,"r",encoding="utf-8") as f:
            j = json.load(f)
            # expect list of {"kind","start_time","end_time","duration_minutes"}
            if isinstance(j,list): return j
    except Exception:
        pass
    return []

# ---------- Predict from historical state ----------
def predict_from_history(state):
    hist = state.get("history",[]) or []
    now = now_tz()
    cutoff = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
    recent=[]
    for ev in hist:
        try:
            st = datetime.fromisoformat(ev["start_time"])
            st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= cutoff: recent.append({"kind":ev.get("kind"),"start":st,"duration":ev.get("duration_minutes",0)})
        except:
            continue
    if len(recent) < MIN_HISTORY_EVENTS_FOR_PRED:
        log("Not enough recent history for prediction.")
        return None
    buckets={}
    for ev in recent:
        weekday = ev["start"].weekday(); hour = ev["start"].hour
        minute_bucket = (ev["start"].minute // PRED_BUCKET_MINUTES) * PRED_BUCKET_MINUTES
        key = (ev["kind"], weekday, hour, minute_bucket)
        if key not in buckets: buckets[key]={"count":0,"durations":[]}
        buckets[key]["count"]+=1; buckets[key]["durations"].append(ev["duration"])
    candidates=[]
    for k,v in buckets.items():
        if v["count"]>=MIN_HISTORY_EVENTS_FOR_PRED:
            avg_dur = round(sum(v["durations"])/len(v["durations"])) if v["durations"] else 10
            candidates.append({"key":k,"count":v["count"],"avg_duration":avg_dur})
    if not candidates: return None
    candidates.sort(key=lambda x:x["count"], reverse=True)
    best = candidates[0]
    kind, weekday, hour, bmin = best["key"]
    now = now_tz()
    days_ahead = (weekday - now.weekday()) % 7
    predicted_start = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=bmin, second=0, microsecond=0)
    if predicted_start < now - timedelta(minutes=1): predicted_start += timedelta(days=7)
    predicted_end = predicted_start + timedelta(minutes=best["avg_duration"])
    return {"kind":kind, "predicted_start":predicted_start, "predicted_end":predicted_end, "avg_duration":best["avg_duration"], "count":best["count"]}

# ---------- Try to fetch DG public history (speculative) ----------
def try_fetch_public_history():
    tried=[]
    for base in DG_LINKS:
        for path in ["/api/history","/history","/api/v1/history","/game/history","/api/records"]:
            url = base.rstrip("/") + path
            tried.append(url)
            try:
                r = requests.get(url, timeout=8)
                if r.status_code==200:
                    try:
                        return r.json()
                    except:
                        continue
            except:
                continue
    log(f"Speculative history endpoints tried: {tried}")
    return None

# ---------- Main run ----------
def main():
    log("=== Run started ===")
    state = load_state()
    # 1) Try real-time capture
    screenshot = None
    try:
        screenshot = capture_dg_page()
    except Exception as e:
        log(f"capture_dg_page exception: {e}\n{traceback.format_exc()}")
    # If screenshot obtained, run real-time detection
    if screenshot:
        try:
            with open(LAST_SCREENSHOT,"wb") as f: f.write(screenshot)
        except: pass
        img = pil_from_bytes(screenshot); bgr = cv_from_pil(img)
        pts,_,_ = detect_color_points(bgr)
        log(f"Realtime points detected: {len(pts)}")
        if len(pts) < MIN_POINTS_FOR_REAL:
            log("Realtime points below threshold -> fallback to history.")
            fallback_with_history(state)
            return
        rects = cluster_points_to_boards(pts, bgr.shape)
        log(f"Realtime clustered boards: {len(rects)}")
        boards=[]
        for r in rects:
            boards.append(analyze_board(bgr, r))
        overall, longCount, superCount = classify_overall(boards)
        log(f"Realtime classification: {overall} (long/超: {longCount}/{superCount})")
        nowiso = now_tz().isoformat()
        was_active = state.get("active",False)
        is_active_now = overall in ("放水时段（提高胜率）","中等胜率（中上）")
        if is_active_now and not was_active:
            state = {"active":True,"kind":overall,"start_time":nowiso,"last_seen":nowiso,"history":state.get("history",[])}
            save_state(state)
            hist = state.get("history",[]) or []
            durations = [h.get("duration_minutes",0) for h in hist if h.get("duration_minutes",0)>0]
            est_min = round(sum(durations)/len(durations)) if durations else 10
            est_end = (now_tz() + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
            msg = f"🔔 [DG提醒] {overall} 開始\n時間: {nowiso}\n长龙/超龙 桌數: {longCount} (超龙:{superCount})\n估計結束: {est_end}（約 {est_min} 分鐘，基於歷史）"
            send_telegram(msg)
            save_state(state)
        elif is_active_now and was_active:
            state["last_seen"] = nowiso; state["kind"] = overall; save_state(state); log("Event continues.")
        elif not is_active_now and was_active:
            start_time = datetime.fromisoformat(state.get("start_time"))
            end_time = now_tz()
            duration_minutes = round((end_time - start_time).total_seconds()/60.0)
            hist = state.get("history",[]); hist.append({"kind":state.get("kind"), "start_time": state.get("start_time"), "end_time": end_time.isoformat(), "duration_minutes": duration_minutes})
            state = {"active":False,"kind":None,"start_time":None,"last_seen":None,"history": hist[-1000:]}
            save_state(state)
            msg = f"✅ [DG提醒] {state.get('kind')} 已結束\n開始: {state.get('start_time')}\n結束: {end_time.isoformat()}\n實際持續: {duration_minutes} 分鐘"
            send_telegram(msg)
            log("End notification sent.")
        else:
            save_state(state); log("Nothing to send realtime.")
        # save summary
        summary = {"ts": now_tz().isoformat(), "overall": overall, "longCount": longCount, "superCount": superCount, "boards": boards[:60]}
        try:
            with open(SUMMARY_FILE,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
        except: pass
        return
    else:
        # fallback
        fallback_with_history(state)
        return

def fallback_with_history(state):
    log("Fallback: use historical full-market data (attempt).")
    # 1) Try DG public history API
    api_hist = None
    try:
        api_hist = try_fetch_public_history()
    except Exception as e:
        log(f"try_fetch_public_history except: {e}")
    if api_hist:
        # attempt to normalize and analyze
        norm = []
        # best-effort parse - depends on API structure
        if isinstance(api_hist, list):
            # try to find fields
            for rec in api_hist:
                st = rec.get("start_time") or rec.get("ts") or rec.get("time")
                end = rec.get("end_time") or rec.get("end")
                dur = rec.get("duration_minutes") or rec.get("duration")
                if st:
                    norm.append({"kind": rec.get("kind","放水"), "start_time": st, "end_time": end, "duration_minutes": dur or 0})
        elif isinstance(api_hist, dict):
            # find list fields
            for key in ("events","history","records"):
                if key in api_hist and isinstance(api_hist[key], list):
                    for rec in api_hist[key]:
                        st = rec.get("start_time") or rec.get("ts") or rec.get("time")
                        end = rec.get("end_time") or rec.get("end")
                        dur = rec.get("duration_minutes") or rec.get("duration")
                        if st:
                            norm.append({"kind": rec.get("kind","放水"), "start_time": st, "end_time": end, "duration_minutes": dur or 0})
                    break
        if norm:
            hist = state.get("history",[]) or []
            hist.extend(norm); state["history"] = hist[-1000:]; save_state(state)
            log(f"Imported {len(norm)} records from public API into state history.")
    # 2) Try Wayback Machine snapshots for DG links within last 28 days
    # Only if history still insufficient
    hist_count_recent = 0
    for h in state.get("history",[]) or []:
        try:
            st = datetime.fromisoformat(h["start_time"])
            st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS): hist_count_recent += 1
        except:
            continue
    if hist_count_recent < MIN_HISTORY_EVENTS_FOR_PRED:
        log("History insufficient -> try Wayback snapshots.")
        # For each DG link try to gather snapshots within last 28 days
        from_date = (now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime("%Y%m%d")
        to_date = now_tz().strftime("%Y%m%d")
        collected = 0
        for base in DG_LINKS:
            # get timestamps
            try:
                tss = get_wayback_snapshots(base, from_date=from_date, to_date=to_date, limit=WAYBACK_MAX_SNAPSHOTS)
            except Exception as e:
                log(f"Wayback list error: {e}"); tss=[]
            if not tss:
                log(f"No Wayback snapshots for {base} in last {HISTORY_LOOKBACK_DAYS} days.")
                continue
            # iterate snapshots (limit)
            for ts in tss[:WAYBACK_MAX_SNAPSHOTS]:
                time.sleep(WAYBACK_RATE_SLEEP)
                try:
                    ss = fetch_wayback_snapshot_and_screenshot(ts, base)
                    if not ss: continue
                    img = pil_from_bytes(ss); bgr = cv_from_pil(img)
                    pts,_,_ = detect_color_points(bgr)
                    if len(pts) < MIN_POINTS_FOR_REAL:
                        continue
                    rects = cluster_points_to_boards(pts, bgr.shape)
                    boards=[]
                    for r in rects:
                        boards.append(analyze_board(bgr,r))
                    overall, longCount, superCount = classify_overall(boards)
                    # If classification indicates putative event, record as historical event with timestamp=ts
                    if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
                        # convert ts -> datetime
                        try:
                            ev_time = datetime.strptime(ts[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).astimezone(TZ)
                        except:
                            ev_time = now_tz()
                        # create record
                        rec = {"kind": overall, "start_time": ev_time.isoformat(), "end_time": (ev_time + timedelta(minutes=10)).isoformat(), "duration_minutes": 10}
                        state_hist = state.get("history",[]) or []
                        state_hist.append(rec)
                        state["history"] = state_hist[-2000:]
                        collected += 1
                        log(f"Wayback detected event {overall} at snapshot {ts}; collected++.")
                except Exception as e:
                    log(f"Error processing Wayback snapshot {ts} for {base}: {e}")
            save_state(state)
        log(f"Wayback collection done. Collected events: {collected}")
    else:
        log(f"History already sufficient (recent count {hist_count_recent}).")

    # 3) After trying to import, if still insufficient -> notify user that we couldn't access historical full-market data now
    hist_new_count = 0
    for h in state.get("history",[]) or []:
        try:
            st = datetime.fromisoformat(h["start_time"]); st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS): hist_new_count+=1
        except:
            continue
    log(f"Recent history count after attempts: {hist_new_count}")
    if hist_new_count < MIN_HISTORY_EVENTS_FOR_PRED:
        log("仍然无法获得足够历史数据来做替补预测。脚本将继续收集并等待未来数据或需人工提供历史数据。")
        # save and exit
        save_state(state)
        return

    # 4) Now produce prediction and if in window, send Telegram (history-based)
    pred = predict_from_history(state)
    if not pred:
        log("No reliable pattern found in history after collection.")
        save_state(state); return
    ps = pred["predicted_start"]; pe = pred["predicted_end"]
    now = now_tz()
    lead = timedelta(minutes=PRED_LEAD_MINUTES)
    if (ps - lead) <= now <= pe:
        remaining = max(0, int((pe - now).total_seconds()//60))
        msg = f"🔔 [DG替補（歷史）提醒] 根據全市場最近 {HISTORY_LOOKBACK_DAYS} 天的歷史模式偵測到可能的『{pred['kind']}』\n預測開始: {ps.strftime('%Y-%m-%d %H:%M:%S')}\n估計結束: {pe.strftime('%Y-%m-%d %H:%M:%S')}（約 {pred['avg_duration']} 分鐘）\n目前剩餘: 約 {remaining} 分鐘\n※ 此通知基於網路歷史快照與公共資料（替補），非即時實盤抓取。"
        # prevent duplicate sends: record a synthetic event start when sending
        # check last history entries to avoid duplicates
        hist = state.get("history",[]) or []
        last_sent_similar = False
        for h in hist[-30:]:
            try:
                if h.get("kind")==pred["kind"] and abs((datetime.fromisoformat(h["start_time"]).astimezone(TZ) - ps).total_seconds()) < 60*5:
                    last_sent_similar=True; break
            except:
                continue
        if not last_sent_similar:
            send_telegram(msg)
            # append synthetic recorded event (mark source)
            hist.append({"kind":pred["kind"], "start_time": ps.isoformat(), "end_time": pe.isoformat(), "duration_minutes": pred["avg_duration"], "source":"wayback_predict"})
            state["history"] = hist[-2000:]
            save_state(state)
            log("Historical fallback notification sent and recorded.")
        else:
            log("A similar historical event already recorded recently -> skip duplicate notification.")
    else:
        log(f"Predicted {pred['kind']} next at {ps.strftime('%Y-%m-%d %H:%M:%S')} - not in lead window; do not send.")

    save_state(state)
    return

# ---------- Entry ----------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        sys.exit(1)
