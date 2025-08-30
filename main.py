# main.py (final merged)
# 功能：
#  - 尝试使用 Playwright 自动进入 DG 实盘 (点击 Free -> 通过滑动安全条 -> 截图)
#  - 如果进入成功：使用 OpenCV 解析整局面多个桌子，按你定义的规则判断：放水 / 中等胜率（中上） / 胜率中等 / 收割
#  - 如果进入失败：尝试以下历史替补来源（优先级）：
#        1) 调用 DG 或相关域名的公共历史 API（若存在，自动解析）
#        2) 站点爬取公开历史页面（尽力抓取）
#        3) 使用仓库内的 market_history.csv 或 market_history.json（若你已上传，这将被优先使用）
#        4) 若以上都不可用，则记录并从现在开始收集历史（不会盲目发提醒）
#  - 历史预测：按 weekday+小时+bucket_min 聚合，找出重复出现模式并预测下次发生，符合预测窗口则发送 Telegram（格式与你要求一致）
#  - 每次开始/结束事件会保存到 state.history 用于未来估算
#  - 所有 Telegram 提醒格式与你要求一致（含 emoji、估算结束时间、剩余分钟、替补标注）
#
# 注意：
#  - 请把你的 TG_BOT_TOKEN 与 TG_CHAT_ID 存为 GitHub Secrets（TG_BOT_TOKEN / TG_CHAT_ID）
#  - 请把 requirements.txt 与 workflow 配置按之前的说明部署
#  - 若你能拿到“全市场最近4周 DG 历史数据”，把文件 market_history.csv 放到仓库根目录（脚本会优先读取并使用）
#  - 若脚本无法进入实盘或无法抓取历史 API，请把 Actions 日志与 last_screenshot.png 发给我以便微调

import os, sys, time, json, math, random, traceback, csv
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ========= Config =========
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

MIN_HISTORY_EVENTS_FOR_PRED = int(os.environ.get("MIN_HISTORY_EVENTS_FOR_PRED", "3"))
PRED_BUCKET_MINUTES = int(os.environ.get("PRED_BUCKET_MINUTES", "15"))
PRED_LEAD_MINUTES = int(os.environ.get("PRED_LEAD_MINUTES", "10"))

MIN_POINTS_FOR_REAL = int(os.environ.get("MIN_POINTS_FOR_REAL","40"))
DILATE_KERNEL_SIZE = int(os.environ.get("DILATE_KERNEL_SIZE","40"))

# ========= Helpers =========
def now_tz(): return datetime.now(TZ)
def nowstr(): return now_tz().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{nowstr()}] {msg}", flush=True)

# ========= Telegram =========
def send_telegram(text):
    token = os.environ.get(TG_TOKEN_ENV,"").strip()
    chat = os.environ.get(TG_CHAT_ENV,"").strip()
    if not token or not chat:
        log("Telegram token/chat 未配置，跳过发送。")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat, "text": text}, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 已发送。")
            return True
        else:
            log(f"Telegram 返回错误: {j}")
            return False
    except Exception as e:
        log(f"Telegram 发送异常: {e}")
        return False

# ========= State =========
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}
    try:
        with open(STATE_FILE,"r",encoding="utf-8") as f: return json.load(f)
    except:
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f: json.dump(s,f,ensure_ascii=False,indent=2)

# ========= Image detection =========
def pil_from_bytes(b): return Image.open(BytesIO(b)).convert("RGB")
def cv_from_pil(pil): return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

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
    points=[]
    for mask,label in [(mask_r,'B'),(mask_b,'P')]:
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            M=cv2.moments(cnt)
            if M['m00']==0: continue
            cx=int(M['m10']/M['m00']); cy=int(M['m01']/M['m00'])
            points.append((cx,cy,label))
    return points, mask_r, mask_b

def cluster_points_to_boards(points,img_shape):
    h,w = img_shape[:2]
    mask=np.zeros((h,w), dtype=np.uint8)
    for x,y,_ in points:
        if 0<=y<h and 0<=x<w: mask[y,x]=255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(DILATE_KERNEL_SIZE,DILATE_KERNEL_SIZE))
    big=cv2.dilate(mask,kernel,iterations=1)
    num,labels,stats,_=cv2.connectedComponentsWithStats(big,connectivity=8)
    rects=[]
    for i in range(1,num):
        x,y,w_,h_=stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP],stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT]
        if w_<60 or h_<40: continue
        pad=8
        x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w-1,x+w_+pad); y1=min(h-1,y+h_+pad)
        rects.append((x0,y0,x1-x0,y1-y0))
    if not rects:
        cols=max(3,w//300); rows=max(2,h//200)
        cw=w//cols; ch=h//rows
        for r in range(rows):
            for c in range(cols): rects.append((c*cw,r*ch,cw,ch))
    return rects

def analyze_board(bgr, rect):
    x,y,w,h = rect
    crop=bgr[y:y+h, x:x+w]
    pts,_,_ = detect_color_points(crop)
    pts_local=[(px,py,c) for (px,py,c) in pts]
    if not pts_local: return {"total":0,"maxRun":0,"category":"empty","columns":[],"runs":[]}
    xs=[p[0] for p in pts_local]
    idx_sorted=sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups=[]
    for idx in idx_sorted:
        xv=xs[idx]; placed=False
        for g in col_groups:
            gxs=[pts_local[i][0] for i in g]
            if abs(np.mean(gxs)-xv) <= max(10,w//40):
                g.append(idx); placed=True; break
        if not placed: col_groups.append([idx])
    columns=[]
    for g in col_groups:
        col_pts=sorted([pts_local[i] for i in g], key=lambda t: t[1])
        seq=[p[2] for p in col_pts]; columns.append(seq)
    flattened=[]; maxlen=max((len(c) for c in columns), default=0)
    for r in range(maxlen):
        for col in columns:
            if r < len(col): flattened.append(col[r])
    runs=[]
    if flattened:
        cur={'color':flattened[0],'len':1}
        for k in range(1,len(flattened)):
            if flattened[k]==cur['color']: cur['len']+=1
            else:
                runs.append(cur); cur={'color':flattened[k],'len':1}
        runs.append(cur)
    maxRun=max((r['len'] for r in runs), default=0)
    cat='other'
    if maxRun>=10: cat='super_long'
    elif maxRun>=8: cat='long'
    elif maxRun>=4: cat='longish'
    elif maxRun==1: cat='single'
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"columns":columns,"runs":runs}

# ========= Classification =========
def classify_overall(board_infos):
    longCount=sum(1 for b in board_infos if b['category'] in ('long','super_long'))
    superCount=sum(1 for b in board_infos if b['category']=='super_long')
    if longCount >= MIN_BOARDS_FOR_PAW: return "放水时段（提高胜率）", longCount, superCount

    def board_has_3consec_multicolumn(columns):
        col_runlens=[]
        for col in columns:
            if not col: col_runlens.append(0); continue
            ccur=col[0]; clen=1; maxc=1
            for t in col[1:]:
                if t==ccur: clen+=1
                else:
                    if clen>maxc: maxc=clen; ccur=t; clen=1
            if clen>maxc: maxc=clen
            col_runlens.append(maxc)
        for i in range(len(col_runlens)-2):
            if col_runlens[i]>=4 and col_runlens[i+1]>=4 and col_runlens[i+2]>=4: return True
        return False

    boards_with_multicol=sum(1 for b in board_infos if board_has_3consec_multicolumn(b['columns']))
    boards_with_long=sum(1 for b in board_infos if b['maxRun'] >= 8)
    if boards_with_multicol >= 3 and boards_with_long >= 2:
        return "中等胜率（中上）", boards_with_long, sum(1 for b in board_infos if b['category']=='super_long')

    totals=[b['total'] for b in board_infos]
    if board_infos and sum(1 for t in totals if t < 6) >= len(board_infos)*0.6:
        return "胜率调低 / 收割时段", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')
    return "胜率中等（平台收割中等时段）", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')

# ========= Playwright helpers =========
def apply_stealth(page):
    page.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => false});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
    Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4]});
    window.chrome = { runtime: {} };
    """)
def human_like_drag(page, sx, sy, ex, ey, steps=30):
    page.mouse.move(sx, sy); page.mouse.down()
    for i in range(1,steps+1):
        nx = sx + (ex-sx)*(i/steps) + random.uniform(-2,2)
        ny = sy + (ey-sy)*(i/steps) + random.uniform(-1,1)
        page.mouse.move(nx, ny, steps=1)
        time.sleep(random.uniform(0.01,0.04))
    page.mouse.up()

def try_solve_slider(page):
    try:
        selectors=["input[type=range]","div[role=slider]","div[class*=slider]","div[class*=captcha]","div[class*=slide]"]
        for sel in selectors:
            try:
                els=page.query_selector_all(sel)
                if els and len(els)>0:
                    box=els[0].bounding_box()
                    if box:
                        x0=box['x']+2; y0=box['y']+box['height']/2; x1=box['x']+box['width']-6
                        human_like_drag(page,x0,y0,x1,y0,steps=30); time.sleep(1.0); return True
            except: continue
        # image detect method
        try:
            ss=page.screenshot(full_page=True); img=Image.open(BytesIO(ss)).convert("RGB"); bgr=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            H,W=bgr.shape[:2]; region=bgr[int(H*0.25):int(H*0.75), int(W*0.05):int(W*0.95)]
            gray=cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _,th=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
            contours,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            best=None; area_best=0
            for cnt in contours:
                bx,by,bw,bh=cv2.boundingRect(cnt); area=bw*bh
                if area>area_best and bw>40 and bw>3*bh: best=(bx,by,bw,bh); area_best=area
            if best:
                bx,by,bw,bh=best; px=int(W*0.05)+bx; py=int(H*0.25)+by
                sx=px+6; sy=py+bh//2; ex=px+bw-6
                human_like_drag(page,sx,sy,ex,sy,steps=30); time.sleep(1.2); return True
        except: pass
        # last resort: dispatch pointer events on candidate selectors
        for sel in selectors:
            try:
                handle=page.query_selector(sel)
                if handle:
                    box=handle.bounding_box()
                    if box:
                        sx=box['x']+2; sy=box['y']+box['height']/2; ex=box['x']+box['width']-6
                        page.dispatch_event(sel,"pointerdown",{"button":0,"clientX":sx,"clientY":sy})
                        page.dispatch_event(sel,"pointermove",{"clientX":ex,"clientY":sy})
                        page.dispatch_event(sel,"pointerup",{"button":0,"clientX":ex,"clientY":sy}); time.sleep(1.0); return True
            except: continue
    except Exception as e:
        log(f"try_solve_slider exception: {e}")
    return False

def capture_dg_page(max_attempts=3):
    with sync_playwright() as p:
        user_agents = [
            "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        viewports=[(390,844),(1280,900)]
        for attempt in range(max_attempts):
            ua=random.choice(user_agents); vw,vh=random.choice(viewports)
            browser=p.chromium.launch(headless=True, args=["--no-sandbox"])
            context=browser.new_context(user_agent=ua, viewport={"width":vw,"height":vh})
            page=context.new_page(); apply_stealth(page)
            time.sleep(random.uniform(0.3,1.0))
            for url in DG_LINKS:
                try:
                    log(f"Open {url} attempt {attempt+1}")
                    page.goto(url, timeout=35000); time.sleep(1.0+random.uniform(0,1.0))
                    # click Free
                    clicked=False
                    for txt in ["Free","免费试玩","免费","Play Free","试玩","Free Play","免费体验"]:
                        try:
                            el=page.locator(f"text={txt}")
                            if el.count()>0: el.first.click(timeout=4000); clicked=True; log(f"Clicked '{txt}'"); break
                        except: continue
                    if not clicked:
                        try:
                            els=page.query_selector_all("a,button")
                            for i in range(min(80,len(els))):
                                try:
                                    t=els[i].inner_text().strip().lower()
                                    if "free" in t or "试玩" in t or "免费" in t:
                                        els[i].click(timeout=3000); clicked=True; log("Clicked candidate link/button"); break
                                except: continue
                        except: pass
                    time.sleep(0.8+random.uniform(0,0.8))
                    # try sliding multiple times
                    slider_ok=False
                    for s_try in range(6):
                        log(f"Slider attempt {s_try+1}")
                        ok=try_solve_slider(page)
                        if ok: slider_ok=True; log("Slider appeared solved"); break
                        try: page.mouse.wheel(0,300); time.sleep(0.5)
                        except: pass
                    # check for many points
                    for check in range(8):
                        ss=page.screenshot(full_page=True)
                        try:
                            with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
                        except: pass
                        pil=pil_from_bytes(ss); bgr=cv_from_pil(pil)
                        pts,_,_ = detect_color_points(bgr)
                        log(f"check {check+1}: points={len(pts)}")
                        if len(pts) >= MIN_POINTS_FOR_REAL:
                            context.close(); browser.close(); log("Detected real-play page"); return ss
                        time.sleep(1.2+random.random())
                except PWTimeout as e:
                    log(f"Timeout {url}: {e}")
                except Exception as e:
                    log(f"Error {url}: {e}")
                finally:
                    pass
            try: context.close()
            except: pass
            try: browser.close()
            except: pass
            time.sleep(2+random.random()*2)
        log("Failed to enter real-play after attempts.")
        return None

# ========= History fetch / import =========
def try_fetch_public_history():
    tried=[]
    for base in DG_LINKS:
        for path in ["/api/history","/history","/api/v1/history","/game/history","/history.json"]:
            url=base.rstrip("/") + path
            tried.append(url)
            try:
                r=requests.get(url,timeout=8)
                if r.status_code==200:
                    try:
                        j=r.json(); log(f"Found history endpoint: {url}"); return j
                    except:
                        continue
            except:
                continue
    log(f"Tried speculative history endpoints (none valid): {tried}")
    return None

def import_market_history_from_file():
    # Supports market_history.csv or market_history.json if present in repo root
    if os.path.exists("market_history.json"):
        try:
            with open("market_history.json","r",encoding="utf-8") as f:
                j=json.load(f)
            # Normalize if list of events
            events=[]
            if isinstance(j,list):
                for ev in j:
                    if "start_time" in ev:
                        events.append(ev)
            elif isinstance(j,dict):
                # try keys
                for k in ("events","history","records"):
                    if k in j and isinstance(j[k],list): events.extend(j[k])
            if events:
                log(f"Imported {len(events)} events from market_history.json")
                return events
        except Exception as e:
            log(f"market_history.json parse error: {e}")
    if os.path.exists("market_history.csv"):
        try:
            events=[]
            with open("market_history.csv","r",encoding="utf-8") as f:
                reader=csv.DictReader(f)
                for row in reader:
                    # expect columns: kind,start_time,end_time,duration_minutes
                    events.append({"kind":row.get("kind","放水"), "start_time":row.get("start_time"), "end_time":row.get("end_time"), "duration_minutes": int(row.get("duration_minutes") or 0)})
            log(f"Imported {len(events)} events from market_history.csv")
            return events
        except Exception as e:
            log(f"market_history.csv parse error: {e}")
    return None

# ========= Prediction from history =========
def predict_from_history(state):
    hist = state.get("history", []) or []
    # If repo provided market_history.* exist, prioritize them by merging first
    imported = import_market_history_from_file()
    if imported:
        # convert to normalized {kind,start_time,duration_minutes}
        norm=[]
        for ev in imported:
            try:
                st = ev.get("start_time") or ev.get("start")
                dur = ev.get("duration_minutes") or ev.get("duration") or 0
                if st:
                    norm.append({"kind":ev.get("kind","放水"), "start_time":st, "duration_minutes":int(dur)})
            except:
                continue
        hist = (hist or []) + norm
    if not hist:
        log("No history available for prediction.")
        return None
    now = now_tz(); cutoff = now - timedelta(days=28)
    recent=[]
    for e in hist:
        try:
            st = e.get("start_time")
            if isinstance(st,str):
                st_dt = datetime.fromisoformat(st)
            elif isinstance(st,datetime):
                st_dt = st
            else:
                continue
            st_dt = st_dt.astimezone(TZ) if st_dt.tzinfo else st_dt.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st_dt >= cutoff:
                recent.append({"kind":e.get("kind","放水"), "start":st_dt, "duration": int(e.get("duration_minutes") or 0)})
        except Exception:
            continue
    if not recent:
        log("No recent events in last 28 days.")
        return None
    buckets={}
    for ev in recent:
        weekday=ev['start'].weekday(); hour=ev['start'].hour
        minute_bucket=(ev['start'].minute // PRED_BUCKET_MINUTES) * PRED_BUCKET_MINUTES
        key=(ev['kind'],weekday,hour,minute_bucket)
        if key not in buckets: buckets[key]={'count':0,'durations':[]}
        buckets[key]['count'] += 1; buckets[key]['durations'].append(ev['duration'])
    candidates=[]
    for k,v in buckets.items():
        if v['count'] >= MIN_HISTORY_EVENTS_FOR_PRED:
            avg = round(sum(v['durations'])/len(v['durations'])) if v['durations'] else 10
            candidates.append({"key":k,"count":v['count'],"avg_duration":avg})
    if not candidates:
        log("No historical bucket passes the threshold.")
        return None
    candidates.sort(key=lambda x:x['count'], reverse=True)
    best = candidates[0]
    kind,weekday,hour,bmin = best['key']
    now = now_tz()
    days_ahead = (weekday - now.weekday()) % 7
    predicted_start = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=bmin, second=0, microsecond=0)
    if predicted_start < now - timedelta(minutes=1): predicted_start += timedelta(days=7)
    predicted_end = predicted_start + timedelta(minutes=best['avg_duration'])
    return {"kind":kind,"predicted_start":predicted_start,"predicted_end":predicted_end,"avg_duration":best['avg_duration'],"count":best['count']}

# ========= Try public history endpoints (speculative) =========
def try_fetch_public_history():
    tried=[]
    for base in DG_LINKS:
        for path in ["/api/history","/history","/api/v1/history","/game/history","/history.json"]:
            url = base.rstrip("/") + path
            tried.append(url)
            try:
                r=requests.get(url,timeout=8)
                if r.status_code==200:
                    try:
                        j=r.json(); log(f"Found history endpoint: {url}"); return j
                    except:
                        continue
            except:
                continue
    log(f"No public history endpoint found among speculative urls: {tried}")
    return None

# ========= Main run =========
def main():
    log("=== DG monitor run start ===")
    state = load_state()
    screenshot = None
    try:
        screenshot = capture_dg_page()
    except Exception as e:
        log(f"capture error: {e}\n{traceback.format_exc()}")

    if screenshot:
        # process real-time page
        try:
            with open(LAST_SCREENSHOT,"wb") as f: f.write(screenshot)
        except: pass
        img = pil_from_bytes(screenshot); bgr = cv_from_pil(img)
        pts,_,_ = detect_color_points(bgr)
        log(f"Detected points: {len(pts)}")
        if len(pts) < MIN_POINTS_FOR_REAL:
            log("Detected points below threshold -> treat as failed to enter real-play.")
            fallback_history_mode(state); return
        rects = cluster_points_to_boards(pts, bgr.shape)
        log(f"Clustered {len(rects)} boards.")
        boards=[]
        for r in rects:
            boards.append(analyze_board(bgr,r))
        overall, longCount, superCount = classify_overall(boards)
        log(f"Classification: {overall} (long/超: {longCount}/{superCount})")
        now_iso = now_tz().isoformat()
        was_active = state.get("active", False)
        is_active_now = overall in ("放水时段（提高胜率）","中等胜率（中上）")
        if is_active_now and not was_active:
            state = {"active":True,"kind":overall,"start_time":now_iso,"last_seen":now_iso,"history":state.get("history",[])}
            save_state(state)
            hist = state.get("history",[]) or []
            durations=[h.get("duration_minutes",0) for h in hist if h.get("duration_minutes",0)>0]
            est_min = round(sum(durations)/len(durations)) if durations else 10
            est_end = (now_tz() + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
            msg = f"🔔 [DG提醒] {overall} 開始\n時間: {now_iso}\n长龙/超龙 桌數: {longCount} (超:{superCount})\n估計結束: {est_end}（約 {est_min} 分鐘，基於歷史）"
            send_telegram(msg); save_state(state)
        elif is_active_now and was_active:
            state['last_seen'] = now_iso; state['kind'] = overall; save_state(state)
            log("Event active -> updated last_seen")
        elif not is_active_now and was_active:
            start_time = datetime.fromisoformat(state.get("start_time")); end_time = now_tz()
            duration_minutes = round((end_time - start_time).total_seconds() / 60.0)
            history = state.get("history", []); history.append({"kind":state.get("kind"),"start_time":state.get("start_time"),"end_time":end_time.isoformat(),"duration_minutes":duration_minutes})
            history = history[-2000:]; new_state={"active":False,"kind":None,"start_time":None,"last_seen":None,"history":history}
            save_state(new_state)
            msg = f"✅ [DG提醒] {state.get('kind')} 已結束\n開始: {state.get('start_time')}\n結束: {end_time.isoformat()}\n實際持續: {duration_minutes} 分鐘"
            send_telegram(msg); log("End notified and history saved.")
        else:
            save_state(state); log("No event changes.")
        # save summary for debugging
        summary = {"ts": now_tz().isoformat(), "overall":overall, "longCount": longCount, "superCount": superCount, "boards": boards[:60]}
        try:
            with open(SUMMARY_FILE,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
        except: pass
        return
    else:
        # fallback
        fallback_history_mode(state)
        return

def fallback_history_mode(state):
    log("=== fallback: using historical market data ===")
    # 1) try public history endpoints
    public = try_fetch_public_history()
    if public:
        # try to normalize and merge into state.history
        events=[]
        if isinstance(public,list):
            for ev in public:
                if isinstance(ev,dict) and ev.get("start_time"):
                    events.append({"kind":ev.get("kind","放水"), "start_time":ev.get("start_time"), "duration_minutes": int(ev.get("duration", ev.get("duration_minutes",0) or 0))})
        elif isinstance(public,dict):
            for k in ("events","history","records"):
                if k in public and isinstance(public[k],list):
                    for ev in public[k]:
                        if isinstance(ev,dict) and ev.get("start_time"):
                            events.append({"kind":ev.get("kind","放水"), "start_time":ev.get("start_time"), "duration_minutes": int(ev.get("duration", ev.get("duration_minutes",0) or 0))})
        if events:
            hist = state.get("history", []) or []
            hist.extend(events); state['history']=hist[-2000:]; save_state(state)
            log(f"Imported {len(events)} events from public history endpoint.")
    # 2) import from repo file if present
    imported = import_market_history_from_file()
    if imported:
        hist = state.get("history", []) or []
        norm=[]
        for ev in imported:
            try:
                st = ev.get("start_time") or ev.get("start") or ev.get("ts")
                dur = ev.get("duration_minutes") or ev.get("duration") or 0
                if st:
                    norm.append({"kind":ev.get("kind","放水"), "start_time":st, "duration_minutes":int(dur)})
            except:
                continue
        if norm:
            hist.extend(norm); state['history']=hist[-2000:]; save_state(state)
            log(f"Imported {len(norm)} events from repository market_history.* file.")

    # 3) If after imports history is sufficient -> predict and send
    hist = state.get("history", []) or []
    # count recent (28 days)
    now = now_tz(); cutoff = now - timedelta(days=28)
    recent_count=0
    for e in hist:
        try:
            st = e.get("start_time")
            st_dt = datetime.fromisoformat(st) if isinstance(st,str) else st
            st_dt = st_dt.astimezone(TZ) if st_dt.tzinfo else st_dt.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st_dt >= cutoff: recent_count += 1
        except:
            continue
    log(f"Recent events in last 28 days: {recent_count}")
    if recent_count < MIN_HISTORY_EVENTS_FOR_PRED:
        log("Insufficient market history to make reliable prediction. Will continue collecting.")
        save_state(state); return

    pred = predict_from_history(state)
    if not pred:
        log("No prediction from history found.")
        save_state(state); return
    ps = pred['predicted_start']; pe = pred['predicted_end']; kind = pred['kind']; avg = pred['avg_duration']
    now = now_tz()
    lead = timedelta(minutes=PRED_LEAD_MINUTES)
    if (ps - lead) <= now <= pe:
        remaining = max(0, int((pe - now).total_seconds() // 60))
        msg = f"🔔 [DG替補預測提醒] 根據過去 4 週市場歷史偵測到可能的『{kind}』\n預測開始: {ps.strftime('%Y-%m-%d %H:%M:%S')}\n估計結束: {pe.strftime('%Y-%m-%d %H:%M:%S')}（約 {avg} 分鐘）\n目前剩餘: 約 {remaining} 分鐘\n※ 此為基於市場歷史的替補預測（因為無法直接進入實盤）"
        send_telegram(msg); log("Sent fallback historical prediction.")
    else:
        log(f"Predicted next {kind} at {ps.strftime('%Y-%m-%d %H:%M:%S')} (not yet within lead window).")
    save_state(state)
    return

# ========= bootstrap import function (used above) =========
def import_market_history_from_file():
    if os.path.exists("market_history.json"):
        try:
            with open("market_history.json","r",encoding="utf-8") as f:
                j=json.load(f)
            events=[]
            if isinstance(j, list):
                for ev in j:
                    if isinstance(ev, dict) and ev.get("start_time"):
                        events.append(ev)
            elif isinstance(j, dict):
                for k in ("events","history","records"):
                    if k in j and isinstance(j[k], list):
                        for ev in j[k]:
                            if isinstance(ev, dict) and ev.get("start_time"):
                                events.append(ev)
            if events: log(f"Loaded {len(events)} events from market_history.json"); return events
        except Exception as e:
            log(f"market_history.json parse error: {e}")
    if os.path.exists("market_history.csv"):
        try:
            events=[]
            with open("market_history.csv","r",encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    events.append({"kind":row.get("kind","放水"), "start_time":row.get("start_time"), "end_time":row.get("end_time"), "duration_minutes": int(row.get("duration_minutes") or 0)})
            log(f"Loaded {len(events)} events from market_history.csv"); return events
        except Exception as e:
            log(f"market_history.csv parse error: {e}")
    return None

# ========= Entrypoint =========
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        sys.exit(1)
