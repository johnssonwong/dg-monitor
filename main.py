# main.py
# DG 实盘检测 + 立即历史替补（Wayback / 公共 API）-> Telegram
# IMPORTANT:
#  - 保持文件名为 main.py （不要改名）
#  - 在运行环境中设置 TG_BOT_TOKEN 与 TG_CHAT_ID（或在 GitHub Secrets 注入）
#  - 可选：设置 FORCE_TEST_SEND=1 来强制发送一次测试 Telegram（部署验证时用）
#
# 改进要点（本版本）：
#  1) 如果无法进入实盘或点数不足 -> 立即发送 Telegram: "进入失败，开始替补"（让你知道脚本在工作）
#  2) 替补若找到符合条件的事件 -> 立即发送放水/中等胜率提醒（格式含 emoji、开始/估计结束/剩余分钟）
#  3) 若替补无法找到足够历史 -> 立即发送 Telegram: "替补无足够历史"（避免你白等）
#  4) 在每个关键步骤写入 last_screenshot.png、last_summary.json、state.json，便于调试
#
# 运行提示：
#  - 本脚本依赖 playwright、Pillow、opencv-python-headless、numpy、requests
#  - 如果在 GitHub Actions 中，确保 workflow 有足够 timeout（Wayback 渲染可能耗时）
#  - 若用本地运行，请先 install requirements 并 playwright install chromium
# ------------------------------------------------------------------------------

import os, sys, json, time, random, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import cv2
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------- 配置（可用环境变量覆盖） ----------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN_ENV = "TG_BOT_TOKEN"
TG_CHAT_ENV = "TG_CHAT_ID"

# 阈值与参数
MIN_POINTS_FOR_REAL = int(os.environ.get("MIN_POINTS_FOR_REAL", "40"))
DILATE_KERNEL_SIZE = int(os.environ.get("DILATE_KERNEL_SIZE", "40"))
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))
HISTORY_LOOKBACK_DAYS = int(os.environ.get("HISTORY_LOOKBACK_DAYS", "28"))
MIN_HISTORY_EVENTS_FOR_PRED = int(os.environ.get("MIN_HISTORY_EVENTS_FOR_PRED", "3"))
PRED_BUCKET_MINUTES = int(os.environ.get("PRED_BUCKET_MINUTES", "15"))
PRED_LEAD_MINUTES = int(os.environ.get("PRED_LEAD_MINUTES", "10"))
WAYBACK_MAX_SNAPSHOTS = int(os.environ.get("WAYBACK_MAX_SNAPSHOTS","40"))
WAYBACK_RATE_SLEEP = float(os.environ.get("WAYBACK_RATE_SLEEP","1.2"))

FORCE_TEST_SEND = os.environ.get("FORCE_TEST_SEND", "0") in ("1","true","True")

STATE_FILE = "state.json"
SUMMARY_FILE = "last_summary.json"
LAST_SCREENSHOT = "last_screenshot.png"

TZ = timezone(timedelta(hours=8))  # Malaysia UTC+8

# ---------- 工具 ----------
def now_tz(): return datetime.now(TZ)
def nowstr(): return now_tz().strftime("%Y-%m-%d %H:%M:%S")
def log(s): print(f"[{nowstr()}] {s}", flush=True)

# ---------- Telegram ----------
def send_telegram(text, max_retries=3):
    token = os.environ.get(TG_TOKEN_ENV, "").strip()
    chat = os.environ.get(TG_CHAT_ENV, "").strip()
    if not token or not chat:
        log("Telegram 未配置（TG_BOT_TOKEN 或 TG_CHAT_ID 缺失）")
        return False, "no_token"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat, "text": text}
    for i in range(max_retries):
        try:
            r = requests.post(url, data=data, timeout=12)
            j = r.json()
            if j.get("ok"):
                log("Telegram 发送成功")
                return True, j
            else:
                log(f"Telegram 返回非 ok: {j}")
                time.sleep(1 + random.random())
        except Exception as e:
            log(f"Telegram 发送异常 (尝试 {i+1}): {e}")
            time.sleep(1 + random.random())
    return False, "failed"

# ---------- state --------------
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

# ---------- image helpers ---------------
def pil_from_bytes(b): return Image.open(BytesIO(b)).convert("RGB")
def cv_from_pil(p): return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

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
        if 0<=x<w and 0<=y<h: mask[y,x]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE,DILATE_KERNEL_SIZE))
    big = cv2.dilate(mask, kernel, iterations=1)
    num,labels,stats,_ = cv2.connectedComponentsWithStats(big, connectivity=8)
    rects=[]
    for i in range(1,num):
        x,y,w_,h_ = stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP], stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]
        if w_<60 or h_<40: continue
        pad=8
        x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w-1,x+w_+pad); y1=min(h-1,y+h_+pad)
        rects.append((x0,y0,x1-x0,y1-y0))
    if not rects:
        cols=max(3,w//300); rows=max(2,h//200)
        cw=w//cols; ch=h//rows
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
        cur={'color':flattened[0],'len':1}
        for k in range(1,len(flattened)):
            if flattened[k]==cur['color']: cur['len']+=1
            else:
                runs.append(cur); cur={'color':flattened[k],'len':1}
        runs.append(cur)
    maxRun = max((r['len'] for r in runs), default=0)
    cat='other'
    if maxRun>=10: cat='super_long'
    elif maxRun>=8: cat='long'
    elif maxRun>=4: cat='longish'
    elif maxRun==1: cat='single'
    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"columns":columns,"runs":runs}

# ---------- classification ----------
def classify_overall(board_infos):
    longCount = sum(1 for b in board_infos if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_infos if b['category']=='super_long')
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
    boards_with_multicol = sum(1 for b in board_infos if board_has_3consec_multicolumn(b['columns']))
    boards_with_long = sum(1 for b in board_infos if b['maxRun'] >= 8)
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount
    if boards_with_multicol >= 3 and boards_with_long >= 2:
        return "中等胜率（中上）", boards_with_long, sum(1 for b in board_infos if b['category']=='super_long')
    totals = [b['total'] for b in board_infos]
    if board_infos and sum(1 for t in totals if t < 6) >= len(board_infos)*0.6:
        return "胜率调低 / 收割时段", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')
    return "胜率中等（平台收割中等时段）", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')

# ---------- Playwright helpers ----------
def apply_stealth(page):
    page.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => false});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
    Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4]});
    window.chrome = { runtime: {} };
    """)

def human_like_drag(page, start_x, start_y, end_x, end_y, steps=30):
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    for i in range(1, steps+1):
        nx = start_x + (end_x - start_x) * (i/steps) + random.uniform(-2,2)
        ny = start_y + (end_y - start_y) * (i/steps) + random.uniform(-1,1)
        page.mouse.move(nx, ny, steps=1)
        time.sleep(random.uniform(0.01, 0.03))
    page.mouse.up()

def try_solve_slider(page):
    try:
        selectors = ["input[type=range]","div[role=slider]","div[class*=slider]","div[class*=captcha]","div[class*=slide]"]
        for sel in selectors:
            try:
                els = page.query_selector_all(sel)
                if els and len(els)>0:
                    box = els[0].bounding_box()
                    if box:
                        x0 = box['x']+2; y0 = box['y'] + box['height']/2
                        x1 = box['x'] + box['width'] - 6
                        human_like_drag(page, x0, y0, x1, y0, steps=30)
                        time.sleep(0.8)
                        return True
            except:
                continue
        # 截图图像辅助找 slider
        ss = page.screenshot(full_page=True)
        img = pil_from_bytes(ss); bgr = cv_from_pil(img)
        H,W = bgr.shape[:2]
        region = bgr[int(H*0.25):int(H*0.75), int(W*0.05):int(W*0.95)]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _,th = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best=None; best_area=0
        for cnt in cnts:
            bx,by,bw,bh = cv2.boundingRect(cnt); area=bw*bh
            if area>best_area and bw>40 and bw>3*bh:
                best=(bx,by,bw,bh); best_area=area
        if best:
            bx,by,bw,bh = best
            px=int(W*0.05)+bx; py=int(H*0.25)+by
            sx = px+6; sy = py + bh//2; ex = px + bw - 6
            human_like_drag(page, sx, sy, ex, sy, steps=30)
            time.sleep(1.0)
            return True
    except Exception as e:
        log(f"try_solve_slider 异常: {e}")
    return False

def capture_dg_page(attempts=3):
    with sync_playwright() as p:
        uas = [
            "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        viewports = [(390,844),(1280,900)]
        for attempt in range(attempts):
            ua = random.choice(uas); vw,vh = random.choice(viewports)
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(user_agent=ua, viewport={"width":vw,"height":vh}, locale="en-US")
            page = context.new_page(); apply_stealth(page)
            time.sleep(random.uniform(0.3,0.9))
            for url in DG_LINKS:
                try:
                    log(f"打开 {url} （尝试 {attempt+1}）")
                    page.goto(url, timeout=30000)
                    time.sleep(0.8 + random.random())
                    clicked=False
                    for txt in ["Free","免费试玩","免费","Play Free","试玩","Free Play","免费体验"]:
                        try:
                            loc = page.locator(f"text={txt}")
                            if loc.count()>0:
                                loc.first.click(timeout=3000); clicked=True; log(f"点击文本 {txt}"); break
                        except:
                            continue
                    if not clicked:
                        try:
                            els = page.query_selector_all("a,button")
                            for i in range(min(80,len(els))):
                                try:
                                    t = els[i].inner_text().strip().lower()
                                    if "free" in t or "试玩" in t or "免费" in t:
                                        els[i].click(timeout=2000); clicked=True; log("点击候选 a/button"); break
                                except:
                                    continue
                        except:
                            pass
                    time.sleep(0.6 + random.random())
                    # 尝试滑块多次
                    for s in range(6):
                        got = try_solve_slider(page)
                        log(f"slider 尝试 {s+1} -> {got}")
                        if got: break
                        else:
                            try:
                                page.mouse.wheel(0,300); time.sleep(0.5)
                            except:
                                pass
                    # 检测点数
                    for chk in range(8):
                        ss = page.screenshot(full_page=True)
                        try:
                            with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
                        except:
                            pass
                        img = pil_from_bytes(ss); bgr = cv_from_pil(img)
                        pts,_,_ = detect_color_points(bgr)
                        log(f"检查 {chk+1}: 点数 {len(pts)}")
                        if len(pts) >= MIN_POINTS_FOR_REAL:
                            log("进入实盘（点数阈值达成）")
                            context.close(); browser.close()
                            return ss
                        time.sleep(1.0 + random.random() * 0.6)
                except PWTimeout as e:
                    log(f"页面打开超时: {e}")
                except Exception as e:
                    log(f"与页面交互异常: {e}")
            try:
                context.close()
            except: pass
            try: browser.close()
            except: pass
            time.sleep(1.8 + random.random())
        log("未能进入实盘（多次尝试失败）")
        return None

# ---------- Wayback / history helpers ----------
def get_wayback_snapshots(url, from_date=None, to_date=None, limit=40):
    base = "http://web.archive.org/cdx/search/cdx"
    params = {"url":url, "output":"json", "filter":"statuscode:200", "limit":str(limit)}
    if from_date: params["from"]=from_date
    if to_date: params["to"]=to_date
    try:
        r = requests.get(base, params=params, timeout=12)
        if r.status_code!=200:
            log(f"Wayback CDX 返回 {r.status_code}")
            return []
        j = r.json()
        if not j or len(j)<2: return []
        rows = j[1:]
        tss = [row[1] for row in rows if len(row)>1]
        return tss
    except Exception as e:
        log(f"Wayback CDX 异常: {e}")
        return []

def fetch_wayback_snapshot_and_screenshot(snapshot_ts, target_url):
    snap = f"https://web.archive.org/web/{snapshot_ts}/{target_url}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(viewport={"width":1280,"height":900})
            page = context.new_page()
            page.goto(snap, timeout=30000)
            time.sleep(1.8)
            ss = page.screenshot(full_page=True)
            context.close(); browser.close()
            return ss
    except Exception as e:
        log(f"Wayback 渲染失败 {snapshot_ts}: {e}")
        return None

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
    log(f"尝试公共历史接口: {tried}")
    return None

def predict_from_history(state):
    hist = state.get("history",[]) or []
    now = now_tz()
    cutoff = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
    recent=[]
    for ev in hist:
        try:
            st = datetime.fromisoformat(ev["start_time"])
            st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= cutoff: recent.append({"kind":ev.get("kind"), "start":st, "duration":ev.get("duration_minutes",0)})
        except:
            continue
    if len(recent) < MIN_HISTORY_EVENTS_FOR_PRED:
        return None
    buckets={}
    for ev in recent:
        weekday = ev["start"].weekday(); hour = ev["start"].hour
        bmin = (ev["start"].minute // PRED_BUCKET_MINUTES) * PRED_BUCKET_MINUTES
        key=(ev["kind"], weekday, hour, bmin)
        if key not in buckets: buckets[key]={"count":0,"durations":[]}
        buckets[key]["count"]+=1; buckets[key]["durations"].append(ev["duration"])
    candidates=[]
    for k,v in buckets.items():
        if v["count"] >= MIN_HISTORY_EVENTS_FOR_PRED:
            avg = round(sum(v["durations"])/len(v["durations"])) if v["durations"] else 10
            candidates.append({"key":k,"count":v["count"],"avg_duration":avg})
    if not candidates: return None
    candidates.sort(key=lambda x:x["count"], reverse=True)
    best = candidates[0]
    kind, weekday, hour, bmin = best["key"]
    now = now_tz()
    days_ahead = (weekday - now.weekday()) % 7
    predicted_start = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=bmin, second=0, microsecond=0)
    if predicted_start < now - timedelta(minutes=1):
        predicted_start += timedelta(days=7)
    predicted_end = predicted_start + timedelta(minutes=best["avg_duration"])
    return {"kind":kind, "predicted_start":predicted_start, "predicted_end":predicted_end, "avg_duration":best["avg_duration"], "count":best["count"]}

# ---------- main logic ----------
def main():
    log("=== DG monitor run start ===")
    # optional forced test send (use for debug only; set FORCE_TEST_SEND=1 env to test)
    if FORCE_TEST_SEND:
        send_telegram("🔧 [DG Monitor] 测试消息：FORCE_TEST_SEND=1，Telegram 通知正常。")
        log("已发送强制测试 Telegram（FORCE_TEST_SEND=1）。退出以避免重复发送。")
        return

    state = load_state()
    screenshot = None
    try:
        screenshot = capture_dg_page()
    except Exception as e:
        log(f"capture_dg_page 异常: {e}\n{traceback.format_exc()}")

    # 如果得到了截图，做实时检测
    if screenshot:
        with open(LAST_SCREENSHOT,"wb") as f: f.write(screenshot)
        img = Image.open(LAST_SCREENSHOT).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        pts,_,_ = detect_color_points(bgr)
        log(f"实时检测点数: {len(pts)}")
        if len(pts) < MIN_POINTS_FOR_REAL:
            # 点数不足 -> 立即通知并进入替补
            send_telegram("⚠️ [DG Monitor] 已进入页面但点数不足，准备执行历史替补判定（马上通知结果）。")
            fallback_with_history(state)
            return
        rects = cluster_points_to_boards(pts, bgr.shape)
        boards=[]
        for r in rects: boards.append(analyze_board(bgr, r))
        overall, longCount, superCount = classify_overall(boards)
        log(f"实时判定: {overall} (长龙/超: {longCount}/{superCount})")
        nowiso = now_tz().isoformat()
        was_active = state.get("active", False)
        is_active_now = overall in ("放水时段（提高胜率）","中等胜率（中上）")
        if is_active_now and not was_active:
            state={"active":True,"kind":overall,"start_time":nowiso,"last_seen":nowiso,"history":state.get("history",[])}
            save_state(state)
            durations = [h.get("duration_minutes",0) for h in state.get("history",[]) if h.get("duration_minutes",0)>0]
            est_min = round(sum(durations)/len(durations)) if durations else 10
            est_end = (now_tz() + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
            msg = f"🔔 [DG提醒 - 即時] {overall} 開始\n時間: {nowiso}\n长龙/超龙 桌數: {longCount} (超龙:{superCount})\n估計結束: {est_end}（約 {est_min} 分鐘）"
            send_telegram(msg)
            save_state(state)
        elif is_active_now and was_active:
            state["last_seen"]=nowiso; state["kind"]=overall; save_state(state); log("事件持續中")
        elif not is_active_now and was_active:
            start_time = datetime.fromisoformat(state.get("start_time"))
            end_time = now_tz()
            dur = round((end_time - start_time).total_seconds() / 60.0)
            hist = state.get("history",[]); hist.append({"kind":state.get("kind"), "start_time": state.get("start_time"), "end_time": end_time.isoformat(), "duration_minutes": dur})
            state={"active":False,"kind":None,"start_time":None,"last_seen":None,"history": hist[-2000:]}
            save_state(state)
            msg = f"✅ [DG提醒] {state.get('kind')} 已結束\n開始: {state.get('start_time')}\n結束: {end_time.isoformat()}\n實際持續: {dur} 分鐘"
            send_telegram(msg)
            log("已發結束通知")
        else:
            log("非放水/中上時段，不發送提醒（但本次實盤檢測成功）")
        # 写 summary 方便调试
        try:
            with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
                json.dump({"ts":now_tz().isoformat(),"overall":overall,"longCount":longCount,"superCount":superCount}, f, ensure_ascii=False, indent=2)
        except:
            pass
        return
    else:
        # 未成功进入实盘 -> 立刻通知并进入历史替补
        send_telegram("⚠️ [DG Monitor] 未能进入 DG 实盘，立即启动历史替补（Wayback/公共 API）并马上通知结果。")
        fallback_with_history(state)
        return

def fallback_with_history(state):
    log("开始历史替补流程（Wayback + 公共 API）")
    # 1) 尝试公共历史 API
    try:
        api_hist = try_fetch_public_history()
    except Exception as e:
        api_hist = None
        log(f"try_fetch_public_history 异常: {e}")
    if api_hist:
        # 归一化并合入 state.history
        norm=[]
        if isinstance(api_hist, list):
            for rec in api_hist:
                st = rec.get("start_time") or rec.get("ts") or rec.get("time")
                end = rec.get("end_time") or rec.get("end")
                dur = rec.get("duration_minutes") or rec.get("duration")
                if st:
                    norm.append({"kind": rec.get("kind","放水"), "start_time": st, "end_time": end, "duration_minutes": dur or 0})
        elif isinstance(api_hist, dict):
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
            hist.extend(norm)
            state["history"] = hist[-2000:]
            save_state(state)
            log(f"从公共 API 导入 {len(norm)} 条历史")
    # 2) 评估现有历史是否足够
    recent_count = 0
    for h in state.get("history",[]) or []:
        try:
            st = datetime.fromisoformat(h["start_time"]); st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS): recent_count += 1
        except:
            continue
    if recent_count < MIN_HISTORY_EVENTS_FOR_PRED:
        log("历史记录不足 -> 尝试 Wayback 快照收集")
        from_date = (now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime("%Y%m%d")
        to_date = now_tz().strftime("%Y%m%d")
        collected = 0
        for base in DG_LINKS:
            tss = get_wayback_snapshots(base, from_date=from_date, to_date=to_date, limit=WAYBACK_MAX_SNAPSHOTS)
            if not tss:
                log(f"Wayback 未发现 {base} 的快照（过去 {HISTORY_LOOKBACK_DAYS} 天）")
                continue
            for ts in tss[:WAYBACK_MAX_SNAPSHOTS]:
                time.sleep(WAYBACK_RATE_SLEEP)
                ss = fetch_wayback_snapshot_and_screenshot(ts, base)
                if not ss: continue
                try:
                    with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
                except: pass
                try:
                    img = Image.open(LAST_SCREENSHOT).convert("RGB")
                    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                except:
                    continue
                pts,_,_ = detect_color_points(bgr)
                if len(pts) < MIN_POINTS_FOR_REAL: continue
                rects = cluster_points_to_boards(pts, bgr.shape)
                boards=[]
                for r in rects: boards.append(analyze_board(bgr, r))
                overall, longCount, superCount = classify_overall(boards)
                if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
                    try:
                        ev_time = datetime.strptime(ts[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).astimezone(TZ)
                    except:
                        ev_time = now_tz()
                    rec = {"kind": overall, "start_time": ev_time.isoformat(), "end_time": (ev_time + timedelta(minutes=10)).isoformat(), "duration_minutes": 10, "source":"wayback_snapshot"}
                    hist = state.get("history",[]) or []
                    hist.append(rec)
                    state["history"] = hist[-2000:]
                    save_state(state)
                    collected += 1
                    log(f"Wayback 发现事件 {overall} @ {ts}，已记录")
        log(f"Wayback 收集结束，共收集 {collected} 条事件")
    else:
        log(f"已有足够最近历史 (count={recent_count})，跳过 Wayback 收集")

    # 3) 检查历史是否足够并做预测/提醒（立即触发）
    hist_now = 0
    for h in state.get("history",[]) or []:
        try:
            st = datetime.fromisoformat(h["start_time"]); st = st.astimezone(TZ) if st.tzinfo else st.replace(tzinfo=timezone.utc).astimezone(TZ)
            if st >= now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS): hist_now += 1
        except:
            continue
    log(f"处理后最近历史数量: {hist_now}")
    if hist_now < MIN_HISTORY_EVENTS_FOR_PRED:
        send_telegram("⚠️ [DG Monitor] 替补执行完毕，但无法找到足够的最近 4 周历史数据来做可靠预测。脚本将继续收集历史并在未来触发。")
        save_state(state)
        return

    pred = predict_from_history(state)
    if not pred:
        send_telegram("⚠️ [DG Monitor] 替补执行完毕，但未能从历史中找出稳定放水 / 中上 模式（无可靠预测）。")
        save_state(state); return

    ps = pred["predicted_start"]; pe = pred["predicted_end"]
    now = now_tz(); lead = timedelta(minutes=PRED_LEAD_MINUTES)
    if (ps - lead) <= now <= pe:
        remaining = max(0, int((pe - now).total_seconds()//60))
        msg = f"🔔 [DG替補 - 歷史全市場] 根據最近 {HISTORY_LOOKBACK_DAYS} 天的全市場歷史模式偵測到可能的『{pred['kind']}』\n預測開始: {ps.strftime('%Y-%m-%d %H:%M:%S')}\n估計結束: {pe.strftime('%Y-%m-%d %H:%M:%S')}（約 {pred['avg_duration']} 分鐘）\n目前剩餘: 約 {remaining} 分鐘\n※ 替補來源: Wayback / 公共歷史資料（非即時實盤）。"
        # 避免重复
        hist = state.get("history",[]) or []
        duplicate=False
        for h in hist[-80:]:
            try:
                if h.get("kind")==pred["kind"]:
                    tm = datetime.fromisoformat(h["start_time"]).astimezone(TZ)
                    if abs((tm - ps).total_seconds()) < 60*5:
                        duplicate=True; break
            except:
                continue
        if not duplicate:
            send_telegram(msg)
            hist.append({"kind":pred["kind"], "start_time": ps.isoformat(), "end_time": pe.isoformat(), "duration_minutes": pred["avg_duration"], "source":"historical_predict"})
            state["history"] = hist[-2000:]; save_state(state)
            log("已发送替补预测提醒并记录")
        else:
            log("检测到近似历史记录，跳过重复提醒")
    else:
        log(f"预测下一次 {pred['kind']} 开始于 {ps.strftime('%Y-%m-%d %H:%M:%S')}，不在提醒窗口内")
    save_state(state)
    return

# ---------- entry ----------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        sys.exit(1)
