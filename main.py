# main.py  — DG 实盘监测（增强：立即尝试次级替补格子检测）
# 请保持文件名 main.py（不要改）
# 环境变量:
#   TG_BOT_TOKEN (必须)
#   TG_CHAT_ID (必须)
#   MIN_POINTS_FOR_REAL (默认 10)
#   MIN_BOARDS_FOR_REAL 或 MIN_BOARDS_FOR_PAW (默认 8)
#   COOLDOWN_MINUTES (默认 10)
#   HISTORY_LOOKBACK_DAYS (默认 28)

import os, sys, json, time, random, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import cv2
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------- 配置 ----------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN_ENV = "TG_BOT_TOKEN"
TG_CHAT_ENV = "TG_CHAT_ID"

MIN_POINTS_FOR_REAL = int(os.environ.get("MIN_POINTS_FOR_REAL", "10"))
# 支持两种环境变量名（向后兼容）
MIN_BOARDS_FOR_REAL = int(os.environ.get("MIN_BOARDS_FOR_REAL", os.environ.get("MIN_BOARDS_FOR_PAW", "8")))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))
HISTORY_LOOKBACK_DAYS = int(os.environ.get("HISTORY_LOOKBACK_DAYS", "28"))

STATE_FILE = "state.json"
SUMMARY_FILE = "last_summary.json"
LAST_SCREENSHOT = "last_screenshot.png"
TZ = timezone(timedelta(hours=8))

# ---------- 辅助 ----------
def now_tz(): return datetime.now(TZ)
def nowstr(): return now_tz().strftime("%Y-%m-%d %H:%M:%S")
def log(s): print(f"[{nowstr()}] {s}", flush=True)

def send_telegram(text, max_retries=3):
    token = os.environ.get(TG_TOKEN_ENV, "").strip()
    chat = os.environ.get(TG_CHAT_ENV, "").strip()
    if not token or not chat:
        log("Telegram 未配置，跳过发送")
        return False, "no_token"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat, "text": text}
    for _ in range(max_retries):
        try:
            r = requests.post(url, data=data, timeout=12)
            j = r.json()
            if j.get("ok"):
                log("Telegram 发送成功")
                return True, j
            else:
                log(f"Telegram 返回非 ok: {j}")
        except Exception as e:
            log(f"Telegram 发送异常: {e}")
        time.sleep(1 + random.random())
    log("Telegram 最终发送失败")
    return False, "failed"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active":False,"kind":None,"start_time":None,"last_alert_time":None,"history":[]}
    try:
        with open(STATE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"active":False,"kind":None,"start_time":None,"last_alert_time":None,"history":[]}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(s,f,ensure_ascii=False,indent=2)

# ---------- 图像：颜色点检测（原） ----------
def pil_from_bytes(b): return Image.open(BytesIO(b)).convert("RGB")
def cv_from_pil(p): return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def detect_color_points(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_r,upper_r = np.array([0,90,60]), np.array([10,255,255])
    lower_r2,upper_r2 = np.array([160,90,60]), np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower_r, upper_r) | cv2.inRange(hsv, lower_r2, upper_r2)
    lowerb,upperb = np.array([85,60,40]), np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    mask = cv2.morphologyEx(mask_r|mask_b, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts=[]
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 6: continue
        M = cv2.moments(cnt)
        if M.get("m00",0)==0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        pts.append((cx,cy))
    return pts

# ---------- 主格子检测（原） ----------
def detect_rectangular_boards(bgr):
    img = bgr.copy()
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    if w > 1400:
        scale = 1280.0 / w
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    edges = cv2.dilate(edges, kernel, iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 500: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x,y,w_,h_ = cv2.boundingRect(approx)
            ar = w_ / float(h_) if h_>0 else 0
            if h_ < 30 or w_ < 30: continue
            if ar < 0.3 or ar > 4.0: continue
            if w_ > gray.shape[1]*0.9 and h_ > gray.shape[0]*0.9: continue
            if scale != 1.0:
                x = int(x/scale); y = int(y/scale); w_ = int(w_/scale); h_ = int(h_/scale)
            rects.append((x,y,w_,h_))
    # 合并近邻
    merged = []
    for r in rects:
        rx,ry,rw,rh = r
        merged_flag=False
        for i,(mx,my,mw,mh) in enumerate(merged):
            if not (rx > mx+mw or mx > rx+rw or ry > my+mh or my > ry+rh):
                nx = min(rx,mx); ny = min(ry,my)
                nx2 = max(rx+rw, mx+mw); ny2 = max(ry+rh, my+mh)
                merged[i] = (nx, ny, nx2-nx, ny2-ny)
                merged_flag=True
                break
        if not merged_flag:
            merged.append(r)
    boards_count = len(merged)
    return merged, boards_count

# ---------- 次级（替补）格子检测：更宽松 + 网格/线检测（新增） ----------
def detect_rectangular_boards_secondary(bgr):
    """
    更宽松的替补检测：
    - 使用自适应阈值 + 较小最小面积
    - 同时尝试基于水平/垂直线的网格检测（morph close）
    - 合并并去重后返回矩形和数量
    """
    img = bgr.copy()
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 缩小加速
    scale = 1.0
    if w > 1600:
        scale = 1280.0 / w
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
    # 自适应阈值，增强线条
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,6)
    # 用形态学找横/竖线（网格）
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(gray.shape[1]//20)),1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, int(gray.shape[0]//20))))
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    grid = cv2.add(horiz, vert)
    # 膨胀以连通格子轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    grid = cv2.dilate(grid, kernel, iterations=2)
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 200: continue   # 次级阈值更小
        x,y,w_,h_ = cv2.boundingRect(cnt)
        if w_ < 25 or h_ < 25: continue
        ar = w_/float(h_) if h_>0 else 0
        if ar < 0.25 or ar > 5.0: continue
        # 恢复原始尺度
        if scale != 1.0:
            x = int(x/scale); y = int(y/scale); w_ = int(w_/scale); h_ = int(h_/scale)
        rects.append((x,y,w_,h_))
    # 再融合近邻
    merged = []
    for r in rects:
        rx,ry,rw,rh = r
        merged_flag=False
        for i,(mx,my,mw,mh) in enumerate(merged):
            if not (rx > mx+mw or mx > rx+rw or ry > my+mh or my > ry+rh):
                nx = min(rx,mx); ny = min(ry,my)
                nx2 = max(rx+rw, mx+mw); ny2 = max(ry+rh, my+mh)
                merged[i] = (nx, ny, nx2-nx, ny2-ny)
                merged_flag=True
                break
        if not merged_flag:
            merged.append(r)
    boards_count = len(merged)
    return merged, boards_count

# ---------- 之前的 board 分析（保留） ----------
def analyze_board(bgr, rect):
    x,y,w,h = rect
    crop = bgr[y:y+h, x:x+w]
    pts = detect_color_points(crop)
    total = len(pts)
    if total==0: return {"total":0,"maxRun":0,"category":"empty"}
    maxRun = 1
    if total>=10: maxRun = 8
    elif total>=6: maxRun = 4
    elif total>=3: maxRun = 2
    cat='other'
    if maxRun>=10: cat='super_long'
    elif maxRun>=8: cat='long'
    elif maxRun>=4: cat='longish'
    elif maxRun==1: cat='single'
    return {"total":total,"maxRun":maxRun,"category":cat}

def classify_overall(board_infos):
    longCount = sum(1 for b in board_infos if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_infos if b['category']=='super_long')
    if longCount >= 3:
        return "放水時段（提高勝率）", longCount, superCount
    multi = sum(1 for b in board_infos if b['total']>=6 and b['maxRun']>=4)
    boards_with_long = sum(1 for b in board_infos if b['maxRun'] >= 8)
    if multi >= 3 and boards_with_long >= 2:
        return "中等勝率（中上）", boards_with_long, sum(1 for b in board_infos if b['category']=='super_long')
    if board_infos and sum(1 for b in board_infos if b['total'] < 6) >= len(board_infos)*0.6:
        return "勝率調低 / 收割時段", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')
    return "勝率中等（平台收割中等時段）", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')

# ---------- Playwright & 滑块（保留） ----------
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
                        time.sleep(0.9)
                        return True
            except:
                continue
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
        viewports = [(390,844),(1280,900),(1366,768)]
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
                    time.sleep(1.0 + random.random())
                    clicked=False
                    for txt in ["Free","免费试玩","免费","Play Free","试玩","Free Play","免费体验"]:
                        try:
                            loc = page.locator(f"text={txt}")
                            if loc.count()>0:
                                loc.first.click(timeout=3000); clicked=True; log(f"点击文本 {txt}"); break
                        except:
                            continue
                    time.sleep(0.8 + random.random())
                    for s in range(8):
                        got = try_solve_slider(page)
                        log(f"slider 尝试 {s+1} -> {got}")
                        time.sleep(0.8 + random.random())
                        ss = page.screenshot(full_page=True)
                        with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
                        img = pil_from_bytes(ss); bgr = cv_from_pil(img)
                        pts = detect_color_points(bgr)
                        rects, boards_count = detect_rectangular_boards(bgr)
                        log(f"检查 {s+1}: 点数 {len(pts)}; 检测格子数 {boards_count}")
                        # 首次判断：颜色点或主格子检测满足阈值
                        if len(pts) >= MIN_POINTS_FOR_REAL or boards_count >= MIN_BOARDS_FOR_REAL:
                            context.close(); browser.close()
                            return ss
                        # 如果首次未满足，**立即**尝试次级替补检测（对同一张截图）
                        sec_rects, sec_boards_count = detect_rectangular_boards_secondary(bgr)
                        log(f"次级替補检测: 格子数 {sec_boards_count}")
                        if sec_boards_count >= MIN_BOARDS_FOR_REAL:
                            log("次級替補满足格子阈值 -> 视为进入实盘")
                            # 优先返回原始截图（供后续处理）
                            context.close(); browser.close()
                            return ss
                        # 否则继续尝试滑块/后续循环
                    # 否则尝试下一个 url
                except PWTimeout as e:
                    log(f"页面打开超时: {e}")
                except Exception as e:
                    log(f"与页面交互异常: {e}")
            try: context.close()
            except: pass
            try: browser.close()
            except: pass
            time.sleep(1.8 + random.random())
        log("未能进入实盘（多次尝试失败或阈值未满足）")
        return None

# ---------- 替补（Wayback 等） 保留 ---------- 
def get_wayback_snapshots(url, from_date=None, to_date=None, limit=40):
    base = "http://web.archive.org/cdx/search/cdx"
    params = {"url":url, "output":"json", "filter":"statuscode:200", "limit":str(limit)}
    if from_date: params["from"]=from_date
    if to_date: params["to"]=to_date
    try:
        r = requests.get(base, params=params, timeout=12)
        if r.status_code!=200: return []
        j = r.json()
        if not j or len(j)<2: return []
        rows = j[1:]
        tss = [row[1] for row in rows if len(row)>1]
        return tss
    except:
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

def fallback_with_history_and_maybe_alert(state):
    from_date = (now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    to_date = now_tz().strftime("%Y%m%d")
    collected = 0
    for base in DG_LINKS:
        tss = get_wayback_snapshots(base, from_date=from_date, to_date=to_date, limit=40)
        if not tss:
            log(f"Wayback 未发现 {base} 的快照（过去 {HISTORY_LOOKBACK_DAYS} 天）")
            continue
        for ts in tss[:40]:
            time.sleep(1.2)
            ss = fetch_wayback_snapshot_and_screenshot(ts, base)
            if not ss: continue
            with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
            img = pil_from_bytes(ss); bgr = cv_from_pil(img)
            pts = detect_color_points(bgr)
            rects, boards_count = detect_rectangular_boards(bgr)
            if len(pts) < MIN_POINTS_FOR_REAL and boards_count < MIN_BOARDS_FOR_REAL:
                # 立即尝试次級替補
                sec_rects, sec_boards_count = detect_rectangular_boards_secondary(bgr)
                log(f"Wayback 次級替補检测: 格子数 {sec_boards_count}")
                if sec_boards_count >= MIN_BOARDS_FOR_REAL:
                    rects = sec_rects; boards_count = sec_boards_count
                else:
                    continue
            board_rects = rects if rects else []
            boards=[]
            for r in board_rects: boards.append(analyze_board(bgr, r))
            overall, longCount, superCount = classify_overall(boards)
            if overall in ("放水時段（提高勝率）","中等勝率（中上）"):
                try:
                    ev_time = datetime.strptime(ts[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).astimezone(TZ)
                except:
                    ev_time = now_tz()
                rec = {"kind": overall, "start_time": ev_time.isoformat(), "end_time": (ev_time + timedelta(minutes=10)).isoformat(), "duration_minutes": 10, "source":"wayback_snapshot"}
                hist = state.get("history",[]) or []
                hist.append(rec); state["history"] = hist[-2000:]; collected += 1
    save_state(state)
    log(f"Wayback 替補收集结束，共收集 {collected} 条事件")

# ---------- 主逻辑 ----------
def main():
    log("=== DG monitor run start ===")
    state = load_state()
    screenshot = None
    try:
        screenshot = capture_dg_page()
    except Exception as e:
        log(f"capture_dg_page 异常: {e}")
    if screenshot:
        with open(LAST_SCREENSHOT,"wb") as f: f.write(screenshot)
        img = pil_from_bytes(screenshot); bgr = cv_from_pil(img)
        pts = detect_color_points(bgr)
        rects, boards_count = detect_rectangular_boards(bgr)
        log(f"实时检测点数: {len(pts)} (阈值 {MIN_POINTS_FOR_REAL}); 检测到格子数: {boards_count} (阈值 {MIN_BOARDS_FOR_REAL})")
        entered = (len(pts) >= MIN_POINTS_FOR_REAL) or (boards_count >= MIN_BOARDS_FOR_REAL)
        # 如果主检测失败，这里再次尝试次级替补（以防 capture_dg_page 未返回因为次级触发）
        if not entered:
            sec_rects, sec_boards_count = detect_rectangular_boards_secondary(bgr)
            log(f"实时次级替補检测: 格子数 {sec_boards_count} (阈值 {MIN_BOARDS_FOR_REAL})")
            if sec_boards_count >= MIN_BOARDS_FOR_REAL:
                log("次级替补满足 -> 视为进入实盘")
                rects = sec_rects; boards_count = sec_boards_count
                entered = True
        if not entered:
            log("未达到进入实盘的任一阈值 -> 静默替补历史收集/预测")
            fallback_with_history_and_maybe_alert(state)
            return
        # 进入实盘 -> 使用矩形（rects）作为 boards（若 rects 为空则基于颜色点聚类）
        board_rects = rects if rects else []
        boards=[]
        for r in board_rects:
            boards.append(analyze_board(bgr, r))
        if not boards:
            h,w = bgr.shape[:2]
            cols = max(3, w//320)
            rows = max(2, h//200)
            cw = w//cols; ch = h//rows
            for r in range(rows):
                for c in range(cols):
                    rect = (c*cw, r*ch, cw, ch)
                    boards.append(analyze_board(bgr, rect))
        overall, longCount, superCount = classify_overall(boards)
        log(f"实时判定: {overall} (长龙/超: {longCount}/{superCount})")
        if overall in ("放水時段（提高勝率）","中等勝率（中上）"):
            last_alert = state.get("last_alert_time")
            if last_alert:
                try:
                    last_dt = datetime.fromisoformat(last_alert)
                except:
                    last_dt = None
            else:
                last_dt = None
            if last_dt and (now_tz() - last_dt).total_seconds() < COOLDOWN_MINUTES*60:
                log("处于冷却期，跳过重复提醒（不发 Telegram）")
            else:
                est_min = 10
                est_end = (now_tz() + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
                msg = f"🔔 [DG提示] {overall} 開始\n時間: {nowstr()}\n檢測到格子數: {len(board_rects)}；长龙/超龙 桌數: {longCount} (超龙:{superCount})\n估計結束: {est_end}（約 {est_min} 分鐘）"
                ok,_ = send_telegram(msg)
                if ok:
                    state["last_alert_time"] = now_tz().isoformat()
                    state["active"] = True
                    state["kind"] = overall
                    state["start_time"] = now_tz().isoformat()
                    save_state(state)
        else:
            log("判定不是放水或中等勝率（中上），不發通知（靜默）")
            if state.get("active"):
                try:
                    start_time = datetime.fromisoformat(state.get("start_time")).astimezone(TZ)
                except:
                    start_time = now_tz()
                end_time = now_tz()
                dur = round((end_time - start_time).total_seconds() / 60.0)
                if dur >= 1:
                    msg = f"✅ [DG結束] {state.get('kind')} 已結束\n開始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n結束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n實際持續: {dur} 分鐘"
                    send_telegram(msg)
                state["active"]=False; state["kind"]=None; state["start_time"]=None
                save_state(state)
        with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
            json.dump({"ts":now_tz().isoformat(),"pts_count":len(pts),"boards_count":boards_count,"overall":overall,"longCount":longCount,"superCount":superCount}, f, ensure_ascii=False, indent=2)
        return
    else:
        log("无法取得实盘截图（静默替补历史收集/预测）")
        fallback_with_history_and_maybe_alert(state)
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        sys.exit(1)
