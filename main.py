# main.py  — DG 实盘监测（只在符合规则时发通知）
# 必须保持文件名 main.py（不要改）
# 环境变量:
#   TG_BOT_TOKEN (必须)
#   TG_CHAT_ID (必须)
#   MIN_POINTS_FOR_REAL (可选，默认 10)
#   COOLDOWN_MINUTES (可选，默认 10)
#   HISTORY_LOOKBACK_DAYS (可选，默认 28)
# 注意：本脚本**只会在**判定为 "放水时段（提高胜率）" 或 "中等胜率（中上）" 时发 Telegram，其他失败/替补/进入失败不发 Telegram（仅写日志和文件）。

import os, sys, json, time, random, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import cv2
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ------------- 配置 -------------
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]
TG_TOKEN_ENV = "TG_BOT_TOKEN"
TG_CHAT_ENV = "TG_CHAT_ID"

MIN_POINTS_FOR_REAL = int(os.environ.get("MIN_POINTS_FOR_REAL", "10"))  # <= 你的日志 9-11 -> 默认 10
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))
HISTORY_LOOKBACK_DAYS = int(os.environ.get("HISTORY_LOOKBACK_DAYS", "28"))
DILATE_KERNEL_SIZE = int(os.environ.get("DILATE_KERNEL_SIZE", "40"))
WAYBACK_MAX_SNAPSHOTS = int(os.environ.get("WAYBACK_MAX_SNAPSHOTS","40"))
WAYBACK_RATE_SLEEP = float(os.environ.get("WAYBACK_RATE_SLEEP","1.2"))

STATE_FILE = "state.json"
SUMMARY_FILE = "last_summary.json"
LAST_SCREENSHOT = "last_screenshot.png"
TZ = timezone(timedelta(hours=8))  # 马来西亚 UTC+8

# ---------- 工具 ----------
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

# ---------- 图像/颜色检测（简化版） ----------
def pil_from_bytes(b): return Image.open(BytesIO(b)).convert("RGB")
def cv_from_pil(p): return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def detect_color_points(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 红、蓝范围 (近似)
    lower_r,upper_r = np.array([0,90,60]), np.array([10,255,255])
    lower_r2,upper_r2 = np.array([160,90,60]), np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower_r, upper_r) | cv2.inRange(hsv, lower_r2, upper_r2)
    lowerb,upperb = np.array([85,60,40]), np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask_r|mask_b, cv2.MORPH_OPEN, kernel, iterations=1)
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

def cluster_points_to_boards(points, img_shape):
    h,w = img_shape[:2]
    if not points:
        cols=max(3,w//300); rows=max(2,h//200)
        rects=[]
        cw=w//cols; ch=h//rows
        for r in range(rows):
            for c in range(cols):
                rects.append((c*cw, r*ch, cw, ch))
        return rects
    mask = np.zeros((h,w), dtype=np.uint8)
    for x,y in points:
        if 0<=x<w and 0<=y<h: mask[y,x]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE,DILATE_KERNEL_SIZE))
    big = cv2.dilate(mask, kernel, iterations=1)
    num,labels,stats,_ = cv2.connectedComponentsWithStats(big, connectivity=8)
    rects=[]
    for i in range(1,num):
        x,y,w_,h_ = stats[i,0], stats[i,1], stats[i,2], stats[i,3]
        if w_<60 or h_<40: continue
        pad=8
        x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w-1,x+w_+pad); y1=min(h-1,y+h_+pad)
        rects.append((x0,y0,x1-x0,y1-y0))
    return rects

def analyze_board(bgr, rect):
    # 只大致返回一个 maxRun(连续同色最大长度) 与 total 点数
    x,y,w,h = rect
    crop = bgr[y:y+h, x:x+w]
    pts = detect_color_points(crop)
    total = len(pts)
    # 简单估计 maxRun：按 x 分列，再按 y 排序，压缩序列计算最长连续（非常粗糙）
    if total==0: return {"total":0,"maxRun":0,"category":"empty"}
    xs = [p[0] for p in pts]
    idx_sorted = sorted(range(len(xs)), key=lambda i: xs[i])
    flattened_colors = []
    for i in idx_sorted:
        flattened_colors.append('x')  # color info lost in simple detect; we only use counts/structure
    # as fallback, assume some runs if many points clustered
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
    # 简化：如果 longCount >= 3 -> 放水； 如果至少 3 张有多列连珠 + >=2 长龙 -> 中等升
    if longCount >= 3:
        return "放水时段（提高胜率）", longCount, superCount
    # 检查“多连/连珠” 简化：若有 >=3 板 total >=6 且 maxRun>=4 视为多连
    multi = sum(1 for b in board_infos if b['total']>=6 and b['maxRun']>=4)
    boards_with_long = sum(1 for b in board_infos if b['maxRun'] >= 8)
    if multi >= 3 and boards_with_long >= 2:
        return "中等胜率（中上）", boards_with_long, sum(1 for b in board_infos if b['category']=='super_long')
    # 若大部分板很空 -> 收割
    if board_infos and sum(1 for b in board_infos if b['total'] < 6) >= len(board_infos)*0.6:
        return "胜率调低 / 收割时段", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')
    return "胜率中等（平台收割中等时段）", sum(1 for b in board_infos if b['maxRun']>=8), sum(1 for b in board_infos if b['category']=='super_long')

# ---------- Playwright & 滑块操作 ----------
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
        # 截图辅助
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
                    for s in range(8):  # 多次尝试滑块与等待
                        got = try_solve_slider(page)
                        log(f"slider 尝试 {s+1} -> {got}")
                        time.sleep(0.8 + random.random())
                        ss = page.screenshot(full_page=True)
                        with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
                        img = pil_from_bytes(ss); bgr = cv_from_pil(img)
                        pts = detect_color_points(bgr)
                        log(f"检查 {s+1}: 点数 {len(pts)}")
                        if len(pts) >= MIN_POINTS_FOR_REAL:
                            context.close(); browser.close()
                            return ss
                    # 若未满足，继续到下一个 url
                except PWTimeout as e:
                    log(f"页面打开超时: {e}")
                except Exception as e:
                    log(f"与页面交互异常: {e}")
            try: context.close()
            except: pass
            try: browser.close()
            except: pass
            time.sleep(1.8 + random.random())
        log("未能进入实盘（多次尝试失败或点数不足）")
        return None

# ---------- Wayback (替补) ----------
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

# ---------- fallback 历史预测（仅用来记录/预测，不会主动发通知，除非预测结果“正好处于当前时间窗口”并且符合提醒规则） ----------
def fallback_with_history_and_maybe_alert(state):
    # 尝试 Wayback 查过去 28 天快照，收集事件到 state.history（静默收集）
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
            with open(LAST_SCREENSHOT,"wb") as f: f.write(ss)
            img = pil_from_bytes(ss); bgr = cv_from_pil(img)
            pts = detect_color_points(bgr)
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
                hist.append(rec); state["history"] = hist[-2000:]; collected += 1
    save_state(state)
    log(f"Wayback 替补收集结束，共收集 {collected} 条事件")

    # 基于 state.history 做最简单的时间窗口预测（若当前时间就在预测窗口内且预测为放水/中上 -> 发提醒）
    recent = []
    for ev in state.get("history",[]) or []:
        try:
            st = datetime.fromisoformat(ev["start_time"]).astimezone(TZ)
            if st >= now_tz() - timedelta(days=HISTORY_LOOKBACK_DAYS):
                recent.append(ev)
        except:
            continue
    if len(recent) < 3:
        log("替补历史数量不足，跳过预测提醒（静默）")
        return
    # 简单汇总：找出现次数最多的 kind 在某个小时段
    buckets = {}
    for ev in recent:
        st = datetime.fromisoformat(ev["start_time"]).astimezone(TZ)
        key = (ev["kind"], st.weekday(), st.hour, (st.minute//15)*15)
        buckets.setdefault(key, 0)
        buckets[key]+=1
    best = sorted(buckets.items(), key=lambda x:x[1], reverse=True)[:1]
    if not best:
        return
    (kind, wk, hr, mn), cnt = best[0]
    predicted_start = now_tz().replace(hour=hr, minute=mn, second=0, microsecond=0)
    if predicted_start < now_tz() - timedelta(minutes=1):
        predicted_start += timedelta(days=1)
    predicted_end = predicted_start + timedelta(minutes=10)
    # 只有当现在在预测窗口内，且预测kind属于我们要提醒的两类，才发提醒
    if predicted_start <= now_tz() <= predicted_end and kind in ("放水时段（提高胜率）","中等胜率（中上）"):
        state_local = load_state()
        last_alert = state_local.get("last_alert_time")
        if last_alert:
            try:
                last_dt = datetime.fromisoformat(last_alert)
            except:
                last_dt = None
        else:
            last_dt = None
        if last_dt and (now_tz() - last_dt).total_seconds() < COOLDOWN_MINUTES*60:
            log("预测窗口命中但在冷却期，跳过通知")
            return
        remaining = int((predicted_end - now_tz()).total_seconds() // 60)
        msg = f"🔔 [DG替補預測] 偵測到可能的「{kind}」\n預測開始: {predicted_start.strftime('%Y-%m-%d %H:%M:%S')}\n估計結束: {predicted_end.strftime('%Y-%m-%d %H:%M:%S')}（約 10 分鐘）\n目前剩餘: 約 {remaining} 分鐘\n※警告：此為基於歷史的替補預測（非即時實盤）。"
        ok,_ = send_telegram(msg)
        if ok:
            state_local["last_alert_time"] = now_tz().isoformat()
            state_local["active"] = True
            state_local["kind"] = kind
            state_local["start_time"] = predicted_start.isoformat()
            save_state(state_local)
    else:
        log("替補预测没有命中当前时间窗口或不是我们要提醒的种类（静默）")

# ---------- 主逻辑 ----------
def main():
    log("=== DG monitor run start ===")
    state = load_state()
    # 1) 尝试进入实盘并截图
    screenshot = None
    try:
        screenshot = capture_dg_page()
    except Exception as e:
        log(f"capture_dg_page 异常: {e}")
    # 2) 若得到截图则分析
    if screenshot:
        with open(LAST_SCREENSHOT,"wb") as f: f.write(screenshot)
        img = pil_from_bytes(screenshot)
        bgr = cv_from_pil(img)
        pts = detect_color_points(bgr)
        log(f"实时检测点数: {len(pts)} (阈值 {MIN_POINTS_FOR_REAL})")
        if len(pts) < MIN_POINTS_FOR_REAL:
            # 点数不足，则不发 Telegram（按你要求：没有符合状态不要通知）
            log("截图点数不足，视为未进入实盘 -> 执行替补历史收集（静默，不通知）")
            fallback_with_history_and_maybe_alert(state)
            return
        rects = cluster_points_to_boards(pts, bgr.shape)
        boards=[]
        for r in rects:
            boards.append(analyze_board(bgr, r))
        overall, longCount, superCount = classify_overall(boards)
        log(f"实时判定: {overall} (长龙/超: {longCount}/{superCount})")
        # 只有当判定为放水或中等胜率（中上）时才发通知
        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            # 去重/冷却：同一类告警不会在 COOLDOWN_MINUTES 内重复
            last_alert = state.get("last_alert_time")
            if last_alert:
                try:
                    last_dt = datetime.fromisoformat(last_alert)
                except:
                    last_dt = None
            else:
                last_dt = None
            if last_dt and (now_tz() - last_dt).total_seconds() < COOLDOWN_MINUTES*60:
                log("处于冷却期，跳过重复提醒（不会发 Telegram）")
            else:
                # 估计结束时间：用历史平均或固定 10 分钟
                est_min = 10
                est_end = (now_tz() + timedelta(minutes=est_min)).strftime("%Y-%m-%d %H:%M:%S")
                msg = f"🔔 [DG提示] {overall} 開始\n時間: {nowstr()}\n长龙/超龙 桌數: {longCount} (超龙:{superCount})\n估計結束: {est_end}（約 {est_min} 分鐘）"
                ok,_ = send_telegram(msg)
                if ok:
                    state["last_alert_time"] = now_tz().isoformat()
                    state["active"] = True
                    state["kind"] = overall
                    state["start_time"] = now_tz().isoformat()
                    save_state(state)
        else:
            log("判定不是放水或中等勝率（中上），不發通知（靜默）")
            # 若之前在 active 中且現在不在 -> 發結束通知（這是允許的通知）
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
        # 写 summary 以供审查
        with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
            json.dump({"ts":now_tz().isoformat(),"overall":overall,"longCount":longCount,"superCount":superCount}, f, ensure_ascii=False, indent=2)
        return
    else:
        # 没有截图 -> 静默进行替补（不发 Telegram），只有替补预测**命中当前时间窗口且预测为放水/中上**时才会发通知
        log("无法取得实盘截图（可能滑块或点数未满足），静默启动替补历史收集/预测（仅在预测命中时才通知）")
        fallback_with_history_and_maybe_alert(state)
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        sys.exit(1)
