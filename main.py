# -*- coding: utf-8 -*-
"""
DG 实盘监测脚本（用于 GitHub Actions）
作者：为用户整合（基于用户在对话中所有规则）
说明：
 - 每次 run 会尝试访问 DG 两个入口（https://dg18.co/wap/, https://dg18.co/）
 - 模拟点击 "Free/免费试玩"、模拟滚动安全条，截取页面
 - 用 OpenCV 检测红/蓝“珠子”，并把点按列/行重建为 baccarat 珠盘序列
 - 依据用户规则判断：放水时段 或 中等胜率（中上） => 发送 Telegram 提醒（一次启动通知 + 结束通知）
 - 事件开始时会估算结束时间（基于历史平均），并在结束时发送真实持续时间
 - 当事件 active 后，脚本会在 estimated_end_time 到来前**跳过频繁截图**以减少误判（GH Actions 仍每5分钟触发脚本，但脚本会按策略决定是否实际抓图）
"""
import os, sys, time, json, math, traceback
from datetime import datetime, timedelta, timezone
import requests, base64
from io import BytesIO
from pathlib import Path

# image & cv
import numpy as np
from PIL import Image
import cv2

# playwright
from playwright.sync_api import sync_playwright

# sklearn for fallback clustering
from sklearn.cluster import KMeans

# ---------- CONFIG (可在 Actions env/secrets 中覆盖) ----------
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID  = os.environ.get("TG_CHAT_ID", "").strip()

DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

# 判定阈值（可根据实际微调）
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))  # 放水至少满足桌数
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))             # 中等胜率需要至少龙桌数量
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))    # 通用冷却

# 文件与时区
STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))  # Malaysia UTC+08:00

# 运行调试开关
DEBUG_SAVE_IMAGE = False  # 若 True 会保存 last_screenshot.png 到仓库，便于离线调参（注意不要泄露）

# ---------- helper ----------
def log(msg):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置（TG_BOT_TOKEN/TG_CHAT_ID 为空），跳过发送。")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode":"HTML"}
    try:
        r = requests.post(url, data=payload, timeout=30)
        j = r.json()
        if j.get("ok"):
            log("Telegram 已发送")
            return True
        else:
            log(f"Telegram API 返回错误：{j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 错误：{e}")
        return False

# ---------- state ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        s = {"active": False, "kind": None, "start_time": None, "estimated_end": None, "last_seen": None, "history": []}
        return s
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"active": False, "kind": None, "start_time": None, "estimated_end": None, "last_seen": None, "history": []}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# ---------- image helpers ----------
def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_red_blue_points(bgr_img):
    """
    检测红/蓝点，返回 point list: [(x,y,'B'|'P'), ...]
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # red
    lower1 = np.array([0, 100, 80]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 80]); upper2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue
    lowerb = np.array([95, 60, 60]); upperb = np.array([135,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    # cleanup
    k = np.ones((3,3),np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)
    points=[]
    for mask,label in [(mask_r,'B'),(mask_b,'P')]:
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 8: continue
            M = cv2.moments(c)
            if M.get("m00",0)==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,label))
    return points, mask_r, mask_b

# ---------- cluster boards heuristic ----------
def cluster_boards(points, img_w, img_h):
    """
    基于点密度把页面聚成若干候选小桌区域 (x,y,w,h) 列表
    fallback: KMeans clustering
    """
    if not points:
        return []
    cell = max(64, int(min(img_w,img_h)/12))
    cols = math.ceil(img_w/cell); rows = math.ceil(img_h/cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, x//cell); cy = min(rows-1, y//cell)
        grid[cy][cx] += 1
    thr = 6
    hits=[]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] >= thr: hits.append((r,c))
    rects=[]
    if hits:
        for (r,c) in hits:
            x = c*cell; y = r*cell; w = cell; h = cell
            placed=False
            for i,(rx,ry,rw,rh) in enumerate(rects):
                if not (x > rx+rw+cell or x+w < rx-cell or y > ry+rh+cell or y+h < ry-cell):
                    nx = min(rx,x); ny = min(ry,y)
                    nw = max(rx+rw, x+w)-nx; nh = max(ry+rh, y+h)-ny
                    rects[i] = (nx,ny,nw,nh)
                    placed=True; break
            if not placed:
                rects.append((x,y,w,h))
        # expand slightly
        regs=[]
        for (x,y,w,h) in rects:
            nx = max(0,x-12); ny = max(0,y-12)
            nw = min(img_w-nx, w+24); nh = min(img_h-ny, h+24)
            regs.append((int(nx),int(ny),int(nw),int(nh)))
        return regs
    # fallback kmeans
    coords = np.array([[p[0],p[1]] for p in points])
    k = min(8, max(1, len(points)//8))
    if k<=1:
        return [(0,0,img_w,img_h)]
    try:
        km = KMeans(n_clusters=k, random_state=0).fit(coords)
        regs=[]
        for lab in range(k):
            pts = coords[km.labels_==lab]
            if pts.size==0: continue
            x0,y0 = pts.min(axis=0); x1,y1 = pts.max(axis=0)
            regs.append((int(max(0,x0-8)), int(max(0,y0-8)), int(min(img_w, x1-x0+16)), int(min(img_h,y1-y0+16))))
        return regs
    except Exception:
        return [(0,0,img_w,img_h)]

# ---------- analyze single board ----------
def analyze_board_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts, mr, mb = detect_red_blue_points(crop)
    if not pts:
        return {"total":0, "maxRun":0, "category":"empty", "runs":[], "cols_info":[]}
    # cluster points into columns by x coordinate
    pts_sorted = sorted(pts, key=lambda p: p[0])
    xs = [p[0] for p in pts_sorted]
    # group by gap threshold
    groups = []
    gap = max(10, w//40)
    for idx,p in enumerate(pts_sorted):
        if not groups:
            groups.append([p])
        else:
            last = groups[-1][-1]
            if p[0] - last[0] <= gap:
                groups[-1].append(p)
            else:
                groups.append([p])
    # for each column group, sort by y, produce color sequence
    cols_seq = []
    cols_info = []
    for col in groups:
        col_sorted = sorted(col, key=lambda z: z[1])
        seq = [z[2] for z in col_sorted]
        cols_seq.append(seq)
        # find max contiguous run length within column (top->bottom)
        maxrun = 0
        cur = None; curc=0
        for c in seq:
            if c==cur:
                curc+=1
            else:
                if curc>maxrun: maxrun=curc
                cur = c; curc=1
        if curc>maxrun: maxrun=curc
        cols_info.append({"col_len":len(seq), "max_run_in_col":maxrun})
    # flatten reading columns left->right, top->bottom
    flattened=[]
    max_h = max((len(s) for s in cols_seq), default=0)
    for r in range(max_h):
        for c in cols_seq:
            if r < len(c): flattened.append(c[r])
    # compute runs across flattened
    runs=[]
    if flattened:
        cur = {"color":flattened[0], "len":1}
        for i in range(1,len(flattened)):
            if flattened[i]==cur["color"]:
                cur["len"] += 1
            else:
                runs.append(cur); cur={"color":flattened[i],"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    category = "other"
    if maxRun >= 10: category="super_long"
    elif maxRun >= 8: category="long"
    elif maxRun >= 4: category="longish"
    # detect 多连连续3排（heuristic）：
    # 若在 cols_info 中，存在 >=3 个连续列（邻近列）其 max_run_in_col >=4 -> 视为 连续3排多连/连珠
    consecutive = 0; max_consecutive=0
    for info in cols_info:
        if info["max_run_in_col"] >= 4:
            consecutive +=1
        else:
            max_consecutive = max(max_consecutive, consecutive)
            consecutive = 0
    max_consecutive = max(max_consecutive, consecutive)
    has_3row_mult = (max_consecutive >= 3)
    return {"total":len(flattened), "maxRun": maxRun, "category": category, "runs": runs, "cols_info": cols_info, "has_3row_mult": has_3row_mult}

# ---------- classify overall ----------
def classify_boards(board_stats):
    longCount = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_stats if b['category']=='super_long')
    longishCount = sum(1 for b in board_stats if b['category']=='longish')
    # check how many boards have has_3row_mult
    multi3_count = sum(1 for b in board_stats if b.get('has_3row_mult'))
    n = len(board_stats)
    # 放水: 至少 MIN_BOARDS_FOR_PAW 张处于 长龙/超长龙（>= MIN_BOARDS_FOR_PAW）
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount, multi3_count
    # 中等胜率（中上）：存在 >=3 张桌呈连续3排多连/连珠 + 至少 MID_LONG_REQ 张龙/超龙 (可同桌)
    if multi3_count >= 3 and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount, multi3_count
    # 收割：多数桌稀疏
    sparse = sum(1 for b in board_stats if b["total"] < 6)
    if n>0 and sparse >= n*0.6:
        return "胜率调低 / 收割时段", longCount, superCount, multi3_count
    return "胜率中等（平台收割中等时段）", longCount, superCount, multi3_count

# ---------- capture DG screenshot with playwright ----------
def capture_dg_screenshot(playwright, url, max_wait=35):
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
    try:
        ctx = browser.new_context(viewport={"width":1366,"height":768})
        page = ctx.new_page()
        log(f"打开 {url}")
        page.goto(url, timeout=30000)
        time.sleep(2)
        # 尝试点击多种 Free / 免费试玩 文本
        tried = False
        for txt in ["Free", "免费试玩", "免费", "Play Free", "试玩", "FREE"]:
            try:
                locator = page.locator(f"text={txt}")
                if locator.count() > 0:
                    locator.first.click(timeout=4000)
                    log(f"点击文字: {txt}")
                    tried = True
                    break
            except Exception:
                continue
        time.sleep(2)
        # 滚动以触发可能的安全滑动条
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.6)
            page.evaluate("window.scrollTo(0, 0);")
            time.sleep(0.6)
        except Exception:
            pass
        time.sleep(3)
        # 再尝试等待某些关键节点？（由于页面差异，这里做宽松等待）
        try:
            shot = page.screenshot(full_page=True)
            log("截图完成")
            return shot
        except Exception as e:
            log(f"截图失败：{e}")
            return None
    finally:
        try:
            browser.close()
        except:
            pass

# ---------- main ----------
def main():
    state = load_state()
    log("==== 新一轮检测开始 ====")
    # If already active and an estimated_end exists and current time < estimated_end:
    #   --> 为减少误判/避免频繁截图，短路本次检测（但仍保留运行以满足 GitHub Actions 调度）。
    now = datetime.now(TZ)
    if state.get("active"):
        est = state.get("estimated_end")
        if est:
            est_dt = datetime.fromisoformat(est)
            if now < est_dt:
                log(f"当前已有活动（{state.get('kind')}），并在预计结束时间 {est_dt.strftime('%Y-%m-%d %H:%M')} 之前。跳过本次重截图检测。")
                # 仍保存 last_seen
                state["last_seen"] = now.isoformat()
                save_state(state)
                return
    # else 执行正常抓取与识别
    screenshot = None
    with sync_playwright() as p:
        for url in DG_LINKS:
            try:
                screenshot = capture_dg_screenshot(p, url)
                if screenshot:
                    break
            except Exception as e:
                log(f"访问 {url} 异常：{e}")
                continue
    if not screenshot:
        log("无法获取到任何截图，本次结束。")
        # optionally send an error message to Telegram on repeated failures? （略）
        save_state(state)
        return
    pil = pil_from_bytes(screenshot)
    bgr = cv_from_pil(pil)
    h,w = bgr.shape[:2]
    if DEBUG_SAVE_IMAGE:
        cv2.imwrite("last_screenshot.png", bgr)
    pts, mr, mb = detect_red_blue_points(bgr)
    log(f"检测到彩点总数: {len(pts)}")
    if len(pts) == 0:
        log("页面可能未加载正确或颜色阈值不匹配，结束本次。")
        save_state(state); return
    regions = cluster_boards(pts, w, h)
    log(f"聚类出候选桌子：{len(regions)}")
    board_stats=[]
    for reg in regions:
        st = analyze_board_region(bgr, reg)
        board_stats.append(st)
    overall, longCount, superCount, multi3_count = classify_boards(board_stats)
    log(f"判定结果 => {overall} (长/超长龙桌数={longCount}, 超长龙={superCount}, 连续3排多连桌数={multi3_count})")
    # update summary
    summary = {"ts": now.isoformat(), "overall":overall, "longCount":longCount, "superCount":superCount, "multi3_count":multi3_count, "boards_count":len(board_stats)}
    with open(LAST_SUMMARY,"w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # state transitions
    was_active = state.get("active", False)
    was_kind = state.get("kind", None)
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
    if is_active_now and not was_active:
        # start event
        state["active"] = True
        state["kind"] = overall
        state["start_time"] = now.isoformat()
        # estimate duration based on history average (use median of history durations if available)
        hist = state.get("history", [])
        durations = [h.get("duration_minutes",0) for h in hist if h.get("duration_minutes",0)>0]
        est_minutes = 10
        if durations:
            est_minutes = int(round(sum(durations)/len(durations)))
            # guard
            est_minutes = max(5, min(est_minutes, 120))
        est_end = now + timedelta(minutes=est_minutes)
        state["estimated_end"] = est_end.isoformat()
        state["last_seen"] = now.isoformat()
        save_state(state)
        # Send Telegram start notification (with emoji)
        est_end_str = est_end.astimezone(TZ).strftime("%Y-%m-%d %H:%M")
        emoji = "🔔"
        msg = f"{emoji} <b>DG 提醒 — {overall} 開始</b>\n時間: {now.astimezone(TZ).strftime('%Y-%m-%d %H:%M')}\n長/超长龙桌數={longCount}, 超长龙={superCount}, 連續3排多連桌數={multi3_count}\n估計結束時間（基於歷史/近似）: {est_end_str}（約 {est_minutes} 分鐘）"
        send_telegram(msg)
        log("事件已標記為 active 並發送開始通知。")
    elif is_active_now and was_active:
        # still active: update last_seen and possibly adjust estimated_end if very long
        state["last_seen"] = now.isoformat()
        save_state(state)
        log("事件仍持續中，已更新 last_seen。")
    elif (not is_active_now) and was_active:
        # event ended -> compute actual duration and push to history
        start_iso = state.get("start_time")
        if start_iso:
            start_dt = datetime.fromisoformat(start_iso)
            duration_min = int(round((now - start_dt).total_seconds()/60.0))
        else:
            duration_min = 0
        hist = state.get("history", [])
        hist.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": now.isoformat(), "duration_minutes": duration_min})
        # keep last N
        state = {"active": False, "kind": None, "start_time": None, "estimated_end": None, "last_seen": None, "history": hist[-120:]}
        save_state(state)
        emoji="✅"
        msg = f"{emoji} <b>DG 提醒 — {was_kind} 已结束</b>\n開始: {start_iso}\n結束: {now.isoformat()}\n實際持續: {duration_min} 分鐘"
        send_telegram(msg)
        log("事件结束，已发送结束通知并记录历史。")
    else:
        # not active, not previously active
        state["last_seen"] = now.isoformat()
        save_state(state)
        log("当前不属于放水/中上，不发送提醒。")
    log("==== 本次检测结束 ====")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("脚本异常: " + str(e))
        traceback.print_exc()
        # 发错警告到 Telegram（可选）
        try:
            send_telegram("⚠️ <b>DG 监测脚本发生异常</b>\n" + str(e))
        except:
            pass
        raise
