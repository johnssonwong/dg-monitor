# -*- coding: utf-8 -*-
"""
DG 实盘监测脚本（GitHub Actions 专用）
满足需求：
- 进入 https://dg18.co/wap/ 或 https://dg18.co/ ，点击“Free/免费试玩”，通过常见“滚动安全条/滑块”
- 进入实盘界面截图后，用 OpenCV 识别红/蓝珠，按你的规则计算：
  * 长连 ≥4
  * 多连/连珠：相邻列的“同色竖向连 ≥4”，连续列数≥2 为“多连”；≥3 为“连续3排连珠”
  * 长龙 ≥8，超长龙 ≥10
  * 单跳 = 1，双跳 = 2~3
  * 断连开单：连之后断开且连续2列单跳
- “放水时段（提高胜率）”与“中等胜率（中上）”才发 Telegram 提醒，结束时自动发“已结束，共持续X分钟”
- 提醒后“暂停检测”到预计结束时间，再恢复（满足你的“提醒后到时再查”要求）
- 进程内每 5 分钟精确循环，结合 workflow，尽量做到 24/7 几乎不间断
"""

import os, json, time, math, traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests
import numpy as np
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ========== 可配 + 你的默认值（也支持从环境变量注入） ==========
TZ = timezone(timedelta(hours=8))  # 马来西亚
DG_LINKS = ["https://dg18.co/wap/", "https://dg18.co/"]

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8").strip()
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "485427847").strip()

# 判定参数（可按需要微调）
MIN_TABLES_LONG_FOR_POUR = int(os.environ.get("MIN_TABLES_LONG_FOR_POUR", "3"))  # 放水：至少≥3 桌 长龙/超长龙（或满足“1超 + 2长”）
IN_LOOP_MINUTES          = int(os.environ.get("IN_LOOP_MINUTES", "350"))        # 单次 Actions 任务跑多长（分钟），<= 355（6小时上限留余量）
DETECT_INTERVAL_SECONDS  = int(os.environ.get("DETECT_INTERVAL_SECONDS", "300"))# 每次检测间隔（5分钟）
SAFE_MAX_SLEEP_MIN       = int(os.environ.get("SAFE_MAX_SLEEP_MIN", "90"))      # 提醒后最长静默等待（上限，防止极端估计过长）

# 估计默认（首次无历史）
DEFAULT_EST_MIN_POUR     = int(os.environ.get("DEFAULT_EST_MIN_POUR", "20"))    # 放水默认估计 20 分钟
DEFAULT_EST_MIN_MIDUP    = int(os.environ.get("DEFAULT_EST_MIN_MIDUP", "12"))   # 中上默认估计 12 分钟

STATE_FILE = "state.json"   # 事件状态
DEBUG_LAST = "last_run_summary.json"

# ========== 工具 ==========
def now():
    return datetime.now(TZ)

def ts():
    return now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("⚠️ 未配置 Telegram，跳过发送")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": TG_CHAT_ID, "text": text}, timeout=20)
        ok = r.json().get("ok")
        if ok: log("Telegram 消息已发送")
        else:  log(f"Telegram 返回异常：{r.text}")
        return bool(ok)
    except Exception as e:
        log(f"发送 Telegram 失败：{e}")
        return False

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active": False, "kind": None, "start": None, "expected_end": None, "history": []}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# ========== 网页自动化 ==========
def try_click_text(page, texts):
    for t in texts:
        try:
            el = page.locator(f"text={t}")
            if el.count() > 0:
                el.first.click(timeout=3000)
                log(f"点击按钮文字：{t}")
                return True
        except Exception:
            pass
    return False

def solve_common_slider(page):
    """
    针对常见滑块 / 安全条的多方案尝试：
    - 查找 role=slider
    - 常见 class 名含 slider/drag/handler 的元素
    - geetest/极验风格的滑块
    """
    ok = False
    # 1) 尝试 aria/role=slider
    try:
        sl = page.locator("[role='slider']")
        if sl.count() > 0:
            box = sl.first.bounding_box()
            if box:
                x = box["x"] + 5
                y = box["y"] + box["height"]/2
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x + box["width"] + 200, y, steps=20)
                page.mouse.up()
                time.sleep(1.5)
                ok = True
    except Exception:
        pass

    # 2) 常见 class 名称
    if not ok:
        try:
            cand = page.locator("css=[class*='slider'], [class*='drag'], [class*='handler']")
            if cand.count() > 0:
                box = cand.first.bounding_box()
                if box:
                    x = box["x"] + 5
                    y = box["y"] + box["height"]/2
                    page.mouse.move(x, y)
                    page.mouse.down()
                    page.mouse.move(x + box["width"] + 220, y, steps=25)
                    page.mouse.up()
                    time.sleep(1.5)
                    ok = True
        except Exception:
            pass

    # 3) geetest 常见结构
    if not ok:
        try:
            # 极验通常在 iframe 内
            frames = page.frames
            for f in frames:
                try:
                    btn = f.locator("css=.geetest_slider_button")
                    if btn.count() > 0:
                        box = btn.first.bounding_box()
                        if box:
                            x = box["x"] + 5
                            y = box["y"] + box["height"]/2
                            page.mouse.move(x, y)
                            page.mouse.down()
                            page.mouse.move(x + 300, y, steps=30)
                            page.mouse.up()
                            time.sleep(1.5)
                            ok = True
                            break
                except Exception:
                    continue
        except Exception:
            pass

    # 4) 兜底：滚动页面触发
    try:
        page.mouse.wheel(0, 1200)
        time.sleep(0.8)
        page.mouse.wheel(0, -1200)
        time.sleep(0.5)
    except Exception:
        pass

    return ok

def enter_dg_and_screenshot(play):
    """
    打开 DG -> 点击 Free/免费试玩 -> 通过安全条 -> 截图整个实盘页面
    返回 PIL.Image （或 None）
    """
    browser = play.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
    try:
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        for url in DG_LINKS:
            try:
                log(f"打开：{url}")
                page.goto(url, timeout=45000)
                time.sleep(2)

                # 点击按钮
                clicked = try_click_text(page, ["Free", "免费试玩", "免费", "试玩", "Play Free"])
                time.sleep(2)

                # 处理弹窗/新页签
                if len(context.pages) > 1:
                    page = context.pages[-1]
                    log("切到新弹出页面")

                # 尝试滑块/安全条
                solve_common_slider(page)
                time.sleep(2)

                # 等待实盘界面加载（尝试寻找常见元素；如果没有，依然截图）
                try:
                    page.wait_for_load_state("networkidle", timeout=20000)
                except PWTimeout:
                    pass

                # 再尝试一些点击（有些站进入后还有一次“进入/同意”）
                try_click_text(page, ["进入", "同意", "开始", "Enter", "I Agree"])
                time.sleep(2)

                # 截图整页
                png = page.screenshot(full_page=True)
                img = Image.open(BytesIO(png)).convert("RGB")

                try:
                    context.close()
                except Exception:
                    pass
                browser.close()
                log("已截图")
                return img

            except Exception as e:
                log(f"进入 {url} 失败：{e}")
                continue

        try:
            context.close()
        except Exception:
            pass
        browser.close()
        return None
    except Exception as e:
        log(f"浏览器异常：{e}")
        try:
            browser.close()
        except Exception:
            pass
        return None

# ========== 图像识别 ==========
def pil_to_bgr(im: Image.Image):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def detect_red_blue_points(bgr):
    """
    用 HSV 阈值找红（庄）蓝（闲）珠点，返回 [(x,y,'B'|'P'), ...]
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 红色（两段）
    mask_r = cv2.inRange(hsv, np.array([0,100,90]), np.array([10,255,255])) \
           | cv2.inRange(hsv, np.array([160,100,90]), np.array([179,255,255]))
    # 蓝色
    mask_b = cv2.inRange(hsv, np.array([95,80,50]), np.array([140,255,255]))

    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)

    points = []
    for m, label in [(mask_r, 'B'), (mask_b, 'P')]:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 12:  # 过滤噪点（可调）
                continue
            M = cv2.moments(c)
            if M["m00"] == 0: 
                continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,label))
    return points

def cluster_regions(points, w, h):
    """
    把散点分成多个“桌子区域”
    采用网格密度法合并近邻块
    """
    if not points:
        return []

    cell = max(60, int(min(w,h)/12))
    cols = math.ceil(w/cell)
    rows = math.ceil(h/cell)
    grid = [[0]*cols for _ in range(rows)]
    cell_pts = [[[] for _ in range(cols)] for __ in range(rows)]

    for (x,y,c) in points:
        cx = min(cols-1, x//cell)
        cy = min(rows-1, y//cell)
        grid[cy][cx] += 1
        cell_pts[cy][cx].append((x,y,c))

    thr = 6
    hits = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] >= thr]
    if not hits:
        # 回退：整体一块
        return [(int(w*0.02), int(h*0.2), int(w*0.96), int(h*0.7))]

    rects=[]
    for (r,c) in hits:
        x = c*cell; y=r*cell; ww=cell; hh=cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x>rx+rw+cell or x+ww<rx-cell or y>ry+rh+cell or y+hh<ry-cell):
                nx=min(rx,x); ny=min(ry,y)
                nw=max(rx+rw,x+ww)-nx; nh=max(ry+rh,y+hh)-ny
                rects[i]=(nx,ny,nw,nh)
                merged=True
                break
        if not merged:
            rects.append((x,y,ww,hh))
    # 扩边
    regs=[]
    for (x,y,ww,hh) in rects:
        nx=max(0,x-10); ny=max(0,y-10)
        nw=min(w-nx, ww+20); nh=min(h-ny, hh+20)
        regs.append((nx,ny,nw,nh))
    return regs

def analyze_board(bgr, region):
    x,y,w,h = region
    crop = bgr[y:y+h, x:x+w]
    pts = detect_red_blue_points(crop)
    if not pts:
        return {"total":0, "max_run":0, "dragon":"none", "multi3":False, "longish_cols":0}

    # 按列聚类（基于 x 接近）
    pts_sorted = sorted(pts, key=lambda p: p[0])
    columns=[]
    for px,py,c in pts_sorted:
        placed=False
        for col in columns:
            if abs(col["mx"] - px) <= max(8, w//40):
                col["pts"].append((px,py,c))
                col["mx"] = (col["mx"]*len(col["pts"][:-1]) + px)/len(col["pts"])
                placed=True
                break
        if not placed:
            columns.append({"mx":px, "pts":[(px,py,c)]})

    # 每列从上到下排序，得到颜色序列；计算该列“最大同色连续长度”与“主色”
    col_info=[]
    for col in columns:
        seq = [c for (_,py,c) in sorted(col["pts"], key=lambda t: t[1])]
        # 最大同色连续长度
        max_run = 0; cur_c=None; cur_len=0
        for ch in seq:
            if ch==cur_c: cur_len+=1
            else:
                if cur_len>max_run: max_run=cur_len
                cur_c=ch; cur_len=1
        if cur_len>max_run: max_run=cur_len
        # 本列主色 = 长连颜色
        # 如果出现相等，取出现次数最多的颜色
        major = max(set(seq), key=seq.count)
        col_info.append({"max_run":max_run, "major":major})

    # 统计整桌“最大同色连续长度”（把列拼接阅读模式）
    # 同时识别“连续多列的同色连≥4”的最长列串（用于 连珠）
    max_any_run = 0
    longish_cols = 0
    best_multi_same = 0
    cur_same = 0
    prev_major = None
    for ci in col_info:
        if ci["max_run"] >= 4:
            longish_cols += 1
            if prev_major is None or ci["major"]==prev_major:
                cur_same += 1
            else:
                cur_same = 1
            prev_major = ci["major"]
            if cur_same > best_multi_same:
                best_multi_same = cur_same
        else:
            # 断开
            prev_major = None
            cur_same = 0

        if ci["max_run"] > max_any_run:
            max_any_run = ci["max_run"]

    # 龙类判断
    dragon = "none"
    if max_any_run >= 10: dragon = "super"
    elif max_any_run >= 8: dragon = "long"
    elif max_any_run >= 4: dragon = "longish"

    total_points = sum(len(c["pts"]) for c in columns)
    return {
        "total": int(total_points),
        "max_run": int(max_any_run),
        "dragon": dragon,          # none/long/ super/ longish
        "multi3": best_multi_same >= 3,  # 连续3排连珠
        "longish_cols": int(longish_cols)
    }

def classify_all(stats):
    """
    只对两种时段提醒：
    1) 放水时段（提高胜率）：
       - 条件1：≥3 桌 dragon in {long, super}    （含超长）
       - 或 条件2：super ≥1 且 另外的 long ≥2     （即 1 超 + 2 长）
    2) 中等胜率（中上）：
       - ≥3 桌 multi3==True （连续3排连珠）
       - 且 ≥2 桌 dragon in {long, super}   （可与上面重叠同桌）
    其它：不提醒
    """
    long_tables = sum(1 for s in stats if s["dragon"] in ("long","super"))
    super_tables = sum(1 for s in stats if s["dragon"] == "super")
    multi3_tables = sum(1 for s in stats if s["multi3"])

    # 放水
    cond_pour = (long_tables >= max(3, MIN_TABLES_LONG_FOR_POUR)) or (super_tables >= 1 and (long_tables - super_tables) >= 2)
    if cond_pour:
        return "放水时段（提高胜率）", {"long_tables": long_tables, "super_tables": super_tables, "multi3_tables": multi3_tables}

    # 中上
    if multi3_tables >= 3 and long_tables >= 2:
        return "中等胜率（中上）", {"long_tables": long_tables, "super_tables": super_tables, "multi3_tables": multi3_tables}

    return None, {"long_tables": long_tables, "super_tables": super_tables, "multi3_tables": multi3_tables}

# ========== 估计/历史 ==========
def estimate_minutes(kind, history):
    # 使用同类历史平均，否则默认
    xs = [h["duration_minutes"] for h in history if h.get("kind")==kind and h.get("duration_minutes",0)>0]
    if xs:
        return max(5, int(round(sum(xs)/len(xs))))
    return DEFAULT_EST_MIN_POUR if kind=="放水时段（提高胜率）" else DEFAULT_EST_MIN_MIDUP

def fmt_time(dt):
    return dt.strftime("%H:%M")

def start_event_and_notify(state, kind, meta):
    history = state.get("history", [])
    est_min = estimate_minutes(kind, history)
    est_min = min(est_min, SAFE_MAX_SLEEP_MIN)
    start = now()
    expected_end = start + timedelta(minutes=est_min)

    state.update({
        "active": True,
        "kind": kind,
        "start": start.isoformat(),
        "expected_end": expected_end.isoformat()
    })
    save_state(state)

    remain = est_min
    emoji = "💧" if kind.startswith("放水") else "⚠️"
    msg = (
        f"{emoji}【{kind}】已检测到\n"
        f"时间（MYT）：{start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"统计：长龙/超长龙桌数={meta['long_tables']}（其中超长龙={meta['super_tables']}），"
        f"连续3排连珠桌数={meta['multi3_tables']}\n"
        f"预计结束时间：{fmt_time(expected_end)}（剩下{remain}分钟）\n"
        f"说明：到预计时间前将暂停再次检测。"
    )
    send_telegram(msg)
    log("已发送开始提醒")

def end_event_and_notify(state, reason="到达预计时间后复检结束"):
    if not state.get("active"):
        return
    kind = state.get("kind")
    start = datetime.fromisoformat(state.get("start"))
    endt  = now()
    dur_min = max(1, int(round((endt - start).total_seconds()/60.0)))

    history = state.get("history", [])
    history.append({
        "kind": kind,
        "start": state.get("start"),
        "end": endt.isoformat(),
        "duration_minutes": dur_min,
        "end_reason": reason
    })
    history = history[-120:]  # 保留近120条
    state.update({"active": False, "kind": None, "start": None, "expected_end": None, "history": history})
    save_state(state)

    emoji = "✅"
    msg = (
        f"{emoji}【{kind}】已结束\n"
        f"开始：{start.strftime('%Y-%m-%d %H:%M:%S')}  结束：{endt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"实际持续：{dur_min} 分钟"
    )
    send_telegram(msg)
    log("已发送结束提醒")

# ========== 一次“实盘检测” ==========
def one_detection():
    """
    返回 (kind/meta) 或 (None/meta) 以及 debug 信息
    """
    with sync_playwright() as p:
        img = enter_dg_and_screenshot(p)
    if img is None:
        return None, {"error":"无法进入或截图为空"}

    bgr = pil_to_bgr(img)
    h, w = bgr.shape[:2]
    points = detect_red_blue_points(bgr)
    if not points:
        return None, {"error":"未检测到红/蓝珠"}

    regions = cluster_regions(points, w, h)
    stats=[]
    for reg in regions:
        st = analyze_board(bgr, reg)
        # 过滤明显空白/噪声区域
        if st["total"] >= 6:
            stats.append(st)

    kind, meta = classify_all(stats)

    # 写 debug
    with open(DEBUG_LAST, "w", encoding="utf-8") as f:
        json.dump({"when": ts(), "kind": kind, "meta": meta, "boards": stats[:50]}, f, ensure_ascii=False, indent=2)

    return kind, meta

# ========== 主循环（进程内 5 分钟） ==========
def main_loop():
    deadline = now() + timedelta(minutes=IN_LOOP_MINUTES)
    log(f"进入主循环，将持续约 {IN_LOOP_MINUTES} 分钟。")

    while now() < deadline:
        try:
            state = load_state()

            # 若已在活动中，且未到预计结束点 -> 休眠到预计结束
            if state.get("active") and state.get("expected_end"):
                exp = datetime.fromisoformat(state["expected_end"])
                if now() < exp:
                    to_sleep = int(max(1, (exp - now()).total_seconds()))
                    mins = int(math.ceil(to_sleep/60))
                    log(f"处于{state.get('kind')} 活动期，预计结束 {fmt_time(exp)}，将暂停检测约 {mins} 分钟。")
                    time.sleep(min(to_sleep, SAFE_MAX_SLEEP_MIN*60))
                    # 到点后继续下一轮（将复检并结束或进入新活动）
                    continue

            # 到这里：可以进行一次实盘检测
            log("开始一次实盘检测 ...")
            kind, meta = one_detection()
            if kind in ("放水时段（提高胜率）", "中等胜率（中上）"):
                if not state.get("active"):
                    start_event_and_notify(state, kind, meta)
                else:
                    # 已经在活动中（可能类型一致/不同），到点才复检；这里不重复提醒
                    log(f"已在活动中：{state.get('kind')}，不重复提醒。")
            else:
                # 非提醒区间，如果之前有活动则结束它（说明复检确认已转差）
                if state.get("active"):
                    end_event_and_notify(state, reason="复检未满足提醒条件")
                else:
                    log("本次无提醒（胜率中等/收割），保持静默。")

            # 正常 5 分钟间隔
            time.sleep(DETECT_INTERVAL_SECONDS)

        except Exception as e:
            log("检测循环异常（已捕获，不会中断）：")
            log(str(e))
            traceback.print_exc()
            # 出错也等 5 分钟再来，避免高频重试
            time.sleep(DETECT_INTERVAL_SECONDS)

    log("主循环结束（将由下一次 Actions 触发继续运行）。")

if __name__ == "__main__":
    main_loop()
