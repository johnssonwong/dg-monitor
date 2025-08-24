# -*- coding: utf-8 -*-
# scripts/monitor_dg.py
# 实盘 DG 检测：Playwright 真实打开页面 + OpenCV 分析珠盘
# 规则要点：
# - 中等胜率（中上） = 至少3桌「连续3排 多连/连珠(每排≥4)」 + 至少3桌「长龙(≥8)或超长龙(≥10)」
# - 放水时段（提高胜率） = 桌面总体“密度高”（空桌少、珠点多） 且 长龙/超长龙桌数≥3，且多连/连珠分布广（强势）
# - 胜率中等 / 收割：不提醒
#
# 提醒策略：
# - 只有「放水时段（提高胜率）」与「中等胜率（中上）」两种会发 Telegram。
# - 进入提醒状态后不重复提醒；当跌出提醒状态，自动发送「放水已结束」并报「持续时长」。
# - 消息包含 emoji、预计结束时间（有趋势才给）、预计剩余时长。
#
# 运行环境：GitHub Actions（每5分钟）
# 依赖：playwright, opencv-python-headless, numpy, pillow, requests

import os, json, time, math, subprocess, traceback
from io import BytesIO
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import cv2
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ========== 配置（按需可微调） ==========
BOT_TOKEN = os.getenv("DG_BOT_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8")
CHAT_ID   = os.getenv("DG_CHAT_ID", "485427847")
DG_URLS   = [os.getenv("DG_URL1","https://dg18.co/wap/"), os.getenv("DG_URL2","https://dg18.co/")]
TZ_OFFSET = os.getenv("TZ_OFFSET","+08:00")  # 马来西亚

STATE_FILE = ".dg_state.json"  # 保存状态（开始时间、历史序列等）

# 连的定义（你原始定义）
LEN_LONGISH = 4         # 长连（≥4）
LEN_LONG    = 8         # 长龙（≥8）
LEN_SUPER   = 10        # 超长龙（≥10）

# 中上与放水的硬阈值（根据你最新口径）
REQ_MULTI3_TABLES = 3   # 至少3桌满足「连续3排 多连/连珠(≥4)」
REQ_LONG_TABLES   = 3   # 至少3桌满足「长龙(≥8)或超长龙(≥10)」

# 放水（提高胜率）额外要求（更强势）
MIN_DENSE_RATIO   = 0.65  # 桌面密度：非空桌（珠点>=6）的占比阈值
MIN_LONG_SPREAD   = 4     # 长龙/超长龙桌数更高一些（≥4更有把握，可按图再调）
MIN_MULTI3_SPREAD = 4     # 连珠“排排连”更广（≥4桌）

# 检测图像参数
RESIZE_WIDTH = 1500      # 统一缩放宽度
CELL_GAP_X   = 16        # 按列聚类的 X 间距
MIN_BLOB     = 5         # 最小色块像素数（噪声剔除）
SPARSE_BEADS = 6         # 小于此珠数视为“空/稀”

# 结束时间预测（基于最近N次趋势）
HIST_KEEP   = 12         # 最多保留最近N次（约1小时）数据点
TREND_MIN_K = 3          # 至少3个点才做线性趋势
FALLING_MIN = 0.05       # 下降速度阈值（单位：每分钟的“桌数”）
COOLDOWN_MINUTES = 10    # 发送一次提醒后的冷却时间（防打扰）

# =====================================

def now_utc():
    return datetime.now(timezone.utc)

def to_local(dt_utc):
    sign = 1 if TZ_OFFSET.startswith("+") else -1
    hh = int(TZ_OFFSET[1:3])
    mm = int(TZ_OFFSET[4:6]) if len(TZ_OFFSET) >= 6 else 0
    tz = timezone(timedelta(hours=sign*hh, minutes=sign*mm))
    return dt_utc.astimezone(tz)

def fmt_ampm_dot(dt_local):
    # 例：7.50am（与你示例一致）
    h = dt_local.hour
    m = dt_local.minute
    ampm = "am" if h < 12 else "pm"
    hh = h if 1 <= h <= 12 else (h-12 if h>12 else 12)
    return f"{hh}.{m:02d}{ampm}"

def send_telegram(text):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=30
        )
        ok = r.json().get("ok", False)
        if not ok:
            print("Telegram failed:", r.text)
        return ok
    except Exception as e:
        print("Telegram error:", e)
        return False

def load_state():
    p = Path(STATE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_state(state):
    Path(STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    # 提交到仓库（供下次运行读取）
    try:
        subprocess.run(["git", "config", "--global", "user.email", "dg-monitor@example.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "dg-monitor-bot"], check=True)
        subprocess.run(["git", "add", STATE_FILE], check=True)
        # 若无变化会报错，忽略
        subprocess.run(["git", "commit", "-m", "dg: update state"], check=False)
        subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print("WARN: git push state failed:", e)

# ------- Playwright 抓图（实盘进入） -------
def open_and_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage"]
        )
        ctx = browser.new_context(
            viewport={"width": 1600, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = ctx.new_page()
        for url in DG_URLS:
            try:
                page.goto(url, timeout=30000)
                page.wait_for_timeout(1500)

                # 点击“免费试玩 / Free”之类
                candidates = [
                    "免费试玩", "免費試玩", "Free", "free", "试玩", "試玩", "立即体验", "立即體驗"
                ]
                clicked = False
                for t in candidates:
                    try:
                        el = page.get_by_text(t, exact=False)
                        el.first.click(timeout=3000)
                        clicked = True
                        page.wait_for_timeout(1500)
                        break
                    except:
                        pass

                # 若有新窗口跳转，切到最新页
                if len(ctx.pages) > 1:
                    page = ctx.pages[-1]
                    page.wait_for_timeout(1500)

                # 简单处理滑动验证/安全条（尽力模拟）
                # 尝试多种常见选择器
                for sel in [
                    "input[type=range]",
                    ".slider", ".slide", ".drag", ".geetest_slider_button"
                ]:
                    try:
                        el = page.locator(sel).first
                        box = el.bounding_box()
                        if box:
                            sx = box["x"] + box["width"] * 0.1
                            sy = box["y"] + box["height"] * 0.5
                            ex = box["x"] + box["width"] * 0.9
                            page.mouse.move(sx, sy)
                            page.mouse.down()
                            page.mouse.move(ex, sy, steps=25)
                            page.mouse.up()
                            page.wait_for_timeout(2000)
                            break
                    except:
                        pass

                # 等待桌面加载
                page.wait_for_timeout(2500)

                img = page.screenshot(full_page=True)
                if img and len(img) > 6000:
                    browser.close()
                    return img
            except PWTimeout:
                print("Timeout:", url)
            except Exception as e:
                print("Open error:", url, e)
        browser.close()
        return None

# ------- OpenCV 分析 -------
def hsv_mask_rb(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 红色两个区间
    r1 = cv2.inRange(hsv, np.array([0, 80, 60]),   np.array([8, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 80, 60]), np.array([180,255,255]))
    red = cv2.bitwise_or(r1, r2)
    # 蓝色
    blue = cv2.inRange(hsv, np.array([95, 70, 60]), np.array([135,255,255]))
    return red, blue

def detect_blobs(bgr):
    red, blue = hsv_mask_rb(bgr)
    blobs = []
    for mask, tag in [(red,'B'),(blue,'P')]:
        n, labels = cv2.connectedComponents(mask)
        for lab in range(1, n):
            ys, xs = np.where(labels==lab)
            cnt = len(xs)
            if cnt < MIN_BLOB: 
                continue
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            blobs.append((cx, cy, tag))
    return blobs

def cluster_columns(xs, gap=CELL_GAP_X):
    xs = sorted(xs)
    groups = [[xs[0]]] if xs else []
    for v in xs[1:]:
        if v - groups[-1][-1] <= gap: groups[-1].append(v)
        else: groups.append([v])
    centers = [int(sum(g)/len(g)) for g in groups]
    return centers

def max_run_len(seq):
    if not seq: return 0
    mx, cur, cnt = 1, seq[0], 1
    for c in seq[1:]:
        if c == cur: cnt += 1
        else:
            mx = max(mx, cnt)
            cur, cnt = c, 1
    mx = max(mx, cnt)
    return mx

def longest_consecutive_true(bools):
    mx = cur = 0
    for v in bools:
        if v: cur += 1
        else:
            mx = max(mx, cur); cur = 0
    return max(mx, cur)

def analyze_one_table(bgr):
    # 1) 取色块（红/蓝珠）
    blobs = detect_blobs(bgr)
    if not blobs:
        return dict(total=0, maxrun=0, is_long=False, is_super=False,
                    is_longish=False, has_multi3=False)

    # 2) 按X聚类得到“列”（珠盘按列放）
    xs = [c for c,_,_ in blobs]
    cols_x = cluster_columns(xs)
    columns = {cx: [] for cx in cols_x}
    for cx, cy, tag in blobs:
        # 归最近列中心
        nearest = min(cols_x, key=lambda c: abs(c - cx))
        columns[nearest].append((cx, cy, tag))

    # 3) 列内按Y排序，得到列序列，再统计每列最长同色run
    col_max_runs = []
    flattened = []
    for cx in sorted(columns.keys()):
        items = sorted(columns[cx], key=lambda t:t[1])  # y升序
        seq = [t[2] for t in items]
        if seq:
            col_max_runs.append(max_run_len(seq))
            flattened.extend(seq)

    # 4) 整体最大连（用于长龙/超长龙判断）
    overall_max = max_run_len(flattened)
    is_super = overall_max >= LEN_SUPER
    is_long  = overall_max >= LEN_LONG
    is_longish = overall_max >= LEN_LONGISH

    # 5) 「连续3排 多连/连珠」：相邻“三列”都满足列内最长run≥4
    has_multi3 = longest_consecutive_true([v >= LEN_LONGISH for v in col_max_runs]) >= 3

    return dict(
        total=len(flattened),
        maxrun=overall_max,
        is_long=is_long,
        is_super=is_super,
        is_longish=is_longish,
        has_multi3=has_multi3
    )

def find_candidate_tables(whole):
    # 粗分：把整图按网格找“彩点密集区”，作为候选桌子区域
    H, W = whole.shape[:2]
    cell = 120  # 较大网格，适配 1500 宽
    hsv = cv2.cvtColor(whole, cv2.COLOR_BGR2HSV)
    heat = np.zeros((H//cell+1, W//cell+1), dtype=np.int32)
    for y in range(0,H,3):
        for x in range(0,W,3):
            h,s,v = hsv[y,x]
            if ((h<=8 or h>=170) and s>70 and v>50) or (95<=h<=135 and s>70 and v>50):
                heat[y//cell, x//cell] += 1

    thr = max(10, int(np.percentile(heat[heat>0], 40)))  # 中位偏下阈值
    hits = np.argwhere(heat >= thr)
    if hits.size == 0:
        return [(0,0,W,H)]
    # 合并相邻块为较大矩形
    rects = []
    for (ry, rx) in hits:
        x,y = rx*cell, ry*cell
        w,h = cell, cell
        merged = False
        for r in rects:
            if not (x>r[0]+r[2]+cell or x+w<r[0]-cell or y>r[1]+r[3]+cell or y+h<r[1]-cell):
                nx = min(r[0], x); ny=min(r[1], y)
                r[2] = max(r[0]+r[2], x+w) - nx
                r[3] = max(r[1]+r[3], y+h) - ny
                r[0], r[1] = nx, ny
                merged = True
                break
        if not merged:
            rects.append([x,y,w,h])
    # 轻微扩张边缘
    out=[]
    for x,y,w,h in rects:
        x = max(0, x-6); y=max(0, y-6)
        w = min(W-x, w+12); h=min(H-y, h+12)
        out.append((x,y,w,h))
    return out

def classify_all_tables(bgr):
    # 统一大小
    H, W = bgr.shape[:2]
    scale = RESIZE_WIDTH / float(W) if W > RESIZE_WIDTH else 1.0
    if scale != 1.0:
        bgr = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    rects = find_candidate_tables(bgr)
    tables = []
    for (x,y,w,h) in rects:
        sub = bgr[y:y+h, x:x+w]
        res = analyze_one_table(sub)
        res.update(rect=(x,y,w,h))
        tables.append(res)

    # 汇总
    long_tables   = sum(1 for t in tables if t["is_long"] or t["is_super"])
    super_tables  = sum(1 for t in tables if t["is_super"])
    multi3_tables = sum(1 for t in tables if t["has_multi3"])
    dense_ratio   = sum(1 for t in tables if t["total"] >= SPARSE_BEADS) / max(1,len(tables))
    longish_spread= sum(1 for t in tables if t["is_longish"])

    # 四类判定：
    # 1) 放水时段（提高胜率）：强势、密度高、长龙广布、连珠广布
    if (long_tables >= REQ_LONG_TABLES and
        ((super_tables >= 1 and long_tables >= REQ_LONG_TABLES) or long_tables >= MIN_LONG_SPREAD) and
        dense_ratio >= MIN_DENSE_RATIO and
        multi3_tables >= MIN_MULTI3_SPREAD):
        overall = "放水时段（提高胜率）"

    # 2) 中等胜率（中上）：你最新明确口径
    elif (multi3_tables >= REQ_MULTI3_TABLES and long_tables >= REQ_LONG_TABLES):
        overall = "中等胜率（中上）"

    # 3) 胜率中等 / 收割（不提醒）
    else:
        # 稀疏且几乎没有连
        sparse_ratio = 1.0 - dense_ratio
        if sparse_ratio >= 0.6 and long_tables < 2 and multi3_tables < 2:
            overall = "收割时段（胜率调低）"
        else:
            overall = "胜率中等"

    summary = dict(
        tables=len(tables),
        long_tables=long_tables,
        super_tables=super_tables,
        multi3_tables=multi3_tables,
        dense_ratio=round(dense_ratio,3),
        longish_spread=longish_spread
    )
    return overall, summary, tables

# ------- 趋势预测（预计结束时间/剩余时长） -------
def push_history(state, overall, summary, nowu):
    h = state.get("history", [])
    h.append(dict(
        ts=nowu.timestamp(),
        overall=overall,
        long_tables=summary["long_tables"],
        multi3_tables=summary["multi3_tables"]
    ))
    if len(h) > HIST_KEEP:
        h = h[-HIST_KEEP:]
    state["history"] = h

def estimate_eta_minutes(state, current_overall, req_long=REQ_LONG_TABLES, req_multi=REQ_MULTI3_TABLES):
    h = state.get("history", [])
    if len(h) < TREND_MIN_K:
        return None  # 样本不足，不给“假预测”
    # 取最近K点
    K = TREND_MIN_K
    hx = h[-K:]
    t0, t1 = hx[0]["ts"], hx[-1]["ts"]
    minutes = max(1e-6, (t1 - t0) / 60.0)

    # 线性下降趋势估计
    l0, l1 = hx[0]["long_tables"], hx[-1]["long_tables"]
    m0, m1 = hx[0]["multi3_tables"], hx[-1]["multi3_tables"]
    slope_l = (l1 - l0) / minutes
    slope_m = (m1 - m0) / minutes

    # 只有在“放水/中上”里才预测结束（跌出阈值）
    if current_overall == "放水时段（提高胜率）":
        # 放水阈值近似：long_tables ≥ MIN_LONG_SPREAD 且 multi3_tables ≥ MIN_MULTI3_SPREAD
        thr_l, thr_m = MIN_LONG_SPREAD, MIN_MULTI3_SPREAD
    elif current_overall == "中等胜率（中上）":
        thr_l, thr_m = req_long, req_multi
    else:
        return None

    etas = []
    if slope_l < -FALLING_MIN and l1 > thr_l:
        etas.append( (l1 - thr_l) / (-slope_l) )
    if slope_m < -FALLING_MIN and m1 > thr_m:
        etas.append( (m1 - thr_m) / (-slope_m) )

    if not etas:
        return None  # 没有明显下降趋势，不给不靠谱的“预计结束时间”
    return max(1.0, min(180.0, float(min(etas))))  # 夹在1~180分钟内，避免极端

# ------- 主流程 -------
def main():
    try:
        nowu = now_utc()
        print("Start:", to_local(nowu).strftime("%Y-%m-%d %H:%M:%S"))

        state = load_state()
        last_status = state.get("status")  # 上一轮总体状态
        start_ts   = state.get("start_ts") # 放水开始UTC秒
        cooldown_until = state.get("cooldown_until", 0)

        # 1) 抓 DG 截图
        shot = open_and_screenshot()
        if not shot:
            print("ERROR: 无法抓到DG截图（可能遇到强验证/网络波动）")
            return  # 不抛异常，确保exit 0

        # 2) OpenCV 分析
        img = cv2.imdecode(np.frombuffer(shot, np.uint8), cv2.IMREAD_COLOR)
        overall, summary, tables = classify_all_tables(img)
        print("OVERALL:", overall, summary)

        # 3) 写入历史（用于“预计结束时间”趋势）
        push_history(state, overall, summary, nowu)

        # 4) 进入/退出提醒逻辑
        in_alert_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")
        in_alert_prev= last_status in ("放水时段（提高胜率）", "中等胜率（中上）")

        # 预计结束时间/剩余时长（仅在提醒态才算）
        eta_minutes = estimate_eta_minutes(state, overall)

        if in_alert_now and not in_alert_prev:
            # 新进入提醒态
            state["status"]    = overall
            state["start_ts"]  = nowu.timestamp()
            state["cooldown_until"] = (nowu + timedelta(minutes=COOLDOWN_MINUTES)).timestamp()

            # 构建消息
            local_now = to_local(nowu)
            if eta_minutes:
                eta_end_local = local_now + timedelta(minutes=eta_minutes)
                msg = (
                    f"🔔 [DG提醒] {overall}\n"
                    f"开始时间：{local_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"长/超长龙桌={summary['long_tables']}（超长={summary['super_tables']}），"
                    f"多连3排桌={summary['multi3_tables']}\n"
                    f"预计结束时间：{fmt_ampm_dot(eta_end_local)}\n"
                    f"此局势预计：剩下{int(round(eta_minutes))}分钟"
                )
            else:
                msg = (
                    f"🔔 [DG提醒] {overall}\n"
                    f"开始时间：{local_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"长/超长龙桌={summary['long_tables']}（超长={summary['super_tables']}），"
                    f"多连3排桌={summary['multi3_tables']}\n"
                    f"预计结束时间：暂无法可靠预测（趋势未显示下降）\n"
                    f"此局势预计：持续中"
                )
            send_telegram(msg)
            save_state(state)
            return

        if in_alert_now and in_alert_prev:
            # 已经在提醒态：只更新历史、但不重复提醒
            save_state(state)
            return

        if (not in_alert_now) and in_alert_prev:
            # 结束：从提醒态 -> 非提醒
            start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc) if start_ts else nowu
            dur = nowu - start_dt
            mins = int(dur.total_seconds() // 60)
            secs = int(dur.total_seconds() % 60)
            msg = (
                f"✅ [DG结束] {last_status}\n"
                f"开始：{to_local(start_dt).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"结束：{to_local(nowu).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"持续：{mins}分{secs}秒"
            )
            send_telegram(msg)
            # 清理状态
            state["status"]    = overall
            state["start_ts"]  = None
            state["cooldown_until"] = 0
            save_state(state)
            return

        # 都不在提醒态：不发
        state["status"] = overall
        save_state(state)

    except Exception as e:
        # 捕获所有异常，打印堆栈避免“exit code 1/2”
        print("UNCAUGHT ERROR:", repr(e))
        traceback.print_exc()
        # 不 raise，确保 GitHub Actions 退出码为 0

if __name__ == "__main__":
    main()
