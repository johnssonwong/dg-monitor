# -*- coding: utf-8 -*-
"""
DreamGaming(DG) 自动检测 + 放水提醒 (Telegram)
规则完全基于你在本聊天中给出的定义与门槛：
- 术语：长连(≥4)、多连/连珠(一排≥4且下一排再≥4)、长龙(≥8)、超长龙(≥10)、单跳(1粒/2~3粒)
- 放水时段（胜率提高）判定：① 满盘长连局势；或 ② 超长龙触发（≥1条超长龙 + ≥2条长龙）
  * 满盘长连局势阈值：20桌≥8张（或）10桌≥4张
- 中等胜率（中上）判定：20桌≥6张（或）10桌≥3张 + 至少2桌(长龙/超长龙) + 出现多连/连珠
- 胜率中等、收割时段：不提醒（空白多、单跳多、连少）
- 少于2桌长龙一律视为“假信号”不入场
- 忽略“持续3次单跳”对放水判定的干扰
- 只对“放水时段（胜率提高）/中等胜率（中上）”发提醒；开始时提示，结束时提示“共持续XX分钟”
- 预计结束时间：基于近几次有效长连桌数的线性趋势估算（无明显下降趋势则不给预计时间）
- 时区：马来西亚 Asia/Kuala_Lumpur
"""

import os, io, json, math, time, statistics, traceback
from datetime import datetime, timedelta
import pytz
import requests
import numpy as np
import cv2
from playwright.sync_api import sync_playwright

# ============ Telegram（已内置你的信息；也支持用Secrets覆盖） ============
TG_TOKEN   = os.getenv("TG_TOKEN") or "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TG_CHAT_ID = os.getenv("TG_CHAT_ID") or "485427847"

# ============ DG入口 ============
DG_URLS = ["https://dg18.co/wap/", "https://dg18.co/"]

# ============ 时区 ============
TZ = pytz.timezone("Asia/Kuala_Lumpur")

# ============ 状态文件 ============
STATE_FILE = "state.json"

# ---------------- 工具函数 ----------------
def now_ms():
    return int(datetime.now(TZ).timestamp() * 1000)

def send_telegram(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=20
        )
    except Exception:
        pass

def bytes_to_bgr(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ---------------- 圆点检测（红/蓝） ----------------
def find_red_blue_points(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 红色（两段）
    lower_red1 = np.array([0, 70, 70]);   upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 70]); upper_red2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    # 蓝色
    lower_blue = np.array([100, 80, 70]); upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    def detect_centers(mask):
        blur = cv2.GaussianBlur(mask, (5,5), 1)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=8,
                                   param1=60, param2=12, minRadius=4, maxRadius=16)
        pts = []
        if circles is not None:
            for c in np.uint16(np.around(circles[0, :])):
                x, y, r = int(c[0]), int(c[1]), int(c[2])
                pts.append((x, y))
        return pts

    red_pts = detect_centers(mask_red)
    blue_pts = detect_centers(mask_blue)
    return red_pts, blue_pts

def cluster_columns(points, x_tol=None):
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in pts]
    diffs = [xs[i+1]-xs[i] for i in range(len(xs)-1)]
    cell_w = statistics.median(diffs) if diffs else 16
    if x_tol is None:
        x_tol = max(8, int(cell_w*0.6))
    cols = [[pts[0]]]
    for p in pts[1:]:
        if abs(p[0] - cols[-1][-1][0]) <= x_tol:
            cols[-1].append(p)
        else:
            cols.append([p])
    cols = [sorted(c, key=lambda p: p[1]) for c in cols]
    return cols

def longest_run_in_col(col_pts):
    ys = [p[1] for p in col_pts]
    if not ys: return 0
    runs = 1; best = 1
    for i in range(1, len(ys)):
        if ys[i] - ys[i-1] > 6:  # 同排向下增加一粒
            runs += 1
            best = max(best, runs)
    return best

def analyze_table_image(img_bgr):
    red_pts, blue_pts = find_red_blue_points(img_bgr)
    marker_count = len(red_pts) + len(blue_pts)

    red_cols = cluster_columns(red_pts)
    blue_cols = cluster_columns(blue_pts)
    all_cols = cluster_columns(red_pts + blue_pts)

    long4_cols_R = [c for c in red_cols if longest_run_in_col(c) >= 4]
    long4_cols_B = [c for c in blue_cols if longest_run_in_col(c) >= 4]

    long8_R = any(longest_run_in_col(c) >= 8 for c in red_cols)
    long8_B = any(longest_run_in_col(c) >= 8 for c in blue_cols)
    long10_R = any(longest_run_in_col(c) >= 10 for c in red_cols)
    long10_B = any(longest_run_in_col(c) >= 10 for c in blue_cols)

    has_long4 = (len(long4_cols_R) + len(long4_cols_B)) > 0
    has_long8 = (long8_R or long8_B)
    has_long10 = (long10_R or long10_B)

    # 多连/连珠：相邻两列同色均>=4
    def has_duolian_two(color_cols):
        if len(color_cols) < 2: return False
        cols_with_x = []
        for col in color_cols:
            xs = [p[0] for p in col]
            cols_with_x.append((statistics.mean(xs), col))
        cols_with_x.sort(key=lambda t: t[0])
        for i in range(len(cols_with_x)-1):
            if longest_run_in_col(cols_with_x[i][1])>=4 and longest_run_in_col(cols_with_x[i+1][1])>=4:
                return True
        return False
    has_duolian = has_duolian_two(long4_cols_R) or has_duolian_two(long4_cols_B)

    # 单跳比例（粗估）：列的最长连<=1 视作单跳列
    if all_cols:
        single_jump_cols = sum(1 for c in all_cols if longest_run_in_col(c) <= 1)
        single_jump_ratio = single_jump_cols/len(all_cols)
    else:
        single_jump_ratio = 1.0

    return {
        "has_long4": has_long4,
        "has_long8": has_long8,
        "has_long10": has_long10,
        "has_duolian": has_duolian,
        "single_jump_ratio": float(single_jump_ratio),
        "marker_count": int(marker_count),
    }

# ---------------- Playwright：进入DG并截取各桌canvas ----------------
def solve_slider_if_any(page):
    candidates = [
        "div.geetest_slider_button", "div.nc_iconfont.btn_slide", "div.slider", "div#nc_1_n1z",
        "div.yidun_slider", "div.verify-slider", "div.slider-btn", "span.btn_slide", "div.slider_button"
    ]
    for sel in candidates:
        try:
            btn = page.query_selector(sel)
            if btn:
                box = btn.bounding_box()
                if box:
                    page.mouse.move(box["x"]+box["width"]/2, box["y"]+box["height"]/2)
                    page.mouse.down()
                    total = 420; step = 42
                    for dx in range(0, total, step):
                        page.mouse.move(box["x"]+box["width"]/2+dx, box["y"]+box["height"]/2, steps=2)
                        time.sleep(0.05)
                    page.mouse.up()
                    time.sleep(1.0)
        except Exception:
            pass

def capture_table_canvases(context, page):
    images = []
    try:
        page.mouse.wheel(0, 400)
        time.sleep(0.3)
    except Exception:
        pass
    canvases = page.query_selector_all("canvas")
    for c in canvases:
        try:
            box = c.bounding_box()
            if not box: 
                continue
            # 过滤尺寸（经验）
            if 120 <= box["width"] <= 420 and 90 <= box["height"] <= 250:
                img_bytes = c.screenshot()
                images.append(img_bytes)
        except Exception:
            continue
    return images

def enter_dg_and_get_tables():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(viewport={"width": 1440, "height": 2800})
        page = context.new_page()

        for url in DG_URLS:
            try:
                page.goto(url, timeout=30000)
                # 点击“Free/免费试玩”
                clicked = False
                for sel in ["text=Free", "text=免费试玩", "text=FREE", "text=free"]:
                    try:
                        page.click(sel, timeout=3000)
                        clicked = True
                        break
                    except Exception:
                        pass
                if not clicked:
                    buttons = page.query_selector_all("button")
                    for b in buttons:
                        txt = (b.inner_text() or "").strip()
                        if "Free" in txt or "免费试玩" in txt:
                            b.click(timeout=2000)
                            clicked = True
                            break

                # 新页面
                for _ in range(20):
                    time.sleep(0.3)
                    if len(context.pages) > 1:
                        page = context.pages[-1]
                        break

                solve_slider_if_any(page)
                page.wait_for_timeout(2000)

                canvases = capture_table_canvases(context, page)
                browser.close()
                return canvases
            except Exception:
                continue
        browser.close()
        return []

# ---------------- 局势分类 ----------------
def classify_overall(table_stats):
    n = len(table_stats)
    long4_tables = sum(1 for t in table_stats if t["has_long4"] and t["single_jump_ratio"] < 0.7)
    long8_tables = sum(1 for t in table_stats if t["has_long8"])
    long10_tables = sum(1 for t in table_stats if t["has_long10"])
    duolian_tables = sum(1 for t in table_stats if t["has_duolian"])
    many_single_jump = sum(1 for t in table_stats if t["single_jump_ratio"] >= 0.7)

    # 超长龙触发：≥1超长龙 + 另外≥2长龙
    long8_only = max(0, long8_tables - long10_tables)
    trigger_super = (long10_tables >= 1 and long8_only >= 2)

    # 满盘长连：20桌≥8张 或 10桌≥4张
    cond_full = (n >= 20 and long4_tables >= 8) or (n >= 10 and long4_tables >= 4)

    # 中等胜率（中上）：20桌≥6张 或 10桌≥3张；且至少2桌有(长龙/超长龙)；且有多连
    cond_mid_up = False
    if ((n >= 20 and long4_tables >= 6) or (n >= 10 and long4_tables >= 3)) \
       and ((long8_tables + long10_tables) >= 2) and (duolian_tables >= 1):
        cond_mid_up = True

    # 收割时段（差势）：长龙很少 + 单跳多（占≥50%）
    cond_harvest = (long8_tables < 2 and many_single_jump >= max(3, int(0.5*n)))

    detail = {
        "tables": n,
        "long4_tables": long4_tables,
        "long8_tables": long8_tables,
        "long10_tables": long10_tables,
        "duolian_tables": duolian_tables,
        "many_single_jump": many_single_jump,
        "trigger_super": trigger_super,
        "cond_full": cond_full,
        "cond_mid_up": cond_mid_up,
        "cond_harvest": cond_harvest
    }

    if trigger_super or cond_full:
        return "FANGSHUI", detail
    if cond_mid_up:
        return "MID_UP", detail
    if cond_harvest:
        return "HARVEST", detail
    return "MID", detail

# ---------------- 状态管理与ETA ----------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def estimate_eta(history, active_threshold):
    # 基于最近最多6次“有效长连桌数”做线性趋势，下降才给ETA
    if len(history) < 3:
        return None, None
    h = history[-6:]
    xs = np.array([(h[i]["ts_ms"] - h[0]["ts_ms"])/60000.0 for i in range(len(h))])
    ys = np.array([h[i]["metric"] for i in range(len(h))], dtype=float)
    A = np.vstack([xs, np.ones(len(xs))]).T
    try:
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    except Exception:
        return None, None
    if m >= -1e-6:
        return None, None
    t_cross = (active_threshold - c)/m
    now0 = h[0]["ts_ms"]
    eta_ms = now0 + int(t_cross*60000)
    mins_left = max(1, int((eta_ms - now_ms())/60000))
    eta_dt = datetime.fromtimestamp(eta_ms/1000, TZ)
    return eta_dt, mins_left

# ---------------- 主流程 ----------------
def main():
    canvases = enter_dg_and_get_tables()
    table_stats = []
    for img in canvases:
        bgr = bytes_to_bgr(img)
        stat = analyze_table_image(bgr)
        table_stats.append(stat)

    status, detail = classify_overall(table_stats)

    st = load_state()
    history = st.get("history", [])
    history.append({"ts_ms": now_ms(), "metric": detail["long4_tables"]})
    history = history[-24:]  # 约近2小时

    last_status = st.get("status")
    msg = None

    brief = f"桌数:{detail['tables']} | 长连≥4:{detail['long4_tables']} | 长龙≥8:{detail['long8_tables']} | 超长龙≥10:{detail['long10_tables']} | 多连:{detail['duolian_tables']}"

    if status in ("FANGSHUI", "MID_UP"):
        if last_status not in ("FANGSHUI", "MID_UP"):
            st["active_since"] = now_ms()
            st["active_type"] = status
            # 预计结束时间（趋势法，可能给不出）
            eta_dt, mins_left = estimate_eta(history, detail["long4_tables"])
            if eta_dt and mins_left:
                msg = (f"【{'放水时段（胜率提高）' if status=='FANGSHUI' else '中等胜率（中上）'}】已开始\n"
                       f"{brief}\n"
                       f"预计结束时间：{eta_dt.strftime('%H:%M')}（马来西亚时间）\n"
                       f"此局势预计：剩余约 {mins_left} 分钟")
            else:
                msg = (f"【{'放水时段（胜率提高）' if status=='FANGSHUI' else '中等胜率（中上）'}】已开始\n"
                       f"{brief}\n预计结束时间：暂无法可靠预估（趋势未显著下降）")
    else:
        if last_status in ("FANGSHUI", "MID_UP") and st.get("active_since"):
            dur_min = max(1, int((now_ms() - st["active_since"]) / 60000))
            msg = f"【放水已结束】共持续 {dur_min} 分钟\n{brief}"
            st["active_since"] = None
            st["active_type"] = None

    st["status"] = status
    st["history"] = history
    save_state(st)

    if msg:
        when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        send_telegram(f"{when}（马来西亚时间）\n{msg}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = f"运行异常：{e}\n{traceback.format_exc()}"
        try:
            send_telegram(err[:3500])
        except Exception:
            pass
        with open("last_error.txt", "w", encoding="utf-8") as f:
            f.write(err)
