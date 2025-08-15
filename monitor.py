# -*- coding: utf-8 -*-
"""
DG 自动检测（最终版）
- 每5分钟由 GitHub Actions 触发
- Playwright 进入 https://dg18.co/wap/ 或 https://dg18.co/ -> 点击 Free/免费试玩 -> 处理安全滑块(若有) -> 截取所有 canvas
- 用 CV 检测红/蓝圆点（红=庄 / 蓝=闲），按 x 聚类为“列”、按 y 判断同排连续数（长连/长龙/超长龙）
- 根据你所有聊天内给出的判定阈值判定四种时段（放水 / 中等胜率(中上) / 胜率中等 / 收割）
- 发送三路 Telegram 通知：放水提醒（✅）、状态心跳（ℹ️）、错误告警（⚠️）
- 记录放水开始时间并在放水结束时发送“放水已结束，共持续XX分钟”
"""

import os, time, json, traceback, statistics
from datetime import datetime, timedelta
import pytz
import requests
import numpy as np
import cv2
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------------- Configuration (已内置) ----------------
# 默认使用 Secrets 覆盖（在仓库 Settings -> Secrets -> Actions 添加 TG_TOKEN / TG_CHAT_ID 可更安全）
TG_TOKEN = os.getenv("TG_TOKEN") or "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TG_CHAT_ID = os.getenv("TG_CHAT_ID") or "485427847"

DG_URLS = ["https://dg18.co/wap/", "https://dg18.co/"]
TZ = pytz.timezone("Asia/Kuala_Lumpur")
STATE_FILE = "state.json"

# ---------------- Utilities ----------------
def now_ms():
    return int(datetime.now(TZ).timestamp() * 1000)

def ts_to_local_str(ts_ms):
    return datetime.fromtimestamp(ts_ms/1000, TZ).strftime("%Y-%m-%d %H:%M")

def send_telegram(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=20
        )
    except Exception:
        pass

# ---------------- CV helpers ----------------
def bytes_to_bgr(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def find_red_blue_points(bgr):
    """HSV阈值+Hough圆检测红蓝点（经验值）"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 红色
    lower_red1 = np.array([0, 70, 70]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 70]); upper_red2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    # 蓝色
    lower_blue = np.array([90, 50, 50]); upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    def detect_centers(mask):
        blur = cv2.GaussianBlur(mask, (5,5), 1)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=8,
                                   param1=60, param2=12, minRadius=4, maxRadius=24)
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
    diffs = [xs[i+1]-xs[i] for i in range(len(xs)-1)] if len(xs)>1 else []
    cell_w = int(statistics.median(diffs)) if diffs else 16
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
        if ys[i] - ys[i-1] > 6:
            runs += 1
            best = max(best, runs)
        else:
            # y差距太小视作同格，忽略
            pass
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

    def has_duolian_two(color_cols):
        if len(color_cols) < 2: return False
        cols_with_x = []
        for col in color_cols:
            xs = [p[0] for p in col]
            cols_with_x.append((statistics.mean(xs), col))
        cols_with_x.sort(key=lambda t:t[0])
        for i in range(len(cols_with_x)-1):
            if longest_run_in_col(cols_with_x[i][1])>=4 and longest_run_in_col(cols_with_x[i+1][1])>=4:
                return True
        return False

    has_duolian = has_duolian_two(long4_cols_R) or has_duolian_two(long4_cols_B)

    if all_cols:
        single_jump_cols = sum(1 for c in all_cols if longest_run_in_col(c) <= 1)
        single_jump_ratio = single_jump_cols / len(all_cols)
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

# ---------------- Playwright navigation & capture ----------------
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
                    total = 420; step = 40
                    for dx in range(0, total, step):
                        page.mouse.move(box["x"]+box["width"]/2+dx, box["y"]+box["height"]/2, steps=2)
                        time.sleep(0.05)
                    page.mouse.up()
                    time.sleep(1.0)
        except Exception:
            continue

def capture_table_canvases(page):
    images = []
    try:
        page.mouse.wheel(0, 400)
        time.sleep(0.4)
    except Exception:
        pass
    # 尝试抓 canvas 元素
    canvases = page.query_selector_all("canvas")
    for c in canvases:
        try:
            box = c.bounding_box()
            if not box: continue
            if 100 <= box["width"] <= 900 and 60 <= box["height"] <= 600:
                img_bytes = c.screenshot()
                images.append(img_bytes)
        except Exception:
            continue
    # 如果没 canvas，也尝试抓 road 图像/截图方式（兜底）
    if not images:
        try:
            screenshot = page.screenshot(full_page=True)
            images.append(screenshot)
        except Exception:
            pass
    return images

def enter_dg_and_get_tables():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(viewport={"width": 1280, "height": 2600})
        page = context.new_page()
        canvases_bytes = []
        for url in DG_URLS:
            try:
                page.goto(url, timeout=30000)
                # 点击 Free / 免费试玩（尝试不同选择器）
                clicked = False
                for sel in ["text=Free", "text=免费试玩", "text=FREE", "text=free", "button:has-text('免费试玩')", "a:has-text('免费试玩')"]:
                    try:
                        page.click(sel, timeout=3000)
                        clicked = True
                        break
                    except Exception:
                        continue
                # 等待并跳转到可能的新页
                for _ in range(12):
                    time.sleep(0.4)
                    if len(context.pages) > 1:
                        page = context.pages[-1]
                        break
                # 尝试处理滑块
                solve_slider_if_any(page)
                time.sleep(1.2)
                canvases_bytes = capture_table_canvases(page)
                if canvases_bytes:
                    browser.close()
                    return canvases_bytes
            except Exception:
                continue
        browser.close()
        return canvases_bytes

# ---------------- Classification logic (your rules) ----------------
def classify_overall(table_stats):
    n = len(table_stats)
    # long4 effective = has_long4 且 single_jump_ratio < 0.7
    long4_tables = sum(1 for t in table_stats if t["has_long4"] and t["single_jump_ratio"] < 0.7)
    long8_tables = sum(1 for t in table_stats if t["has_long8"])
    long10_tables = sum(1 for t in table_stats if t["has_long10"])
    duolian_tables = sum(1 for t in table_stats if t["has_duolian"])
    many_single_jump = sum(1 for t in table_stats if t["single_jump_ratio"] >= 0.7)

    long8_only = max(0, long8_tables - long10_tables)
    trigger_super = (long10_tables >= 1 and long8_only >= 2)

    cond_full = (n >= 20 and long4_tables >= 8) or (n >= 10 and long4_tables >= 4)

    cond_mid_up = False
    if ((n >= 20 and long4_tables >= 6) or (n >= 10 and long4_tables >= 3)) \
       and ((long8_tables + long10_tables) >= 2) and (duolian_tables >= 1):
        cond_mid_up = True

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

# ---------------- State & ETA estimation ----------------
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

def estimate_eta(history, threshold_active):
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
    t_cross = (threshold_active - c)/m
    now0 = h[0]["ts_ms"]
    eta_ms = now0 + int(t_cross*60000)
    mins_left = max(1, int((eta_ms - now_ms())/60000))
    eta_dt = datetime.fromtimestamp(eta_ms/1000, TZ)
    return eta_dt, mins_left

# ---------------- Main ----------------
def main():
    st = load_state()
    try:
        canvases = enter_dg_and_get_tables()
    except Exception as e:
        canvases = []
        err = f"Exception during navigation: {e}\\n{traceback.format_exc()}"
        send_telegram(f"⚠️ DG 导航异常：可能被网站限制或结构变化。\n{err[:1500]}")
    table_stats = []
    try:
        if canvases:
            for img_bytes in canvases:
                bgr = bytes_to_bgr(img_bytes)
                stat = analyze_table_image(bgr)
                table_stats.append(stat)
        # classification
        if table_stats:
            status, detail = classify_overall(table_stats)
            brief = f"桌数:{detail['tables']} | 长连≥4:{detail['long4_tables']} | 长龙≥8:{detail['long8_tables']} | 超长龙≥10:{detail['long10_tables']} | 多连:{detail['duolian_tables']}"
            # update history metric for ETA (use long4_tables)
            history = st.get("history", [])
            history.append({"ts_ms": now_ms(), "metric": detail["long4_tables"]})
            history = history[-24:]
            st["history"] = history

            last_status = st.get("status")
            msg = None
            if status in ("FANGSHUI", "MID_UP"):
                if last_status not in ("FANGSHUI", "MID_UP"):
                    st["active_since"] = now_ms()
                    st["active_type"] = status
                    eta_dt, mins_left = estimate_eta(history, detail["long4_tables"])
                    if eta_dt and mins_left:
                        msg = (f"✅ {'放水时段（胜率提高）' if status=='FANGSHUI' else '中等胜率（中上）'} 已开始\\n{brief}\\n预计结束时间：{eta_dt.strftime('%H:%M')}（马来西亚时间）\\n预计剩余：{mins_left} 分钟")
                    else:
                        msg = (f"✅ {'放水时段（胜率提高）' if status=='FANGSHUI' else '中等胜率（中上）'} 已开始\\n{brief}\\n预计结束时间：暂无法可靠预估（趋势未显著下降）")
                else:
                    # 仍处于放水/中上，发送简短心跳避免沉默（每 N 次可发，默认仅在开始时发）
                    msg = None
            else:
                # 非活跃
                if last_status in ("FANGSHUI", "MID_UP") and st.get("active_since"):
                    dur_min = max(1, int((now_ms() - st["active_since"]) / 60000))
                    msg = f"🔔 放水已结束，共持续 {dur_min} 分钟。\\n{brief}"
                    st["active_since"] = None
                    st["active_type"] = None
                else:
                    # 检测成功但无放水 — 发送状态心跳，保证你不会长时间无消息
                    msg = f"ℹ️ 检测完成：目前无放水迹象。\\n{brief}"

            st["status"] = status
            save_state(st)
            if msg:
                send_telegram(f"{datetime.now(TZ).strftime('%Y-%m-%d %H:%M')}（马来西亚时间）\\n{msg}")
        else:
            # 无 canvases - 代表可能无法正确抓取页面（也发送错误提示）
            send_telegram(f"⚠️ 检测失败：未抓取到桌面画面（canvas），可能需要更新脚本或网站启用反爬。")
    except Exception as e:
        save_state(st)
        err = f"运行异常：{e}\\n{traceback.format_exc()}"
        send_telegram(f"⚠️ 脚本异常：{str(e)[:800]}")
        with open("last_error.txt", "w", encoding="utf-8") as f:
            f.write(err)

if __name__ == '__main__':
    main()
