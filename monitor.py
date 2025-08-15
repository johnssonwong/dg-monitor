# -*- coding: utf-8 -*-
"""
DreamGaming(DG) 自动检测 + 放水提醒 (Telegram)
- 依据你提供的全部判定/术语/阈值与“提醒机制”
- 每5分钟运行（由 GitHub Actions 触发）
- 马来西亚时区
- 自动记录放水开始/结束，并估算“预计结束时间/剩余分钟”（基于近几次检测趋势）
"""

import os, io, json, math, time, statistics, traceback
from datetime import datetime, timedelta
import pytz
import requests
import numpy as np
import cv2

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ============ 你的Telegram信息（支持Secrets覆盖） ============
TG_TOKEN   = os.getenv("TG_TOKEN") or "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TG_CHAT_ID = os.getenv("TG_CHAT_ID") or "485427847"

# ============ DG入口 ============
DG_URLS = ["https://dg18.co/wap/", "https://dg18.co/"]

# ============ 时区 ============
TZ = pytz.timezone("Asia/Kuala_Lumpur")

# ============ 状态文件（用来记住“放水开始/结束/历史趋势”） ============
STATE_FILE = "state.json"

# ============ 术语&策略（来自你的定义，核心用于分类） ============
# 同排纵向连续 >=4 => 长连
# 同排纵向连续 >=8 => 长龙
# 同排纵向连续 >=10 => 超长龙
# 多连/连珠：一排>=4之后，下一排再>=4（且同色同边）
# 特判：连续3次“单跳”不计入放水判定

def now_ms():
    return int(datetime.now(TZ).timestamp() * 1000)

def fmt_time(ts_ms):
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

# ---------------- CV部分：从桌子小图上识别红/蓝圈点并抽出“列” ----------------

def bytes_to_bgr(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def find_red_blue_points(bgr):
    """以颜色阈值+霍夫圆检测近似找出红/蓝圆点的中心。"""
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
    """按x坐标聚类成列。返回：list[ list[(x,y)] ]，列内按y升序。"""
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
    # 列内按y排序
    cols = [sorted(c, key=lambda p: p[1]) for c in cols]
    return cols

def analyze_table_image(img_bgr):
    """
    返回该桌的统计：
    {
      'has_long4': bool,
      'has_long8': bool,
      'has_long10': bool,
      'has_duolian': bool,  # 多连/连珠
      'single_jump_ratio': float,  # 估算单跳比例
      'marker_count': int
    }
    """
    red_pts, blue_pts = find_red_blue_points(img_bgr)
    marker_count = len(red_pts) + len(blue_pts)

    # 合并两色用于估计单跳比例
    all_cols = cluster_columns(red_pts + blue_pts)
    red_cols = cluster_columns(red_pts)
    blue_cols = cluster_columns(blue_pts)

    # 计算每列的长连
    def longest_run_in_col(col_pts, color_tag):
        # col_pts: list[(x,y)] 排序后视为每格一粒
        # 相邻y差距过小的视为同一格，简化：
        ys = [p[1] for p in col_pts]
        runs = 1 if ys else 0
        best = 1 if ys else 0
        for i in range(1, len(ys)):
            if ys[i] - ys[i-1] > 6:  # y差距>6像素视为下一粒
                runs += 1
                best = max(best, runs)
            else:
                # 太近，忽略
                pass
        return best

    long4_cols_R = [c for c in red_cols if longest_run_in_col(c,"R")>=4]
    long4_cols_B = [c for c in blue_cols if longest_run_in_col(c,"B")>=4]

    long8_R = any(longest_run_in_col(c,"R")>=8 for c in red_cols)
    long8_B = any(longest_run_in_col(c,"B")>=8 for c in blue_cols)
    long10_R = any(longest_run_in_col(c,"R")>=10 for c in red_cols)
    long10_B = any(longest_run_in_col(c,"B")>=10 for c in blue_cols)

    has_long4 = (len(long4_cols_R)+len(long4_cols_B))>0
    has_long8 = (long8_R or long8_B)
    has_long10 = (long10_R or long10_B)

    # 多连/连珠：相邻两列同色均>=4
    def has_duolian_two(color_cols):
        # 判定相邻列是否都>=4。用列的x均值判断相邻。
        if len(color_cols) < 2:
            return False
        # 计算每列x均值
        cols_with_x = []
        for col in color_cols:
            xs = [p[0] for p in col]
            cols_with_x.append( (statistics.mean(xs), col) )
        cols_with_x.sort(key=lambda t:t[0])
        # 检查相邻
        for i in range(len(cols_with_x)-1):
            if longest_run_in_col(cols_with_x[i][1],"")>=4 and longest_run_in_col(cols_with_x[i+1][1],"")>=4:
                return True
        return False

    has_duolian = has_duolian_two(long4_cols_R) or has_duolian_two(long4_cols_B)

    # 单跳比例（粗估）：取所有列中，列内“最长连”为1的列占比
    def longest_run_len(col):
        ys = [p[1] for p in col]
        if not ys: return 0
        # 不分颜色，近似估长连长度
        runs, best = 1, 1
        for i in range(1, len(ys)):
            if ys[i] - ys[i-1] > 6:
                runs += 1
                best = max(best, runs)
        return best
    if all_cols:
        single_jump_cols = sum(1 for c in all_cols if longest_run_len(c)<=1)
        single_jump_ratio = single_jump_cols/len(all_cols)
    else:
        single_jump_ratio = 1.0  # 没有数据当作极差

    return {
        "has_long4": has_long4,
        "has_long8": has_long8,
        "has_long10": has_long10,
        "has_duolian": has_duolian,
        "single_jump_ratio": float(single_jump_ratio),
        "marker_count": int(marker_count),
    }

# ---------------- Playwright 部分：进入DG并截取各桌大路canvas ----------------

def solve_slider_if_any(page):
    """尝试处理‘滚动安全条/滑块’。不同站点实现不同，这里做几种通用尝试。"""
    # 常见滑块类名（尽量通用）
    candidates = [
        "div.geetest_slider_button", "div.nc_iconfont.btn_slide", "div.slider", "div#nc_1_n1z",
        "div.yidun_slider", "div.verify-slider", "div.slider-btn", "span.btn_slide", "div.slider_button"
    ]
    for sel in candidates:
        try:
            btn = page.query_selector(sel)
            if btn:
                box = btn.bounding_box()
                bar = btn.evaluate_handle("e => e.parentElement")
                if box:
                    page.mouse.move(box["x"]+box["width"]/2, box["y"]+box["height"]/2)
                    page.mouse.down()
                    # 往右拖很长，分段拖动更像真人
                    total = 400
                    step = 40
                    for dx in range(0, total, step):
                        page.mouse.move(box["x"]+box["width"]/2+dx, box["y"]+box["height"]/2, steps=2)
                        time.sleep(0.05)
                    page.mouse.up()
                    time.sleep(1.0)
        except Exception:
            pass

def capture_table_canvases(context, page):
    """
    返回本屏幕内疑似“桌面大路canvas”的图片（bytes）列表。
    我们用‘canvas元素’并按尺寸过滤。
    """
    images = []
    # 尝试滚一点，加载更多
    try:
        page.mouse.wheel(0, 300)
        time.sleep(0.3)
    except Exception:
        pass

    canvases = page.query_selector_all("canvas")
    for c in canvases:
        try:
            box = c.bounding_box()
            if not box: 
                continue
            # 过滤太小/太大，经验阈值
            if 120 <= box["width"] <= 420 and 90 <= box["height"] <= 250:
                img_bytes = c.screenshot()
                images.append(img_bytes)
        except Exception:
            continue
    return images

def enter_dg_and_get_tables():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True,
                                    args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(viewport={"width": 1440, "height": 2800})
        page = context.new_page()

        ok = False
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
                    # 有些是按钮
                    buttons = page.query_selector_all("button")
                    for b in buttons:
                        txt = (b.inner_text() or "").strip()
                        if "Free" in txt or "免费试玩" in txt:
                            b.click(timeout=2000)
                            clicked = True
                            break

                # 可能会打开新页
                for _ in range(15):
                    time.sleep(0.5)
                    if len(context.pages) > 1:
                        page = context.pages[-1]
                        break

                # 处理滑块/安全条
                solve_slider_if_any(page)

                # 此时应在大厅或选场页面，等待canvas渲染
                page.wait_for_timeout(2000)
                canvases = capture_table_canvases(context, page)
                if canvases:
                    ok = True
                    browser.close()
                    return canvases
            except Exception:
                continue

        browser.close()
        return []

# ---------------- 全局局势判定（依据你的规则） ----------------

def classify_overall(table_stats):
    """
    根据所有桌的统计，输出 overall 分类：
    - "FANGSHUI" (放水时段 / 胜率提高)  -> 必须提醒
    - "MID_UP"   (中等胜率-中上)        -> 小提醒
    - "MID"      (胜率中等)              -> 不提醒
    - "HARVEST"  (收割时段)              -> 不提醒
    并返回判据详情（计数等）。
    """
    n = len(table_stats)
    long4_tables = sum(1 for t in table_stats if t["has_long4"])
    long8_tables = sum(1 for t in table_stats if t["has_long8"])
    long10_tables = sum(1 for t in table_stats if t["has_long10"])
    duolian_tables = sum(1 for t in table_stats if t["has_duolian"])

    # “忽略持续3次单跳不计入放水判定” —— 我们用一个粗指标：单跳很高的桌子不计入长连统计
    effective_long4 = sum(1 for t in table_stats if t["has_long4"] and t["single_jump_ratio"] < 0.7)

    # 超长龙触发型：≥1超长龙 + 另外≥2长龙（不含那条超长龙）
    long8_only = max(0, long8_tables - long10_tables)  # 扣掉超长龙
    trigger_super = (long10_tables >= 1 and long8_only >= 2)

    # 满盘长连局势型：整体密集 & 长连桌足够多
    # 阈值（来自你给的：20桌≥8张、10桌≥4张）
    cond_full = False
    if n >= 20 and effective_long4 >= 8:
        cond_full = True
    elif n >= 10 and effective_long4 >= 4:
        cond_full = True

    # 中等胜率（中上）
    # - 20桌≥6张 或 10桌≥3张
    # - 至少2桌 长龙/超长龙
    # - 有若干多连/连珠
    cond_mid_up = False
    lng2 = (long8_tables + long10_tables) >= 2
    if (n >= 20 and effective_long4 >= 6) or (n >= 10 and effective_long4 >= 3):
        if lng2 and duolian_tables >= 1:
            cond_mid_up = True

    # 收割时段：绝大多数空荡/单跳多/几乎无连
    many_single_jump = sum(1 for t in table_stats if t["single_jump_ratio"] >= 0.7)
    cond_harvest = (long8_tables < 2 and many_single_jump >= max(3, int(0.5*n)))

    # 判定优先级（放水 > 中上 > 中等 > 收割）
    detail = {
        "tables": n,
        "long4_tables": effective_long4,
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

# ---------------- 放水“持续时长 & 预计结束时间”计算 ----------------

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
    """
    用最近几次（最多6次）活跃度指标来线性预估ETA。
    history: list of dict {ts_ms, metric}，metric用“有效长连桌数”等。
    threshold_active: 进入放水的阈值（这里用 long4_tables 门槛）
    返回 (eta_dt, minutes_left) 或 (None, None)
    """
    if len(history) < 3:
        return None, None
    h = history[-6:]
    xs = np.array([(h[i]["ts_ms"] - h[0]["ts_ms"])/60000.0 for i in range(len(h))])  # 分钟
    ys = np.array([h[i]["metric"] for i in range(len(h))], dtype=float)
    # 简单双变量线性回归
    A = np.vstack([xs, np.ones(len(xs))]).T
    try:
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    except Exception:
        return None, None
    # 只在趋势下降(m<0)时预估
    if m >= -1e-6:
        return None, None
    # 求 ys = threshold_active 的时刻
    t_cross = (threshold_active - c)/m  # 分钟
    now0 = h[0]["ts_ms"]
    eta_ms = now0 + int(t_cross*60000)
    eta_dt = datetime.fromtimestamp(eta_ms/1000, TZ)
    mins_left = max(1, int((eta_ms - now_ms())/60000))
    return eta_dt, mins_left

# ---------------- 主流程 ----------------

def main():
    # 1) 进入DG并抓取当前所有桌面的大路小图
    canvases = enter_dg_and_get_tables()

    table_stats = []
    for img_bytes in canvases:
        bgr = bytes_to_bgr(img_bytes)
        stat = analyze_table_image(bgr)
        table_stats.append(stat)

    # 2) 根据你的规则做整体判定
    status, detail = classify_overall(table_stats)

    # 3) 读取/更新状态以控制提醒节流 & 时长统计
    st = load_state()
    # 历史活跃指标：用 effective_long4 = detail["long4_tables"]
    history = st.get("history", [])
    history.append({"ts_ms": now_ms(), "metric": detail["long4_tables"]})
    history = history[-24]  # 保留近24条（约2小时）

    last_status = st.get("status")
    active_since = st.get("active_since")  # ts_ms
    active_type  = st.get("active_type")   # FANGSHUI / MID_UP

    # 4) 判定是否需要发提醒
    #   - 只有 FANGSHUI 或 MID_UP 才提醒
    #   - 状态切换时发“开始”，从活跃转为非活跃时发“结束”
    msg = None

    # 构造简短统计串
    brief = f"桌数:{detail['tables']} | 长连≥4:{detail['long4_tables']} | 长龙≥8:{detail['long8_tables']} | 超长龙≥10:{detail['long10_tables']} | 多连:{detail['duolian_tables']}"

    if status in ("FANGSHUI", "MID_UP"):
        # 进入/继续活跃
        if last_status not in ("FANGSHUI", "MID_UP"):
            # 新开始
            st["active_since"] = now_ms()
            st["active_type"] = status
            # 估ETA
            eta_dt, mins_left = estimate_eta(history, threshold_active=detail["long4_tables"])
            if eta_dt and mins_left:
                msg = (f"【{ '放水时段（胜率提高）' if status=='FANGSHUI' else '中等胜率（中上）' }】已开始\n"
                       f"{brief}\n"
                       f"预计结束时间：{eta_dt.strftime('%H:%M')}（马来西亚时间）\n"
                       f"此局势预计：剩余约 {mins_left} 分钟")
            else:
                msg = (f"【{ '放水时段（胜率提高）' if status=='FANGSHUI' else '中等胜率（中上）' }】已开始\n{brief}\n"
                       f"预计结束时间：暂无法可靠预估（趋势未显著下降）")
        else:
            # 仍在活跃，不重复刷屏；只在结束时再发
            pass
    else:
        # 非活跃（MID/HARVEST）
        if last_status in ("FANGSHUI", "MID_UP") and st.get("active_since"):
            dur_min = max(1, int((now_ms()-st["active_since"])/60000))
            msg = f"【放水已结束】共持续 {dur_min} 分钟\n{brief}"
            st["active_since"] = None
            st["active_type"] = None

    # 保存状态
    st["status"] = status
    st["history"] = history
    save_state(st)

    # 发送提醒（如需要）
    if msg:
        when = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        send_telegram(f"{when}（马来西亚时间）\n{msg}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 避免异常中断后无可见输出
        err = f"运行异常：{e}\n{traceback.format_exc()}"
        # 选发到TG便于你知道哪次出错（可保留/可注释）
        try:
            send_telegram(err[:3500])
        except Exception:
            pass
        # 同时也写入state，方便排查
        with open("last_error.txt", "w", encoding="utf-8") as f:
            f.write(err)
