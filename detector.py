# detector.py
# -*- coding: utf-8 -*-
"""
DG Baccarat 放水检测器
说明：
- 使用 Playwright 访问 DG 平台并自动进入 Free（免费试玩）
- 截图整页并尝试定位每张桌子区域（若页面 DOM 能直接定位则优先用 DOM）
- 使用 OpenCV 识别红/蓝圈，重构每桌的格子布局并判断连数
- 判定放水 / 中等胜率（中上） / 其它，并通过 Telegram 通知
- 使用仓库内 state.json 保存放水状态与开始时间，便于跨次 run 追踪持续时间
"""

import os, sys, json, time, math, datetime, io
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import requests

# Playwright
from playwright.sync_api import sync_playwright

# ========== 配置区（可根据你的需求改动） ==========
DG_URLS = ["https://dg18.co/", "https://dg18.co/wap/"]
# Telegram (我们会在 workflow 里注入这两个环境变量)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# 保存截图、状态文件路径
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
STATE_FILE = Path("state.json")

# 判定阈值（按你规定）
LONG_CHAIN = 4      # ≥4 粒 = 长连
DRAGON = 8          # ≥8 粒 = 长龙
SUPER_DRAGON = 10   # ≥10 粒 = 超长龙

# 全局占比阈值：若 >=50% 桌子符合长连/长龙 则判定为放水
PERCENT_THRESHOLD = 0.5

# 超长龙 + 至少 2 张长龙 的组合也判为放水
# 最低需要 >=3 张 桌子出长龙/超长龙 才成立（你指定）
MIN_DRAGON_TABLES = 3

# 每张桌最少像素尺寸阈值（防止误判）
MIN_TABLE_W = 200
MIN_TABLE_H = 80

# Playwright 访问超时（秒）
PAGE_TIMEOUT = 30_000

# ========== 工具函数 ==========
def send_telegram_message(text, img_bytes=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/Chat ID 未设置，跳过发消息。")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=15)
    except Exception as e:
        print("Telegram send error:", e)
        return
    if img_bytes:
        files_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("screenshot.jpg", img_bytes)}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": text}
        try:
            requests.post(files_url, files=files, data=data, timeout=20)
        except Exception as e:
            print("Telegram photo send error:", e)

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except:
            return {}
    return {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

def pretty_time(ts=None):
    if ts is None:
        ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ========== 图像处理/识别相关（核心） ==========
def detect_red_blue_circles(img_bgr):
    """
    输入 BGR 图像（单张桌子区域）
    输出：list of circles with (cx, cy, radius, color) ； color: 'B' 或 'P'（蓝/庄）
    原理：
    - 转 HSV 做颜色分割（红、蓝）
    - 轮廓检测并返回中心位置
    """
    circles = []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 蓝色范围（大致）
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 红色范围（红有两个区段）
    lower_red1 = np.array([0, 80, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 60])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # 找轮廓
    def find_centers(mask, color_label):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 20:  # 忽略很小噪音（可调）
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            out.append((int(x), int(y), int(r), color_label))
        return out

    circles += find_centers(mask_blue, 'P')  # P = 闲 (blue)
    circles += find_centers(mask_red, 'B')   # B = 庄 (red)
    return circles

def cluster_to_grid(circles):
    """
    将检测到的圆点按 x 分组为列，按 y 分组为行（栅格化）
    返回：grid: dict[row_idx][col_idx] = color
    说明：实际应用可能需要根据截图尺寸微调分组阈值
    """
    if not circles:
        return {}
    xs = sorted([c[0] for c in circles])
    ys = sorted([c[1] for c in circles])
    # 聚类分列：使用间距聚类
    def cluster_positions(pos_list, gap_ratio=0.6):
        # 基于相邻差值聚类
        diffs = [pos_list[i+1]-pos_list[i] for i in range(len(pos_list)-1)]
        if not diffs:
            return [pos_list]
        median_gap = np.median([d for d in diffs if d>0]) if diffs else 10
        groups = []
        current = [pos_list[0]]
        for i, d in enumerate(diffs):
            if d > median_gap * (1+gap_ratio):  # 新组起点（可调）
                groups.append(current)
                current = [pos_list[i+1]]
            else:
                current.append(pos_list[i+1])
        groups.append(current)
        return groups

    x_groups = cluster_positions(xs, gap_ratio=0.6)
    y_groups = cluster_positions(ys, gap_ratio=0.6)
    # 使用中位数作为列/行的代表坐标
    col_coords = [int(np.median(g)) for g in x_groups]
    row_coords = [int(np.median(g)) for g in y_groups]

    # 建格子：根据每个圆的坐标映射到最近的 row/col
    grid = {}
    for (x,y,r,color) in circles:
        # 找最近列
        col = min(range(len(col_coords)), key=lambda i: abs(col_coords[i]-x))
        row = min(range(len(row_coords)), key=lambda j: abs(row_coords[j]-y))
        grid.setdefault(row, {})[col] = color
    return grid

def analyze_grid_sequences(grid):
    """
    从 grid（row->col->color）分析连数。
    规则：同一列自上而下为连续开（你指定：同排连续为主）。
    我们转成按列读取的序列（第一列->第二列...），每列内从上到下为先后顺序。
    返回：
      - sequences: list of segments (color, length) 按出现顺序
      - max_run: 当前最大连续同色数量
    """
    if not grid:
        return [], 0
    # 把列索引排序，按列拼接
    cols = sorted({c for row in grid.values() for c in row.keys()})
    # 建立按列的“列内最大高度”以便按列读取
    max_row = max(grid.keys())
    # 读取序列：按列，从上(row0)到下(rowN)读取每个格子（若无格子则跳过）
    seq = []
    last_color = None
    run = 0
    max_run = 0
    for col in cols:
        # 读取这一列上到下
        rows_sorted = sorted(grid.keys())
        for row in rows_sorted:
            color = grid.get(row, {}).get(col)
            if color is None:
                continue
            if last_color is None:
                last_color = color
                run = 1
            else:
                if color == last_color:
                    run += 1
                else:
                    seq.append((last_color, run))
                    if run > max_run: max_run = run
                    last_color = color
                    run = 1
    if last_color is not None:
        seq.append((last_color, run))
        if run > max_run: max_run = run
    return seq, max_run

def classify_table_by_sequence(seq, max_run):
    """
    根据序列与最大连续数来分类桌子类型（返回一个字符串）
    按你定义的术语：单跳/双跳/长连/长龙/超长龙/多连/断连开单 等
    注：这里仅给一个简化分类，后面判定整组局势会根据这些结果组合判定
    """
    # max_run 定义
    if max_run >= SUPER_DRAGON:
        return "超长龙"
    if max_run >= DRAGON:
        return "长龙"
    if max_run >= LONG_CHAIN:
        return "长连"
    # 统计单粒占比
    single_count = sum(1 for (c,l) in seq if l==1)
    double_count = sum(1 for (c,l) in seq if 2<=l<=3)
    if single_count >= len(seq)*0.7:
        return "单跳"
    if double_count >= len(seq)*0.5:
        return "双跳"
    return "杂"

# ========== 页面操作 / 截图 / 表格定位 ==========
def attempt_detect_tables_and_analyze(page):
    """
    尝试两种策略定位桌子：
      1) 利用页面 DOM（若能找到每张桌子的容器），直接截图每张桌子
      2) 若 DOM 无法定位，截图整页并通过视觉分割（按网格或模板）来切割各桌子（后者更脆弱，需要调节）
    返回：
      - summary: [{'table_id': id, 'type': classification, 'max_run': n, 'seq': seq}, ...]
      - full_screenshot_bytes
    """
    # 截整页
    full_path = OUT_DIR / "fullpage.png"
    page.screenshot(path=str(full_path), full_page=True)
    full_img = cv2.imread(str(full_path))
    h, w = full_img.shape[:2]

    # 尝试 DOM 定位：常见桌子容器带有类名 'table' 或 'baccarat'
    table_images = []
    try:
        # 试图抓每个可视的 table DOM 并截取区域
        # 优先使用 page.query_selector_all 找到可能的 table 容器
        candidates = page.query_selector_all("div")
        boxes = []
        for el in candidates:
            try:
                box = el.bounding_box()
                if not box:
                    continue
                x,y,width,height = box['x'], box['y'], box['width'], box['height']
                # 过滤宽高太小或太大的元素
                if width < MIN_TABLE_W or height < MIN_TABLE_H or width > w*0.9 or height > h*0.9:
                    continue
                # 过滤可能不是桌子但可能是
                boxes.append((int(x), int(y), int(width), int(height)))
            except:
                continue
        # 简单去重（合并相近的框）
        boxes_sorted = sorted(boxes, key=lambda b:(b[1], b[0]))
        merged = []
        for b in boxes_sorted:
            if not merged:
                merged.append(b)
            else:
                last = merged[-1]
                # 若重叠/非常接近则合并
                if abs(b[0]-last[0])<20 and abs(b[1]-last[1])<20:
                    nx = min(last[0], b[0])
                    ny = min(last[1], b[1])
                    nw = max(last[0]+last[2], b[0]+b[2]) - nx
                    nh = max(last[1]+last[3], b[1]+b[3]) - ny
                    merged[-1] = (nx, ny, nw, nh)
                else:
                    merged.append(b)
        # 把每个候选框裁图并做检测
        table_id = 0
        summary = []
        for (x,y,width,height) in merged:
            # 裁图
            crop = full_img[y:y+height, x:x+width]
            circles = detect_red_blue_circles(crop)
            grid = cluster_to_grid(circles)
            seq, max_run = analyze_grid_sequences(grid)
            classification = classify_table_by_sequence(seq, max_run)
            summary.append({
                'table_id': f"auto-{table_id}",
                'bbox': (x,y,width,height),
                'classification': classification,
                'max_run': int(max_run),
                'seq': seq
            })
            table_id += 1
        # 若 merged 为空，则改用视觉切割（下）
        if not summary:
            raise Exception("DOM 未定位到合适桌子，改用视觉分割。")
        # 读回截图 bytes
        _, buf = cv2.imencode('.jpg', full_img)
        return summary, buf.tobytes()
    except Exception as e:
        print("DOM 定位失败/或未找到有效桌子，尝试视觉切割：", e)

    # ---------- 视觉切割（保底） ----------
    # 我们尝试把页面按常见的多列多行网格切割（例如 3 列或 4 列），并分析每个切片
    possible_cols = [2,3,4]
    summary = []
    _, buf = cv2.imencode('.jpg', full_img)
    # 遍历假设的列数，按行列切割
    for cols in possible_cols:
        slice_w = w//cols
        rows = int(math.ceil(h / 180))  # 假设每张桌高约 180 像素（可调）
        for r in range(rows):
            for c in range(cols):
                x0 = c*slice_w
                y0 = r*180
                x1 = min(x0+slice_w, w)
                y1 = min(y0+180, h)
                if x1-x0 < MIN_TABLE_W or y1-y0 < MIN_TABLE_H:
                    continue
                crop = full_img[y0:y1, x0:x1]
                circles = detect_red_blue_circles(crop)
                if not circles:
                    continue
                grid = cluster_to_grid(circles)
                seq, max_run = analyze_grid_sequences(grid)
                classification = classify_table_by_sequence(seq, max_run)
                summary.append({
                    'table_id': f"grid-{r}-{c}",
                    'bbox': (x0,y0,x1-x0,y1-y0),
                    'classification': classification,
                    'max_run': int(max_run),
                    'seq': seq
                })
    return summary, buf.tobytes()

# ========== 全局判定逻辑（按你规则实现） ==========
def classify_overall(summary):
    """
    summary: list of table dicts {'classification':..., 'max_run': ...}
    返回 overall_status: one of ['放水', '中等胜率(中上)', '胜率中等', '收割']
    以及用于 Telegram 的说明
    """
    total = len(summary)
    if total == 0:
        return "无桌", "未检测到桌子"

    long_like = sum(1 for t in summary if t['classification'] in ("长连","长龙","超长龙"))
    dragon_count = sum(1 for t in summary if t['classification'] in ("长龙","超长龙"))
    super_dragon_count = sum(1 for t in summary if t['classification']=="超长龙")

    pct_long_like = long_like / total

    # 规则1：满桌长连/长龙类型 >=50%
    if pct_long_like >= PERCENT_THRESHOLD and long_like >= 1:
        return "放水", f"满桌长连/长龙占比 {pct_long_like*100:.0f}%（>=50%）"

    # 规则2：1 张超长龙 + 至少 2 张长龙
    if super_dragon_count >= 1 and dragon_count >= 3:  # dragon_count 包含 super 的情况已统计
        return "放水", f"超长龙 {super_dragon_count} 张，长龙合计 {dragon_count} 张"

    # 中等胜率（中上）：部分桌有多连/长龙，但不满足放水占比 50%，且总体仍有中等特征
    # 我们用：若有 >=2 张（长龙或超长龙）且 pct_long_like >= 0.2（可微调）判为中等胜率(中上)
    if dragon_count >= 2 and pct_long_like >= 0.2:
        return "中等胜率(中上)", f"检测到 {dragon_count} 张长龙/超长龙，长连占比 {pct_long_like*100:.0f}%"

    # 胜率中等：桌面多数空，单跳/双跳占比高
    single_like = sum(1 for t in summary if t['classification']=="单跳" or t['classification']=="双跳" or t['classification']=="杂")
    if single_like / total >= 0.6:
        return "胜率中等", f"单跳/双跳/杂占比 {single_like/total*100:.0f}%（>=60%）"

    # 默认视为收割（胜率调低）
    return "收割", f"未满足放水或中等上条件，长连占比 {pct_long_like*100:.0f}%"

# ========== 主流程 ==========
def main():
    state = load_state()  # 读取历史状态（是否正在放水）
    last_status = state.get("status")
    last_start_ts = state.get("start_ts")
    # Playwright 启动并进入页面
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1366, "height":900})
        page = context.new_page()
        # 依列表尝试打开可用 URL
        opened = False
        for url in DG_URLS:
            try:
                page.goto(url, timeout=PAGE_TIMEOUT)
                opened = True
                break
            except Exception as e:
                print("打开 URL 失败:", url, e)
        if not opened:
            print("无法访问 DG 网站")
            return

        # 等页面加载并尝试点击 Free / 免费试玩 的按钮（同时模拟滑动安全条）
        # 优化：尝试寻找带有 “Free” 或 “免费试玩” 文本的按钮
        try:
            # 如果页面含有 iframe，需要先切入 iframe（部分 DG 版本）
            time.sleep(2)
            # 尝试点击匹配文本
            try:
                page.get_by_text("Free").click(timeout=3000)
            except:
                try:
                    page.get_by_text("免费试玩").click(timeout=3000)
                except:
                    # 若无法直接点击，则尝试点击首个可能为 table 区域的按钮
                    print("未找到 Free/免费试玩 按钮，继续尝试滑动页面。。。")
            # 页面会跳转/弹出，需要等待并模拟安全滑块
            time.sleep(2)
            # 尝试查找滑动安全条并模拟拖拽（常见为 div.slider 等）
            try:
                slider = page.query_selector("div[role='slider'], .slider, .drag, .slide")
                if slider:
                    box = slider.bounding_box()
                    if box:
                        sx = box['x'] + 5
                        sy = box['y'] + box['height']/2
                        ex = box['x'] + box['width'] - 5
                        page.mouse.move(sx, sy)
                        page.mouse.down()
                        page.mouse.move(ex, sy, steps=15)
                        page.mouse.up()
                        time.sleep(2)
            except Exception as e:
                print("滑动安全条尝试失败或不存在：", e)
        except Exception as e:
            print("点击 Free 失败：", e)

        # 等待一会，让页面形成桌面
        time.sleep(2)
        # 尝试检测各桌并分析
        summary, screenshot_bytes = attempt_detect_tables_and_analyze(page)

        # overall classify
        status, reason = classify_overall(summary)

        # 当依据是放水或中等胜率(中上) 时要提醒
        notify_now = False
        notify_level = None
        if status == "放水":
            notify_now = True
            notify_level = "放水"
        elif status == "中等胜率(中上)":
            notify_now = True
            notify_level = "中等胜率(中上)"

        now_ts = time.time()
        # 如果当前为放水并且之前并未处于放水状态 -> 标记开始
        if notify_now:
            if last_status != notify_level:
                # 新的放水开始
                state['status'] = notify_level
                state['start_ts'] = now_ts
                save_state(state)
                # 发送 Telegram：放水开始
                text = f"【{notify_level} 开始】\n时间：{pretty_time(now_ts)} (MYT)\n判定：{reason}\n总桌数：{len(summary)}\n满足长连/长龙表数：{sum(1 for t in summary if t['classification'] in ('长连','长龙','超长龙'))}\n说明：若符合状况A（断连开单）会列出建议桌子。\n"
                # 找出可能的「状况A」桌子（max_run >= LONG_CHAIN 且序列符合断连开单模式）
                candidates = [t for t in summary if t['max_run'] >= LONG_CHAIN]
                if candidates:
                    text += "可能可入场的桌子（max_run）：\n"
                    for t in candidates[:8]:
                        text += f" - {t['table_id']} : {t['classification']} (max {t['max_run']})\n"
                send_telegram_message(text, img_bytes=screenshot_bytes)
            else:
                # 已处于放水中，仍然维持放水：我们可选择不每次都通知，或每 N 次发送一次（此处默认不重复发送）
                print("仍在放水中，已通知过；本次不重复发送。")
        else:
            # 当前非放水，若之前处于放水 -> 放水结束
            if last_status in ("放水", "中等胜率(中上)"):
                start_ts = last_start_ts
                if not start_ts:
                    start_ts = state.get("start_ts", now_ts)
                duration_min = int((now_ts - start_ts)/60)
                text = f"【放水已结束】\n结束时间：{pretty_time(now_ts)} (MYT)\n放水开始时间：{pretty_time(start_ts)}\n持续时长：{duration_min} 分钟\n本次结束判定理由：{reason}\n"
                send_telegram_message(text, img_bytes=screenshot_bytes)
                # 重置状态
                state['status'] = "空"
                state['start_ts'] = None
                save_state(state)
            else:
                print("当前不处于放水，且此前也未处于放水状态。无需通知。")

        # 退出浏览器
        browser.close()

if __name__ == "__main__":
    main()
