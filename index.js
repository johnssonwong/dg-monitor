// index.js  — DG monitor 改良版
const fs = require('fs');
const fetch = require('node-fetch');
const Jimp = require('jimp');
const { Octokit } = require('@octokit/rest');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

/* ---------------- Config (可通过 workflow env 覆盖) ---------------- */
const DG_URLS = [
  process.env.DG_URL1 || 'https://dg18.co/wap/',
  process.env.DG_URL2 || 'https://dg18.co/'
];

const MIN_LONG_BOARDS_FOR_POW = parseInt(process.env.MIN_LONG_BOARDS || '3', 10); // 放水判定：最少多少张长龙/超长龙
const MID_LONG_REQ = parseInt(process.env.MID_LONG_REQ || '3', 10); // 中等（中上）所需长龙数（你要求默认3）
const MID_MULTI_BOARDS = parseInt(process.env.MID_MULTI_BOARDS || '3', 10); // 中等（中上）所需“多连/连珠 连续3排”桌数
const COOLDOWN_MIN = parseInt(process.env.COOLDOWN_MIN || '10', 10); // 提醒重复冷却（分钟）
const STATE_PATH = 'state.json';
const HISTORY_MAX = 30; // 保存历史记录条数上限
const TIMEZONE = 'Asia/Kuala_Lumpur'; // Malaysia UTC+8

/* ---------------- Secrets from env (set in GH secrets) ---------------- */
const TG_TOKEN = process.env.TG_BOT_TOKEN;
const TG_CHAT = process.env.TG_CHAT_ID;
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const REPO = process.env.GITHUB_REPOSITORY;

/* ---------------- Octokit for state read/write ---------------- */
const oct = new Octokit({ auth: GITHUB_TOKEN });

async function readState(){
  try {
    const [owner, repo] = REPO.split('/');
    const res = await oct.repos.getContent({ owner, repo, path: STATE_PATH });
    const json = Buffer.from(res.data.content, 'base64').toString();
    return { ...JSON.parse(json), sha: res.data.sha };
  } catch(e){
    // return default
    return { inPow:false, startAt:null, lastAlertAt:0, history:[] };
  }
}

async function writeState(state){
  const [owner, repo] = REPO.split('/');
  const content = Buffer.from(JSON.stringify(state, null, 2)).toString('base64');
  const params = { owner, repo, path: STATE_PATH, message: `update state ${new Date().toISOString()}`, content };
  if(state.sha) params.sha = state.sha;
  const res = await oct.repos.createOrUpdateFileContents(params);
  state.sha = res.data.content.sha;
}

/* ---------------- Telegram helper ---------------- */
async function sendTelegramMsg(text){
  if(!TG_TOKEN || !TG_CHAT){
    console.log('Telegram 未配置 (TG_BOT_TOKEN / TG_CHAT_ID)。');
    return;
  }
  const url = `https://api.telegram.org/bot${TG_TOKEN}/sendMessage`;
  const body = { chat_id: TG_CHAT, text, parse_mode: 'Markdown' };
  try {
    const r = await fetch(url, { method:'POST', body: JSON.stringify(body), headers:{'Content-Type':'application/json'} });
    const j = await r.json();
    console.log('Telegram send:', j.ok ? 'OK' : 'FAIL', j);
    return j;
  } catch(err){
    console.error('Telegram 发送失败：', err);
  }
}

/* ---------------- Image analysis helpers ---------------- */
function colorIsRed(r,g,b){ return r>140 && g<120 && b<120; }
function colorIsBlue(r,g,b){ return b>140 && r<120 && g<120; }

async function analyzeScreenshot(buffer){
  const img = await Jimp.read(buffer);
  const W = img.bitmap.width, H = img.bitmap.height;
  // cell size heuristic: smaller cell for finer detection
  const cell = Math.max(40, Math.floor(Math.min(W,H) / 18));
  const cols = Math.ceil(W / cell), rows = Math.ceil(H / cell);
  const counts = [];

  for(let ry=0; ry<rows; ry++){
    for(let rx=0; rx<cols; rx++){
      let rc=0, bc=0;
      const sx = rx*cell, sy = ry*cell, ex = Math.min(W, sx+cell), ey = Math.min(H, sy+cell);
      for(let y = sy; y < ex ? y < ey : y < ey; y+=2){
        for(let x = sx; x < ex; x+=2){
          const idx = (y*W + x) * 4;
          const r = img.bitmap.data[idx], g = img.bitmap.data[idx+1], b = img.bitmap.data[idx+2];
          if(colorIsRed(r,g,b)) rc++;
          else if(colorIsBlue(r,g,b)) bc++;
        }
      }
      counts.push({rx,ry,rc,bc});
    }
  }

  // find densest cells
  const hits = counts.filter(c=> (c.rc + c.bc) >= 24 ); // adjustable threshold
  // merge hits to regions (tables)
  const regions = [];
  hits.forEach(h=>{
    const x = h.rx*cell, y = h.ry*cell, w = cell, hh = cell;
    let merged=false;
    for(const g of regions){
      if(!(x > g.x + g.w + cell || x + w < g.x - cell || y > g.y + g.h + cell || y + hh < g.y - cell)){
        g.x = Math.min(g.x, x); g.y = Math.min(g.y, y);
        g.w = Math.max(g.w, x+w - g.x); g.h = Math.max(g.h, y+hh - g.y);
        merged=true; break;
      }
    }
    if(!merged) regions.push({x,y,w: w, h: hh});
  });

  const boards = [];
  for(const r of regions){
    const sx = Math.max(0, r.x), sy = Math.max(0, r.y), sw = Math.min(W - sx, r.w), sh = Math.min(H - sy, r.h);
    if(sw < 40 || sh < 40) continue;
    const crop = img.clone().crop(sx, sy, sw, sh);
    // detect bead centers
    const centers = [];
    // sample grid - detect color pixels and cluster into blobs by proximity
    const visited = new Uint8Array(sw * sh);
    for(let y=2; y<sh-2; y+=2){
      for(let x=2; x<sw-2; x+=2){
        const idx = (y*sw + x) * 4;
        const r0 = crop.bitmap.data[idx], g0 = crop.bitmap.data[idx+1], b0 = crop.bitmap.data[idx+2];
        if(colorIsRed(r0,g0,b0) || colorIsBlue(r0,g0,b0)){
          centers.push({x,y,color: colorIsRed(r0,g0,b0)?'B':'P'});
        }
      }
    }
    // cluster centers into columns by x
    centers.sort((a,b)=>a.x - b.x || a.y - b.y);
    const colGroups = [];
    const colGap = Math.max(10, Math.floor(sw/30));
    centers.forEach(c=>{
      let placed=false;
      for(const g of colGroups){
        if(Math.abs(c.x - g.xAvg) <= colGap){
          g.items.push(c);
          g.xAvg = (g.xAvg * (g.items.length-1) + c.x) / g.items.length;
          placed=true; break;
        }
      }
      if(!placed) colGroups.push({xAvg:c.x, items:[c]});
    });
    // for each column produce seq top->bottom
    const sequences = colGroups.map(g=>{
      g.items.sort((a,b)=>a.y - b.y);
      return g.items.map(i=>i.color);
    }).filter(s=>s.length>0);
    // flatten reading order (col by col, top to bottom)
    const flattened=[];
    const maxlen = sequences.reduce((m,s)=>Math.max(m, s.length), 0);
    for(let rrow=0; rrow<maxlen; rrow++){
      for(let c=0;c<sequences.length;c++){
        if(sequences[c][rrow]) flattened.push(sequences[c][rrow]);
      }
    }
    // runs for whole flattened
    const runs=[];
    if(flattened.length>0){
      let cur = {color: flattened[0], len:1};
      for(let i=1;i<flattened.length;i++){
        if(flattened[i] === cur.color) cur.len++;
        else { runs.push(cur); cur={color:flattened[i], len:1}; }
      }
      runs.push(cur);
    }
    // per-column max run (to detect multi连 across adjacent columns)
    const colMaxRuns = sequences.map(seq=>{
      let m=0, c=seq[0]||null, len=0;
      for(let i=0;i<seq.length;i++){
        if(seq[i]===c) len++; else { m=Math.max(m,len); c=seq[i]; len=1; }
      }
      m=Math.max(m,len);
      return m;
    });
    // detect multi连 across adjacent columns: find groups of >=3 adjacent columns where column maxRun >=4
    let multiGroups = 0;
    if(colMaxRuns.length > 0){
      let curCount=0;
      for(let i=0;i<colMaxRuns.length;i++){
        if(colMaxRuns[i] >= 4) { curCount++; }
        else { if(curCount >= 3) multiGroups++; curCount=0; }
      }
      if(curCount >= 3) multiGroups++;
    }
    // board summary
    const maxRun = runs.reduce((m,r)=>Math.max(m, r.len), 0);
    boards.push({
      region: r,
      totalBeads: flattened.length,
      maxRun,
      runs,
      colCount: sequences.length,
      colMaxRuns,
      multiGroups // number of multi连 groups found
    });
  }

  // counts
  const longCount = boards.filter(b=>b.maxRun >= 8).length; // 长龙
  const superCount = boards.filter(b=>b.maxRun >= 10).length; // 超长龙
  // boards with at least one multi-group (>=3 adjacent columns each with col maxRun>=4)
  const multiBoardsCount = boards.filter(b=>b.multiGroups >= 1).length;

  return { boards, longCount, superCount, multiBoardsCount };
}

/* ---------------- Decision & run ---------------- */
function nowInTZ(){
  return new Date().toLocaleString('en-US', { timeZone: TIMEZONE });
}
function isoInTZ(d){
  // d is Date object or ISO string
  const dt = new Date(d);
  // return formatted: YYYY-MM-DD HH:MM (Malaysia)
  const opts = { timeZone: TIMEZONE, year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false };
  const parts = new Intl.DateTimeFormat('en-GB', opts).formatToParts(dt);
  // build "YYYY-MM-DD HH:MM"
  const map = {}; parts.forEach(p=> map[p.type]=p.value);
  return `${map.year}-${map.month}-${map.day} ${map.hour}:${map.minute}`;
}

(async ()=>{
  console.log('DG monitor start @', new Date().toISOString());
  const state = await readState();
  // launch puppeteer
  const browser = await puppeteer.launch({ args:['--no-sandbox','--disable-setuid-sandbox'], headless:true });
  const page = await browser.newPage();
  page.setDefaultTimeout(60000);

  let screenshotBuffer = null;
  try {
    let loaded=false;
    for(const url of DG_URLS){
      try {
        await page.goto(url, { waitUntil: 'networkidle2' });
        // attempt to click Free/免费/试玩
        await page.evaluate(()=>{
          const nodes = Array.from(document.querySelectorAll('a,button,div'));
          for(const n of nodes){
            const t = (n.innerText||n.textContent||'').trim();
            if(/free|免费|试玩|Free/i.test(t)){
              try{ n.click(); }catch(e){}
            }
          }
        }).catch(()=>{});
        // wait a bit for content to load / new window
        await page.waitForTimeout(4000);
        // if new pages opened, pick the latest
        const pages = await browser.pages();
        let target = pages[pages.length-1];
        // try to solve a slider if exists by searching for common elements
        try {
          const sliderCandidates = await target.$$('input[type=range], .slider, .captcha-slider, .drag, .slide-btn, .nc_slider');
          if(sliderCandidates.length > 0){
            const box = await sliderCandidates[0].boundingBox();
            if(box){
              await target.mouse.move(box.x + 5, box.y + box.height/2);
              await target.mouse.down();
              await target.mouse.move(box.x + box.width - 5, box.y + box.height/2, { steps: 12 });
              await target.mouse.up();
              await target.waitForTimeout(1200);
            }
          }
        } catch(e){ /* ignore if not possible */ }
        // wait for board-like content (colored circles). Use timeout but don't throw too early
        await target.waitForTimeout(3000);
        screenshotBuffer = await target.screenshot({ fullPage: true, type: 'png' });
        loaded = true;
        break;
      } catch(e){
        console.warn('load failed for', url, e.message);
        continue;
      }
    }
    if(!screenshotBuffer){
      console.error('无法获取页面截图（可能被防爬虫挡下）');
      await sendTelegramMsg(`⚠️ [DG监测] 无法获取 DG 页面截图（可能被防爬虫阻挡）。请人工检查页面或稍后重试。`);
      await browser.close();
      return;
    }

    const analysis = await analyzeScreenshot(screenshotBuffer);
    console.log('分析结果：longCount=', analysis.longCount, 'superCount=', analysis.superCount, 'multiBoards=', analysis.multiBoardsCount, 'boards=', analysis.boards.length);

    // decision logic (严格化，根据你最新要求)
    let overall = 'unknown';
    // 放水（强提醒）：至少 MIN_LONG_BOARDS_FOR_POW 张长龙/超长龙
    if(analysis.longCount >= MIN_LONG_BOARDS_FOR_POW){
      overall = '放水时段（提高胜率）';
    } else if( (analysis.multiBoardsCount >= MID_MULTI_BOARDS) && (analysis.longCount >= MID_LONG_REQ) ){
      // 中等（中上）判定：满足你要求的 两个条件：至少 MID_MULTI_BOARDS 张有 连续多连（3列）+ 至少 MID_LONG_REQ 张长龙/超长龙
      overall = '中等胜率（中上）';
    } else {
      // 判断收割 / 中等
      const sparse = analysis.boards.filter(b=>b.totalBeads < 6).length;
      if(analysis.boards.length > 0 && sparse >= Math.floor(analysis.boards.length * 0.6)) overall = '胜率调低 / 收割时段';
      else overall = '胜率中等（平台收割中等时段）';
    }

    // read state & apply transitions
    const now = Date.now();
    const lastAlert = state.lastAlertAt || 0;
    const inPow = state.inPow || false;

    if(overall === '放水时段（提高胜率）' || overall === '中等胜率（中上）'){
      // enter or continue pow
      if(!inPow){
        // start pow
        state.inPow = true;
        state.startAt = new Date().toISOString();
        state.lastAlertAt = now;
        if(!Array.isArray(state.history)) state.history = [];
        await writeState(state);
        // send start alert (emoji + details + estimated remaining if available)
        let msg = `🟢 [DG提醒] *${overall}* 已触发！\n`;
        msg += `開始時間: ${isoInTZ(state.startAt)} (${TIMEZONE})\n`;
        msg += `長龍/超長龍數: ${analysis.longCount}，超長龍: ${analysis.superCount}\n`;
        msg += `多連/連珠 (連續3列) 桌數: ${analysis.multiBoardsCount}\n`;
        // estimate remaining based on history average
        if(Array.isArray(state.history) && state.history.length >= 3){
          const mean = Math.round(state.history.reduce((s,v)=>s+v,0) / state.history.length);
          msg += `\n⏳ 歷史平均放水時長: ${mean} 分鐘。估計剩餘: ${Math.max(1, mean - 0)} 分鐘（以歷史為估）；實際以結束判定為準。\n`;
        } else {
          msg += `\n⏳ 尚無足夠歷史，暫無可靠剩餘時間估計；系統將實時監測並在放水結束時通知結束時間與實際持續時長。\n`;
        }
        await sendTelegramMsg(msg);
      } else {
        // already in pow: only notify if cooldown passed
        const cdMs = COOLDOWN_MIN * 60 * 1000;
        if(now - lastAlert >= cdMs){
          state.lastAlertAt = now;
          await writeState(state);
          const msg = `🔁 [DG提醒] *${overall}* 仍在進行。\n目前長龍/超長龍: ${analysis.longCount}，多連桌數: ${analysis.multiBoardsCount}\n（重複提醒）`;
          await sendTelegramMsg(msg);
        } else {
          console.log('仍在冷卻內，跳過通知。');
        }
      }
    } else {
      // not in pow now
      if(inPow){
        // pow just ended => compute duration and push history, send end notice
        const start = new Date(state.startAt);
        const endISO = new Date().toISOString();
        const durationMin = Math.round((Date.now() - start.getTime()) / 60000);
        // push to history
        if(!Array.isArray(state.history)) state.history = [];
        state.history.push(durationMin);
        if(state.history.length > HISTORY_MAX) state.history.shift();
        state.inPow = false;
        state.startAt = null;
        state.lastAlertAt = now;
        await writeState(state);
        // send end msg
        let msg = `🔴 [DG提醒] 放水已結束。\n開始: ${isoInTZ(start)}\n結束: ${isoInTZ(endISO)}\n持續: ${durationMin} 分鐘\n`;
        // update estimated mean after push
        if(state.history.length >= 2){
          const mean = Math.round(state.history.reduce((s,v)=>s+v,0) / state.history.length);
          msg += `歷史平均放水: ${mean} 分鐘（用於未來估計）。`;
        }
        await sendTelegramMsg(msg);
      } else {
        console.log('非放水時段，無須通知。判定：', overall);
      }
    }

  } catch(err){
    console.error('主流程錯誤：', err);
    await sendTelegramMsg(`❗[DG監測] 主流程錯誤：${err.message}`);
  } finally {
    try{ await browser.close(); }catch(e){}
  }

})();
