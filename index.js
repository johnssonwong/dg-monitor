// index.js - DG monitor 强化与 debug 版
const fs = require('fs');
const fetch = require('node-fetch');
const Jimp = require('jimp');
const { Octokit } = require('@octokit/rest');
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());

/* Config (可由 workflow env 覆盖) */
const DG_URLS = [ process.env.DG_URL1 || 'https://dg18.co/wap/', process.env.DG_URL2 || 'https://dg18.co/' ];
const MIN_LONG_BOARDS_FOR_POW = parseInt(process.env.MIN_LONG_BOARDS || '3', 10);
const MID_LONG_REQ = parseInt(process.env.MID_LONG_REQ || '3', 10);
const MID_MULTI_BOARDS = parseInt(process.env.MID_MULTI_BOARDS || '3', 10);
const COOLDOWN_MIN = parseInt(process.env.COOLDOWN_MIN || '10', 10);
const STATE_PATH = 'state.json';
const HISTORY_MAX = 50;
const TIMEZONE = 'Asia/Kuala_Lumpur';
const DEBUG_TO_TG = (process.env.DEBUG_TO_TG === 'true');
const UPLOAD_DEBUG_ARTIFACT = (process.env.UPLOAD_DEBUG_ARTIFACT === 'true');

/* Secrets/env */
const TG_TOKEN = process.env.TG_BOT_TOKEN;
const TG_CHAT = process.env.TG_CHAT_ID;
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const REPO = process.env.GITHUB_REPOSITORY;

/* Octokit */
const oct = new Octokit({ auth: GITHUB_TOKEN });

async function readState(){
  try {
    const [owner, repo] = REPO.split('/');
    const res = await oct.repos.getContent({ owner, repo, path: STATE_PATH });
    const json = Buffer.from(res.data.content, 'base64').toString();
    const parsed = JSON.parse(json);
    parsed.sha = res.data.sha;
    return parsed;
  } catch(e){
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
  console.log('state.json updated, sha=', state.sha);
}

async function sendTelegramMsg(text){
  if(!TG_TOKEN || !TG_CHAT) {
    console.log('Telegram 未配置，跳过发送。');
    return;
  }
  const url = `https://api.telegram.org/bot${TG_TOKEN}/sendMessage`;
  const body = { chat_id: TG_CHAT, text, parse_mode: 'Markdown' };
  try {
    const r = await fetch(url, { method:'POST', body: JSON.stringify(body), headers:{'Content-Type':'application/json'} });
    const j = await r.json();
    console.log('Telegram send:', j.ok ? 'OK' : 'FAIL', j);
    return j;
  } catch(err){ console.error('Telegram 发送异常', err); }
}

/* Image helpers */
function colorIsRed(r,g,b){ return r>140 && g<120 && b<120; }
function colorIsBlue(r,g,b){ return b>140 && r<120 && g<120; }

async function analyzeScreenshot(buffer){
  const img = await Jimp.read(buffer);
  const W = img.bitmap.width, H = img.bitmap.height;
  // finer cell & denser sampling for better sensitivity
  const cell = Math.max(36, Math.floor(Math.min(W,H)/22));
  const cols = Math.ceil(W/cell), rows = Math.ceil(H/cell);
  const counts = [];
  for(let ry=0; ry<rows; ry++){
    for(let rx=0; rx<cols; rx++){
      let rc=0, bc=0;
      const sx = rx*cell, sy = ry*cell, ex = Math.min(W, sx+cell), ey = Math.min(H, sy+cell);
      for(let y=sy; y<ey; y+=2){
        for(let x=sx; x<ex; x+=2){
          const idx = (y*W + x)*4;
          const r = img.bitmap.data[idx], g = img.bitmap.data[idx+1], b = img.bitmap.data[idx+2];
          if(colorIsRed(r,g,b)) rc++;
          else if(colorIsBlue(r,g,b)) bc++;
        }
      }
      counts.push({rx,ry,rc,bc});
    }
  }
  // lowered threshold -> hits
  const hits = counts.filter(c => (c.rc + c.bc) >= 12); // 从24降到12，感度更高
  const regions = [];
  hits.forEach(h=>{
    const x = h.rx*cell, y = h.ry*cell, w = cell, hh = cell;
    let merged = false;
    for(const g of regions){
      if(!(x > g.x + g.w + cell || x + w < g.x - cell || y > g.y + g.h + cell || y + hh < g.y - cell)){
        g.x = Math.min(g.x, x); g.y = Math.min(g.y, y);
        g.w = Math.max(g.w, x+w - g.x); g.h = Math.max(g.h, y+hh - g.y);
        merged = true; break;
      }
    }
    if(!merged) regions.push({x,y,w:h.w || w, h:hh});
  });

  const boards = [];
  for(const r of regions){
    const sx = Math.max(0, r.x), sy = Math.max(0, r.y), sw = Math.min(W - sx, r.w), sh = Math.min(H - sy, r.h);
    if(sw < 36 || sh < 36) continue;
    const crop = img.clone().crop(sx, sy, sw, sh);
    const centers = [];
    for(let y=2; y<sh-2; y+=1){
      for(let x=2; x<sw-2; x+=1){
        const idx = (y*sw + x)*4;
        const r0 = crop.bitmap.data[idx], g0 = crop.bitmap.data[idx+1], b0 = crop.bitmap.data[idx+2];
        if(colorIsRed(r0,g0,b0) || colorIsBlue(r0,g0,b0)){
          centers.push({x,y,color: colorIsRed(r0,g0,b0)?'B':'P'});
        }
      }
    }
    centers.sort((a,b)=>a.x - b.x || a.y - b.y);
    const colGroups = [];
    const colGap = Math.max(8, Math.floor(sw/40));
    centers.forEach(c=>{
      let placed=false;
      for(const g of colGroups){
        if(Math.abs(c.x - g.xAvg) <= colGap){
          g.items.push(c);
          g.xAvg = (g.xAvg * (g.items.length-1) + c.x)/g.items.length;
          placed=true; break;
        }
      }
      if(!placed) colGroups.push({xAvg:c.x, items:[c]});
    });
    const sequences = colGroups.map(g=>{ g.items.sort((a,b)=>a.y-b.y); return g.items.map(i=>i.color); }).filter(s=>s.length>0);
    const flattened=[];
    const maxlen = sequences.reduce((m,s)=>Math.max(m, s.length), 0);
    for(let rrow=0; rrow<maxlen; rrow++){
      for(let c=0;c<sequences.length;c++){
        if(sequences[c][rrow]) flattened.push(sequences[c][rrow]);
      }
    }
    const runs=[];
    if(flattened.length>0){
      let cur = { color: flattened[0], len:1 };
      for(let i=1;i<flattened.length;i++){
        if(flattened[i] === cur.color) cur.len++; else { runs.push(cur); cur={color:flattened[i], len:1}; }
      }
      runs.push(cur);
    }
    const colMaxRuns = sequences.map(seq=>{
      let m=0, c=seq[0]||null, len=0;
      for(let i=0;i<seq.length;i++){
        if(seq[i]===c) len++; else { m=Math.max(m,len); c=seq[i]; len=1; }
      }
      m=Math.max(m,len); return m;
    });
    let multiGroups = 0;
    if(colMaxRuns.length>0){
      let curCount=0;
      for(let i=0;i<colMaxRuns.length;i++){
        if(colMaxRuns[i] >= 4) { curCount++; } else { if(curCount >= 3) multiGroups++; curCount=0; }
      }
      if(curCount >= 3) multiGroups++;
    }
    const maxRun = runs.reduce((m,r)=>Math.max(m, r.len), 0);
    boards.push({ region: r, totalBeads: flattened.length, maxRun, runs, colCount: sequences.length, colMaxRuns, multiGroups });
  }

  const longCount = boards.filter(b=>b.maxRun >= 8).length;
  const superCount = boards.filter(b=>b.maxRun >= 10).length;
  const multiBoardsCount = boards.filter(b=>b.multiGroups >= 1).length;

  return { boards, longCount, superCount, multiBoardsCount, rawRegions: regions.length };
}

/* Helpers for time formatting */
function isoInTZ(d){
  const dt = typeof d === 'string' ? new Date(d) : d;
  const opts = { timeZone: TIMEZONE, year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:false };
  const parts = new Intl.DateTimeFormat('en-GB', opts).formatToParts(dt);
  const map = {}; parts.forEach(p=> map[p.type] = p.value);
  return `${map.year}-${map.month}-${map.day} ${map.hour}:${map.minute}`;
}

/* Main run */
(async ()=>{
  console.log('--- DG monitor run start @', new Date().toISOString());
  const state = await readState();
  const browser = await puppeteer.launch({ args:['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage'], headless:true });
  const page = await browser.newPage();
  try {
    await page.setUserAgent('Mozilla/5.0 (Linux; Android 10; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Mobile Safari/537.36');
    await page.setViewport({ width:1280, height:900 });

    let screenshotBuffer=null;
    let loaded=false;
    // try each url with retries
    for(const url of DG_URLS){
      for(let attempt=0; attempt<2; attempt++){
        try {
          console.log('访问 URL:', url, 'attempt', attempt+1);
          await page.goto(url, { waitUntil:'networkidle2', timeout:45000 });
          // try clicking Free / 免费 / 试玩
          await page.evaluate(()=> {
            const nodes = Array.from(document.querySelectorAll('a,button,div'));
            for(const n of nodes){
              const t = (n.innerText||n.textContent||'').trim();
              if(/free|免费|试玩|Free/i.test(t)){
                try{ n.click(); }catch(e){}
              }
            }
          }).catch(()=>{});
          await page.waitForTimeout(3500);
          // if new pages opened pick last
          const pages = await browser.pages();
          const target = pages[pages.length-1];
          // attempt slider auto-drag (multiple common selectors)
          try {
            const sliderCandidates = await target.$$('input[type=range], .slider, .captcha-slider, .drag, .slide-btn, .nc_slider, .geetest_slider_button');
            if(sliderCandidates.length > 0){
              const el = sliderCandidates[0];
              const box = await el.boundingBox();
              if(box){
                await target.mouse.move(box.x+5, box.y + box.height/2);
                await target.mouse.down();
                await target.mouse.move(box.x + box.width - 10, box.y + box.height/2, { steps: 14 });
                await target.mouse.up();
                await target.waitForTimeout(1200);
                console.log('尝试滑块拖动');
              }
            }
          } catch(e){ console.log('slider attempt error', e.message); }
          await target.waitForTimeout(3000);
          screenshotBuffer = await target.screenshot({ fullPage:true, type:'png' });
          loaded=true; break;
        } catch(e){
          console.warn('访问或截图失败:', e.message);
          await page.waitForTimeout(2000);
        }
      }
      if(loaded) break;
    }

    if(!screenshotBuffer){
      console.error('无法获取页面截图（防爬/被挡或超时）');
      await sendTelegramMsg(`⚠️ [DG监测] 无法取得 DG 截图（可能被平台阻挡）。請人工查看。`);
      await browser.close();
      return;
    }

    // save local debug screenshot if needed
    const ts = Date.now();
    const debugFile = `debug-screenshot-${ts}.png`;
    fs.writeFileSync(debugFile, screenshotBuffer);
    if(UPLOAD_DEBUG_ARTIFACT) console.log('debug screenshot saved:', debugFile);

    // analyze
    const analysis = await analyzeScreenshot(screenshotBuffer);
    console.log('分析-> boards=', analysis.boards.length, 'regions(粗略)=', analysis.rawRegions, 'long=', analysis.longCount, 'super=', analysis.superCount, 'multiBoards=', analysis.multiBoardsCount);
    console.log('每个board maxRun:', analysis.boards.map(b=>b.maxRun).slice(0,12));

    // decision
    let overall='unknown';
    if(analysis.longCount >= MIN_LONG_BOARDS_FOR_POW) overall = '放水時段（提高勝率）';
    else if(analysis.multiBoardsCount >= MID_MULTI_BOARDS && analysis.longCount >= MID_LONG_REQ) overall = '中等勝率（中上）';
    else {
      const sparse = analysis.boards.filter(b=>b.totalBeads < 6).length;
      if(analysis.boards.length > 0 && sparse >= Math.floor(analysis.boards.length * 0.6)) overall = '勝率調低 / 收割時段';
      else overall = '勝率中等（平台收割中等時段）';
    }

    // debug -> send summary to TG if DEBUG_TO_TG true (useful temporarily)
    if(DEBUG_TO_TG){
      const dbgMsg = `🛠 [DG Debug] 判定: ${overall}\nboards:${analysis.boards.length} regions:${analysis.rawRegions}\nlong:${analysis.longCount} super:${analysis.superCount} multiBoards:${analysis.multiBoardsCount}\nmaxRuns: ${analysis.boards.map(b=>b.maxRun).slice(0,10).join(',')}`;
      await sendTelegramMsg(dbgMsg);
      // try upload screenshot via Telegram (if token/chat provided)
      try {
        // send photo via sendPhoto using multipart/form-data
        const formData = require('form-data');
        const fd = new formData();
        fd.append('chat_id', TG_CHAT);
        fd.append('photo', fs.createReadStream(debugFile));
        await fetch(`https://api.telegram.org/bot${TG_TOKEN}/sendPhoto`, { method:'POST', body: fd });
      } catch(e){ console.log('sendPhoto err', e.message); }
    }

    // state transitions
    const now = Date.now();
    const lastAlert = state.lastAlertAt || 0;
    const inPow = state.inPow || false;

    if(overall === '放水時段（提高勝率）' || overall === '中等勝率（中上）'){
      if(!inPow){
        state.inPow = true;
        state.startAt = new Date().toISOString();
        state.lastAlertAt = now;
        if(!Array.isArray(state.history)) state.history = [];
        await writeState(state);
        let msg = `🟢 [DG提醒] *${overall}* 已觸發！\n開始: ${isoInTZ(state.startAt)} (${TIMEZONE})\n長龍/超長龍: ${analysis.longCount}，超長龍: ${analysis.superCount}\n多連(連續3列)桌數: ${analysis.multiBoardsCount}\n`;
        if(state.history && state.history.length >= 3){
          const mean = Math.round(state.history.reduce((s,v)=>s+v,0)/state.history.length);
          msg += `\n⏳ 歷史平均放水: ${mean} 分鐘，估計剩餘: 約 ${Math.max(1, mean)} 分鐘（僅供參考）`;
        } else {
          msg += `\n⏳ 歷史不足，暫無可靠剩餘時間估計；系統將在結束時通知實際持續時間。`;
        }
        await sendTelegramMsg(msg);
      } else {
        const cdMs = COOLDOWN_MIN * 60 * 1000;
        if(now - lastAlert >= cdMs){
          state.lastAlertAt = now;
          await writeState(state);
          await sendTelegramMsg(`🔁 [DG提醒] *${overall}* 仍在進行（重複提醒）\n長龍:${analysis.longCount} 多連桌:${analysis.multiBoardsCount}`);
        } else {
          console.log('冷卻中，跳過重複提醒。');
        }
      }
    } else {
      if(inPow){
        const start = new Date(state.startAt);
        const endISO = new Date().toISOString();
        const durationMin = Math.round((Date.now() - start.getTime())/60000);
        if(!Array.isArray(state.history)) state.history = [];
        state.history.push(durationMin);
        if(state.history.length > HISTORY_MAX) state.history.shift();
        state.inPow = false;
        state.startAt = null;
        state.lastAlertAt = now;
        await writeState(state);
        let msg = `🔴 [DG提醒] 放水已結束。\n開始: ${isoInTZ(start)}\n結束: ${isoInTZ(endISO)}\n持續: ${durationMin} 分鐘`;
        if(state.history.length >= 2){
          const mean = Math.round(state.history.reduce((s,v)=>s+v,0)/state.history.length);
          msg += `\n歷史平均放水: ${mean} 分鐘。`;
        }
        await sendTelegramMsg(msg);
      } else {
        console.log('非放水時段，判定：', overall);
      }
    }

  } catch(err){
    console.error('主流程錯誤:', err);
    await sendTelegramMsg(`❗[DG監測] 主流程錯誤：${err.message}`);
  } finally {
    try{ await browser.close(); } catch(e){}
  }
})();
