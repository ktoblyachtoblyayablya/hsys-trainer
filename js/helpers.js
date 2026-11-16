// SPDX-License-Identifier: GPL-3.0-or-later · Copyright (C) 2026 Пантелеев Г.Д.
/* Общие данные и функции.
   Q — банк вопросов (data/questions/*.js), A — ответы-билеты (data/answers/*.js),
   C — шпаргалки (data/cards.js), FC — подписи к картинкам (data/figs.js). */
const Q = [];
const A = {};
const C = {};
const FC = {};

/* ---------- картинки ----------
   Все схемы и графики — img/svg/sv_*.svg, их рисует tools/draw_src/draw.py
   (графики работ 1–2 — по замерам из results.json). */
function IMGSRC(k){ return "img/svg/"+k+".svg"; }
function figCap(k, cap){ return cap||FC[k]||""; }
/* F — картинка внутри текста билета */
function F(k, cap){
  return '<figure class="fig"><img src="'+IMGSRC(k)+'" alt="'+(cap||FC[k]||k).replace(/"/g,"&quot;")+'" loading="lazy" data-z="1" onerror="this.closest(\'figure\').hidden=true"><figcaption>'+figCap(k,cap)+'</figcaption></figure>';
}
/* F2 — две картинки рядом */
function F2(a, ca, b, cb){ return '<div class="fig2">'+F(a,ca)+F(b,cb)+'</div>'; }
/* GAL — сетка небольших картинок */
function GAL(keys){ return '<div class="gal">'+keys.map(k=>F(k)).join("")+'</div>'; }
/* CARD — шпаргалка внутри билета */
function CARD(k){ const c=C[k]; return c?'<div class="cardx" data-card="'+k+'"><h3>'+c.title+'</h3>'+c.html+'</div>':''; }
/* L — ссылка на другой билет */
function L(id, text){ return '<a class="tlink" href="#'+id+'" data-t="'+id+'">'+(text||tlabel(id))+'</a>'; }
const SECLABEL = {hpc:"Суперкомпьютеры", gpu:"GPGPU", arch:"Архитектура", clu:"Кластеры", dist:"Распределённые", vend:"Вендоры", mgmt:"Управление", perf:"Производительность", lab:"Работы"};
/* Теоретические темы подписываются номером вопроса к экзамену (№ 1–24), практические — «Работа N». */
function tlabel(id){ const p=id.split("-"); return p[0]==="lab" ? "Работа "+p[1] : "№ "+p[1]; }
/* H — подзаголовок внутри билета, N — заметка «на экзамене / на защите» */
function H(t){ return '<h4>'+t+'</h4>'; }
function N(t){ return '<p class="tip">'+t+'</p>'; }
function UL(items){ return '<ul>'+items.map(x=>'<li>'+x+'</li>').join("")+'</ul>'; }
function OL(items){ return '<ol>'+items.map(x=>'<li>'+x+'</li>').join("")+'</ol>'; }
/* CODE — листинг; спецсимволы HTML экранируются, поэтому в листинги можно писать <<<grid, block>>> и т. п. */
function esc(t){ return String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function CODE(t){ return '<pre class="code">'+esc(t)+'</pre>'; }
/* K — код, команда или имя файла в тексте */
function K(t){ return '<code>'+esc(t)+'</code>'; }
/* FX — формула отдельной строкой (HTML: <sub>, <sup>, &alpha; и т. п.) */
function FX(t){ return '<p class="fx">'+t+'</p>'; }

/* ---------- сверка введённого ответа ----------
   Регистр важен (код CUDA, команды), если у вопроса нет флага ci. Не учитываются: лишние пробелы и пробелы
   вокруг знаков и скобок, приглашение $ или >>, точка с запятой в конце.
   Если ответ — число, сравнивается значение: можно писать запятую или точку, дробь (1/16), степень (2^20),
   добавлять единицы (320 ГБ/с, 2 КиБ, 16 нитей); допускается отклонение 0,5 %. */
function norm(s, ci){
  s=String(s)
    .replace(/[‘’‚‛′`]/g,"'")
    .replace(/[“”„«»]/g,'"')
    .replace(/[–—−]/g,"-")
    .replace(/[×·]/g,"*")
    .replace(/ /g," ")
    .trim()
    .replace(/^(>>|\$)\s*/,"")
    .replace(/\s+/g," ")
    .replace(/\s*([=,;:(){}\[\]+*\/^<>|&~-])\s*/g,"$1")
    .replace(/;+$/,"");
  return ci ? s.toLowerCase() : s;
}
const UNITS=/(гб\/с|мб\/с|gb\/s|тфлопс|пфлопс|эфлопс|гфлопс|флопс|tflops|pflops|eflops|gflops|flops|гиб|миб|киб|гб|мб|кб|gib|mib|kib|gb|mb|kb|байт(а|ов)?|бит(а|ов)?|б|ггц|мгц|ghz|mhz|мвт|квт|вт|mw|kw|w|мкс|нс|мс|с|ns|us|ms|раз|%|°c|нит(ь|и|ей)|блок(а|ов)?|варп(а|ов)?|ядер|ядра|ядро|sm|тест(а|ов)?|элемент(а|ов)?|строк(а|и)?|узл(а|ов)|узел|место|месте|шаг(а|ов)?|вызов(а|ов)?)$/iu;
/* Множители для объёмов памяти: «2 КиБ» сравнивается и как 2, и как 2048 байт. */
const BYTEMUL={"киб":[1024],"kib":[1024],"миб":[1048576],"mib":[1048576],"гиб":[1073741824],"gib":[1073741824],
  "кб":[1e3,1024],"kb":[1e3,1024],"мб":[1e6,1048576],"mb":[1e6,1048576],"гб":[1e9,1073741824],"gb":[1e9,1073741824]};
function parseNum(t){
  if(/^[-+]?\d+,\d+$/.test(t)) t=t.replace(",",".");
  if(/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) return parseFloat(t);
  let m=t.match(/^([-+]?\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/);
  if(m&&+m[2]) return parseFloat(m[1])/parseFloat(m[2]);
  m=t.match(/^(\d+)\^(\d+)$/);
  if(m) return Math.pow(+m[1], +m[2]);
  m=t.match(/^(\d+(?:\.\d+)?)\*(\d+(?:\.\d+)?)\^(\d+)$/);
  if(m) return parseFloat(m[1])*Math.pow(+m[2], +m[3]);
  return null;
}
/* Все числовые прочтения ответа: число без единиц и, для КиБ/МиБ/…, то же в байтах. */
function toNums(s){
  const t=String(s).trim().replace(/[–—−]/g,"-").replace(/\s+/g,"").replace(/^≈/,"");
  const u=t.match(UNITS), unit=u?u[0].toLowerCase():"";
  const x=parseNum(u?t.slice(0,t.length-u[0].length):t);
  if(x===null) return [];
  return [x].concat((BYTEMUL[unit]||[]).map(k=>x*k));
}
function toNum(s){ const v=toNums(s); return v.length?v[0]:null; }
function checkInput(q, val){
  const v=norm(val,q.ci); if(!v) return false;
  const xs=toNums(val);
  return q.a.some(x=>{
    if(norm(x,q.ci)===v) return true;
    const ys=toNums(x);
    return xs.some(a=>ys.some(b=>Math.abs(a-b)<=Math.max(1e-9,Math.abs(b)*0.005)));
  });
}

/* ---------- хелперы для шпаргалок ---------- */
function card(title){ return {title:title, html:[].slice.call(arguments,1).join("")}; }
function tbl(head, rows){
  let h='<div class="tw"><table class="ct"><thead><tr>'+head.map(x=>'<th>'+x+'</th>').join("")+'</tr></thead><tbody>';
  rows.forEach(r=>{ h+='<tr>'+r.map(x=>'<td>'+x+'</td>').join("")+'</tr>'; });
  return h+'</tbody></table></div>';
}
function chain(items){
  return '<div class="chain">'+items.map((it,i)=>(i?'<span class="arr">→</span>':'')+'<div class="node"><b>'+it[0]+'</b>'+(it[1]?'<span>'+it[1]+'</span>':'')+'</div>').join("")+'</div>';
}
function note(t){ return '<p class="cn">'+t+'</p>'; }
