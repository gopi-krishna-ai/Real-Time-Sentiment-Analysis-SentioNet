document.addEventListener('DOMContentLoaded', () => {
  const $ = id => document.getElementById(id);
  const history = [];
  let debounceTimer = null;
  let inFlight = false;

  function escapeHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
  function setError(el,msg){el.textContent=msg||'';}

  document.querySelectorAll('.tab-btn').forEach(btn=>btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c=>{c.classList.remove('active');c.hidden=true;});
    btn.classList.add('active');const target=$(btn.dataset.tab);target.hidden=false;target.classList.add('active');
  }));

  const textInput=$('text-input'), charCount=$('char-count'), liveMode=$('live-mode'), errorEl=$('form-error');
  textInput.addEventListener('input',()=>{charCount.textContent=`${textInput.value.length} / 1000`; if(!liveMode.checked)return; clearTimeout(debounceTimer); if(!textInput.value.trim())return; debounceTimer=setTimeout(()=>runText(true),450);});
  $('analyze-btn').addEventListener('click',()=>runText(false));
  $('demo-btn').addEventListener('click',()=>{textInput.value='The support team solved my issue quickly and the experience was excellent.';textInput.dispatchEvent(new Event('input'));runText(false);});

  async function runText(isLive){
    const text=textInput.value.trim();setError(errorEl,'');if(!text){if(!isLive)setError(errorEl,'Please enter some text to analyze.');return;}if(inFlight)return;inFlight=true;
    try{const res=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});const data=await res.json();if(!res.ok){setError(errorEl,data.error);return;}renderText(data,isLive);}catch(e){setError(errorEl,'Could not reach the server. Please try again.');}finally{inFlight=false;}
  }
  function renderText(data,isLive){
    $('sentiment-badge').textContent=data.sentiment;$('sentiment-badge').className=`badge ${data.sentiment}`;
    $('confidence-text').textContent=`${data.confidence}% confidence · ${data.p_positive}% positive signal`;
    const pct=Math.max(0,Math.min(100,(data.polarity+100)/2));$('polarity-marker').style.left=`${pct}%`;$('polarity-value').textContent=`${data.polarity>0?'+':''}${data.polarity}`;
    $('words-list').innerHTML='';if(!data.influential_words.length){$('words-list').innerHTML='<span class="small-note">No strongly weighted vocabulary was found.</span>';} else data.influential_words.forEach(w=>{const c=document.createElement('span');c.className=`word-chip ${w.weight>=0?'pos':'neg'}`;c.innerHTML=`${escapeHtml(w.word)}<span class="word-weight">${w.weight>=0?'+':''}${w.weight}</span>`;$('words-list').appendChild(c);});
    $('agents-list').innerHTML='';data.agents.forEach(a=>{const card=document.createElement('div');card.className='agent-card';card.innerHTML=`<div class="agent-top"><div><div class="agent-name">${escapeHtml(a.agent)}</div><div class="agent-role">${escapeHtml(a.role)}</div></div><span class="agent-action">${escapeHtml(a.action)}</span></div><div class="agent-message">${escapeHtml(a.message)}</div>`;$('agents-list').appendChild(card);});
    $('result-panel').hidden=false;$('agents-panel').hidden=false;
    if(!isLive){history.push({sentiment:data.sentiment,polarity:data.polarity});if(history.length>30)history.shift();renderHistory();}
  }
  function renderHistory(){const chart=$('history-chart');chart.innerHTML='';history.forEach(h=>{const w=document.createElement('div');w.className='history-bar-wrap';const b=document.createElement('div');b.className=`history-bar ${h.sentiment}`;b.style.height=`${Math.max(4,Math.abs(h.polarity))}%`;b.title=`${h.sentiment} (${h.polarity})`;w.appendChild(b);chart.appendChild(w);});$('history-panel').hidden=!history.length;}

  const audioInput=$('audio-input'), dropZone=$('drop-zone');
  $('choose-audio').addEventListener('click',()=>audioInput.click());
  audioInput.addEventListener('change',()=>handleAudioFile(audioInput.files[0]));
  ['dragenter','dragover'].forEach(e=>dropZone.addEventListener(e,ev=>{ev.preventDefault();dropZone.classList.add('dragover');}));
  ['dragleave','drop'].forEach(e=>dropZone.addEventListener(e,ev=>{ev.preventDefault();dropZone.classList.remove('dragover');}));
  dropZone.addEventListener('drop',ev=>handleAudioFile(ev.dataTransfer.files[0]));
  function handleAudioFile(file){if(!file)return;$('audio-name').textContent=`${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`;const url=URL.createObjectURL(file);$('audio-preview').src=url;$('audio-preview').hidden=false;setError($('audio-error'),'');}
  $('analyze-audio-btn').addEventListener('click',runAudio);
  async function runAudio(){const file=audioInput.files[0];setError($('audio-error'),'');if(!file){setError($('audio-error'),'Choose an audio file first.');return;}const fd=new FormData();fd.append('audio',file);const btn=$('analyze-audio-btn');btn.disabled=true;btn.textContent='Analyzing audio…';try{const res=await fetch('/analyze_audio',{method:'POST',body:fd});const data=await res.json();if(!res.ok){setError($('audio-error'),data.error);return;}renderAudio(data);}catch(e){setError($('audio-error'),'Could not reach the server. Please try again.');}finally{btn.disabled=false;btn.textContent='Analyze Audio';}}
  function renderAudio(data){$('audio-badge').textContent=data.emotion;$('audio-badge').className=`badge ${data.emotion}`;$('audio-confidence').textContent=`${data.confidence}% confidence`;$('audio-probabilities').innerHTML='';Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([label,p])=>{const row=document.createElement('div');row.className='prob-row';row.innerHTML=`<span>${escapeHtml(label)}</span><div class="prob-track"><div class="prob-fill" style="width:${p}%"></div></div><b>${p.toFixed(1)}%</b>`;$('audio-probabilities').appendChild(row);});$('audio-result-panel').hidden=false;}
});
