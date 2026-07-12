# -*- coding: utf-8 -*-
"""Generate a LaTeX report of all MER-Cross architectures: TikZ diagram +
idea/rationale/reference + in-domain confusion matrix + per-class metrics.
Compile with XeLaTeX (Vietnamese, fontspec)."""
import json

EMOS=['neutral','angry','happy','sad','worried','surprise']
EAB=['neu','ang','hap','sad','wor','sur']

CM=json.loads(r'''[{"name":"v6","waf":0.785,"cm":[[1635,178,156,96,161,58],[174,1681,36,71,109,84],[97,26,1662,14,11,17],[89,58,17,1390,81,23],[111,86,24,55,756,49],[23,52,21,12,37,245]],"per":{"neutral":{"p":0.768,"r":0.716,"f1":0.741,"n":2284},"angry":{"p":0.808,"r":0.78,"f1":0.794,"n":2155},"happy":{"p":0.867,"r":0.91,"f1":0.888,"n":1827},"sad":{"p":0.849,"r":0.838,"f1":0.843,"n":1658},"worried":{"p":0.655,"r":0.699,"f1":0.676,"n":1081},"surprise":{"p":0.515,"r":0.628,"f1":0.566,"n":390}}},
{"name":"v8","waf":0.7816,"cm":[[1625,164,148,108,166,73],[150,1683,34,70,114,104],[98,33,1648,12,16,20],[90,60,18,1389,74,27],[107,95,18,67,725,69],[31,46,15,10,30,258]],"per":{"neutral":{"p":0.773,"r":0.711,"f1":0.741,"n":2284},"angry":{"p":0.809,"r":0.781,"f1":0.795,"n":2155},"happy":{"p":0.876,"r":0.902,"f1":0.889,"n":1827},"sad":{"p":0.839,"r":0.838,"f1":0.838,"n":1658},"worried":{"p":0.644,"r":0.671,"f1":0.657,"n":1081},"surprise":{"p":0.468,"r":0.662,"f1":0.548,"n":390}}},
{"name":"v3","waf":0.7816,"cm":[[1588,196,150,100,184,66],[155,1708,38,71,108,75],[88,31,1657,13,21,17],[83,64,15,1385,86,25],[98,98,22,63,758,42],[24,54,22,12,36,242]],"per":{"neutral":{"p":0.78,"r":0.695,"f1":0.735,"n":2284},"angry":{"p":0.794,"r":0.793,"f1":0.793,"n":2155},"happy":{"p":0.87,"r":0.907,"f1":0.888,"n":1827},"sad":{"p":0.842,"r":0.835,"f1":0.839,"n":1658},"worried":{"p":0.635,"r":0.701,"f1":0.667,"n":1081},"surprise":{"p":0.518,"r":0.621,"f1":0.565,"n":390}}},
{"name":"v10 (CLIP+AU)","waf":0.7803,"cm":[[1526,224,134,119,192,89],[127,1720,26,61,116,105],[95,39,1630,17,23,23],[61,73,15,1412,79,18],[85,100,11,66,762,57],[26,51,12,11,23,267]],"per":{"neutral":{"p":0.795,"r":0.668,"f1":0.726,"n":2284},"angry":{"p":0.779,"r":0.798,"f1":0.789,"n":2155},"happy":{"p":0.892,"r":0.892,"f1":0.892,"n":1827},"sad":{"p":0.837,"r":0.852,"f1":0.844,"n":1658},"worried":{"p":0.638,"r":0.705,"f1":0.67,"n":1081},"surprise":{"p":0.478,"r":0.685,"f1":0.563,"n":390}}},
{"name":"v11","waf":0.7792,"cm":[[1593,218,142,103,168,60],[154,1694,39,72,112,84],[98,25,1663,11,16,14],[86,75,16,1376,85,20],[89,108,17,61,752,54],[27,62,20,15,28,238]],"per":{"neutral":{"p":0.778,"r":0.697,"f1":0.736,"n":2284},"angry":{"p":0.776,"r":0.786,"f1":0.781,"n":2155},"happy":{"p":0.877,"r":0.91,"f1":0.893,"n":1827},"sad":{"p":0.84,"r":0.83,"f1":0.835,"n":1658},"worried":{"p":0.648,"r":0.696,"f1":0.671,"n":1081},"surprise":{"p":0.506,"r":0.61,"f1":0.553,"n":390}}},
{"name":"v9 (CLIP+AU)","waf":0.7778,"cm":[[1583,200,149,107,167,78],[161,1691,30,81,103,89],[96,30,1643,16,20,22],[72,72,18,1400,74,22],[104,94,17,82,722,62],[26,51,15,11,27,260]],"per":{"neutral":{"p":0.775,"r":0.693,"f1":0.732,"n":2284},"angry":{"p":0.791,"r":0.785,"f1":0.788,"n":2155},"happy":{"p":0.878,"r":0.899,"f1":0.888,"n":1827},"sad":{"p":0.825,"r":0.844,"f1":0.835,"n":1658},"worried":{"p":0.649,"r":0.668,"f1":0.658,"n":1081},"surprise":{"p":0.488,"r":0.667,"f1":0.563,"n":390}}},
{"name":"v3 (FER)","waf":0.7713,"cm":[[1529,201,155,125,192,82],[125,1707,34,100,100,89],[97,40,1616,22,25,27],[75,67,19,1385,87,25],[100,93,16,80,744,48],[27,48,20,10,28,257]],"per":{"neutral":{"p":0.783,"r":0.669,"f1":0.722,"n":2284},"angry":{"p":0.792,"r":0.792,"f1":0.792,"n":2155},"happy":{"p":0.869,"r":0.885,"f1":0.877,"n":1827},"sad":{"p":0.804,"r":0.835,"f1":0.82,"n":1658},"worried":{"p":0.633,"r":0.688,"f1":0.659,"n":1081},"surprise":{"p":0.487,"r":0.659,"f1":0.56,"n":390}}},
{"name":"v7","waf":0.7184,"cm":[[1466,280,158,113,180,87],[233,1489,46,88,176,123],[104,35,1641,10,18,19],[120,96,21,1268,120,33],[111,132,25,74,677,62],[37,68,26,15,54,190]],"per":{"neutral":{"p":0.708,"r":0.642,"f1":0.673,"n":2284},"angry":{"p":0.709,"r":0.691,"f1":0.7,"n":2155},"happy":{"p":0.856,"r":0.898,"f1":0.877,"n":1827},"sad":{"p":0.809,"r":0.765,"f1":0.786,"n":1658},"worried":{"p":0.553,"r":0.626,"f1":0.587,"n":1081},"surprise":{"p":0.37,"r":0.487,"f1":0.42,"n":390}}}]''')
CMBY={d['name']:d for d in CM}

def cm_table(d):
    s=[r"\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{l"+"c"*6+"}",r"\toprule",
       r"\textbf{true$\downarrow$/pred$\rightarrow$} & "+" & ".join(r"\textbf{%s}"%a for a in EAB)+r" \\",r"\midrule"]
    for i,row in enumerate(d['cm']):
        tot=sum(row) or 1; cells=[]
        for j,c in enumerate(row):
            pct=round(c/tot*100); shade=min(65,round(c/tot*70))
            val=r"\textbf{%d}"%pct if i==j else "%d"%pct
            cells.append(r"\cellcolor{cmblue!%d}%s"%(shade,val))
        s.append(r"\textbf{%s} & "%EAB[i]+" & ".join(cells)+r" \\")
    s+=[r"\bottomrule",r"\end{tabular}",
        r"\caption{Confusion matrix (\% theo hàng, in-domain). Đường chéo = recall.}",r"\end{table}"]
    return "\n".join(s)

def per_table(d):
    s=[r"\begin{table}[H]\centering\footnotesize",r"\begin{tabular}{lcccc}",r"\toprule",
       r"\textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} & \textbf{N} \\",r"\midrule"]
    for e in EMOS:
        p=d['per'][e]; s.append(r"%s & %.3f & %.3f & %.3f & %d \\"%(e,p['p'],p['r'],p['f1'],p['n']))
    s+=[r"\midrule",r"\textbf{WAF (in-domain)} & \multicolumn{4}{c}{\textbf{%.4f}} \\"%d['waf'],
        r"\bottomrule",r"\end{tabular}",r"\caption{Per-class precision/recall/F1 (in-domain).}",r"\end{table}"]
    return "\n".join(s)

# ---------- TikZ diagrams ----------
def N(i,col,row,lab,c,span=1,sub=None): return dict(id=i,col=col,row=row,lab=lab,c=c,span=span,sub=sub)
def E(f,t,l=None,d=False): return dict(f=f,t=t,l=l,d=d)

def tikz(spec,cap):
    dx,dy=3.75,-1.5
    out=[r"\begin{figure}[H]\centering",
         r"\begin{tikzpicture}"]
    for n in spec['nodes']:
        sp=n['span']; x=n['col']*dx+(sp-1)*dx/2; y=n['row']*dy
        mw=sp*3.2; tw=mw-0.4
        lab=n['lab']
        if n['sub']: lab=lab+r"\\{\scriptsize\ttfamily %s}"%n['sub']
        sty='txt' if n['c']=='text' else n['c']
        out.append(r"\node[%s,minimum width=%.2fcm,text width=%.2fcm] (%s) at (%.2f,%.2f) {%s};"
                   %(sty,mw,tw,n['id'],x,y,lab))
    for e in spec['edges']:
        st='ard' if e['d'] else 'ar'
        lab=r" node[midway,font=\tiny,fill=white,inner sep=1pt]{%s}"%e['l'] if e['l'] else ''
        out.append(r"\draw[%s] (%s.south) --%s (%s.north);"%(st,e['f'],lab,e['t']))
    out+=[r"\end{tikzpicture}",r"\caption{%s}"%cap,r"\end{figure}"]
    return "\n".join(out)

# backbone helper (v3-style speaker+listener up to sp_a/sp_t + listener_feat)
def v3top():
    return [N('a',0,0,'WavLM','audio',sub='speaker'),N('t',1,0,'RoBERTa','text',sub='speaker'),
            N('v',2,0,'CLIP-ViT','video',sub='listener'),
            N('la',0,1,'LSTM','audio'),N('lt',1,1,'LSTM','text'),N('lv',2,1,'LSTM','video'),
            N('xa',0,2,r'Bidir Cross-Attn (a$\leftrightarrow$t)','spk',span=2),
            N('pl',2,2,'Learnable Query Pooling','video'),
            N('spa',0,3,r'Mean sp\_a','spk'),N('spt',1,3,r'Mean sp\_t','spk')]
def v3topedges():
    return [E('a','la'),E('t','lt'),E('v','lv'),E('la','xa'),E('lt','xa'),E('lv','pl'),
            E('xa','spa'),E('xa','spt')]

DIAG={}
DIAG['v3']=(dict(nodes=v3top()+[
    N('fus',0,4,r'Cross-Attn Fusion: Q=listener, K/V=[sp\_a,sp\_t]','fuse',span=3),
    N('res',0,5,'Residual + LayerNorm','fuse',span=3),
    N('cls',0,6,r'Linear $\rightarrow$ 6 emotions','out',span=3)],
    edges=v3topedges()+[E('spa','fus'),E('spt','fus'),E('pl','fus','query'),
    E('fus','res'),E('pl','res','anchor',True),E('res','cls')]),
 'v3: pool a2t/t2a riêng (Fix2) + fusion 2-token [sp\\_a,sp\\_t] (Fix3).')

DIAG['v1']=(dict(nodes=[N('a',0,0,'WavLM','audio',sub='speaker'),N('t',1,0,'RoBERTa','text',sub='speaker'),
    N('v',2,0,'CLIP-ViT','video',sub='listener'),
    N('la',0,1,'LSTM','audio'),N('lt',1,1,'LSTM','text'),N('lv',2,1,'LSTM','video'),
    N('xa',0,2,r'Bidir Cross-Attn (a$\leftrightarrow$t)','spk',span=2),N('pl',2,2,'Learnable Query Pooling','video'),
    N('sp',0,3,'Concat + Mean (1 vector)','spk',span=2),
    N('fus',0,4,'Cross-Attn Fusion: Q=listener, K/V=speaker (1 token)','fuse',span=3),
    N('res',0,5,'Residual + LayerNorm','fuse',span=3),N('cls',0,6,r'Linear $\rightarrow$ 6','out',span=3)],
    edges=[E('a','la'),E('t','lt'),E('v','lv'),E('la','xa'),E('lt','xa'),E('lv','pl'),
    E('xa','sp'),E('sp','fus'),E('pl','fus','query'),E('fus','res'),E('pl','res','anchor',True),E('res','cls')]),
 'v1: gộp chung a2t+t2a rồi mean $\\to$ 1 token; fusion 1-token.')

DIAG['v2']=(dict(nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,'CLIP-ViT','video'),
    N('la',0,1,'LSTM','audio'),N('lt',1,1,'LSTM','text'),N('sv',2,1,'BiLSTM + Self-Attn','video'),
    N('cva',0,2,'Cross-Attn (Q=visual, K/V=audio)','video'),N('cvt',1,2,'Cross-Attn (Q=visual, K/V=text)','video'),
    N('g',0,3,r'Gate $g\cdot v_a+(1-g)\cdot v_t + h_v$ + LayerNorm','fuse',span=3),
    N('p',0,4,'Learnable Query Pooling','video',span=3),N('cls',0,5,r'Linear $\rightarrow$ 6','out',span=3)],
    edges=[E('a','la'),E('t','lt'),E('v','sv'),E('la','cva'),E('lt','cvt'),E('sv','cva','',True),
    E('sv','cvt','',True),E('cva','g'),E('cvt','g'),E('sv','g','residual',True),E('g','p'),E('p','cls')]),
 'v2: video làm anchor, attend a/t, gate + residual video thô.')

DIAG['v4/v5']=(dict(nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,'CLIP-ViT','video'),
    N('xa',0,1,r'Bidir Cross-Attn (a$\leftrightarrow$t), full seq','spk',span=2),N('pl',2,1,'Query Pooling','video'),
    N('fa',0,2,'Cross-Attn (Q=list, K/V=a2t)','fuse'),N('ft',1,2,'Cross-Attn (Q=list, K/V=t2a)','fuse'),
    N('mix',0,3,r'Combine: v4 $0.5/0.5$ / v5 gate $g$','fuse',span=2),
    N('res',0,4,'Residual listener + LayerNorm','fuse',span=3),N('cls',0,5,r'Linear $\rightarrow$ 6','out',span=3)],
    edges=[E('a','xa'),E('t','xa'),E('v','pl'),E('xa','fa'),E('xa','ft'),E('pl','fa','',True),E('pl','ft','',True),
    E('fa','mix'),E('ft','mix'),E('mix','res'),E('pl','res','anchor',True),E('res','cls')]),
 'v4/v5: listener attend full-seq a2t \\& t2a riêng; trộn cố định (v4) / gate (v5).')

def aux_backbone(auxnode,auxedges,cap):
    nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,'CLIP-ViT','video'),
     N('sp',0,1,r'Speaker Bi-Cross-Attn $\rightarrow$ sp\_a, sp\_t','spk',span=2),N('pl',2,1,'Query Pool listener','video'),
     N('fus',0,2,'2-Token Cross-Attn Fusion + Residual (v3)','fuse',span=3),
     N('cls',0,3,r'Linear $\rightarrow$ 6 (CE chính)','out',span=3)]+auxnode
    edges=[E('a','sp'),E('t','sp'),E('v','pl'),E('sp','fus'),E('pl','fus'),E('fus','cls')]+auxedges
    return (dict(nodes=nodes,edges=edges),cap)

DIAG['v6']=aux_backbone([N('ax',0,4,r'aux (train): L1 dịch listener$\rightarrow$sp\_a/sp\_t (detach) + align cosine','dash',span=3)],
    [E('pl','ax','',True),E('sp','ax','',True)],'v6: backbone v3 + CM-StEW aux (train-only).')
DIAG['v11']=aux_backbone([N('ax',2,4,'aux CE video-only','dash')],
    [E('pl','ax','',True)],'v11: backbone v3 + deep-supervision CE trên listener\\_feat.')

DIAG['v7']=(dict(nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,'CLIP-ViT','video'),
    N('sw',0,1,r'Conflict Swap (train): audio+text $\leftarrow$ donor khác nhãn','dash',span=2),
    N('sp',0,2,r'Speaker Bi-Cross-Attn $\rightarrow$ sp\_a, sp\_t','spk',span=2),N('pl',2,2,'Query Pool listener','video'),
    N('fus',0,3,'2-Token Cross-Attn Fusion + Residual','fuse',span=3),N('cls',0,4,r'Linear $\rightarrow$ 6','out',span=3)],
    edges=[E('a','sw',None,True),E('t','sw',None,True),E('sw','sp'),E('v','pl'),E('sp','fus'),E('pl','fus'),E('fus','cls')]),
 'v7: backbone v3 + tráo speaker khác-nhãn lúc train (NCE).')

DIAG['v8']=(dict(nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,'CLIP-ViT','video'),
    N('ra',0,1,'Retrieve a\\_ref (Q=h\\_v)','video'),N('rt',1,1,'Retrieve t\\_ref (Q=h\\_v)','video'),
    N('hv',2,1,'Query Pool $\\rightarrow$ h\\_v','video'),N('aux',2,2,'aux video-only $\\rightarrow$ c\\_v','neu'),
    N('d',0,2,r'$\Delta$ = MLP[t\_ref$-$h\_v, a\_ref$-$h\_v]','fuse',span=2),
    N('star',0,3,r'h\_v$^{*}$ = h\_v + (1$-$c\_v)$\Delta$ + LayerNorm','fuse',span=3),
    N('cls',0,4,r'Concat[h\_v$^{*}$, t\_ref, a\_ref] $\rightarrow$ Linear $\rightarrow$ 6','out',span=3)],
    edges=[E('a','ra'),E('t','rt'),E('v','hv'),E('hv','ra','',True),E('hv','rt','',True),E('hv','aux'),
    E('ra','d'),E('rt','d'),E('hv','star','anchor'),E('d','star'),E('aux','star','gate',True),E('star','cls')]),
 'v8: reliability-gated complementation (video anchor + speaker theo độ tin c\\_v).')

DIAG['v9/v10']=(dict(nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,r'CLIP $\oplus$ AU','video'),
    N('sp',0,1,r'Speaker Bi-Cross-Attn $\rightarrow$ sp\_a, sp\_t','spk',span=2),
    N('clip',2,1,'CLIP enc + pool','video'),N('au',2,2,'AU enc + pool (au\\_feat)','neu'),
    N('fus',0,3,r'Fusion: v9 [sp\_a,sp\_t] / v10 [sp\_a,sp\_t,au\_feat]','fuse',span=3),
    N('cls',0,4,r'Linear $\rightarrow$ 6','out',span=3)],
    edges=[E('a','sp'),E('t','sp'),E('v','clip'),E('v','au'),E('sp','fus'),E('clip','fus','anchor'),
    E('au','fus','v9 merge / v10 token',True),E('fus','cls')]),
 'v9/v10: thêm FACS Action-Unit vào nhánh video.')

DIAG['v12']=(dict(nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,'CLIP-ViT','video'),
    N('sp',0,1,r'Speaker Bi-Cross-Attn $\rightarrow$ sp\_a, sp\_t','spk',span=2),N('pl',2,1,'Query Pool listener','video'),
    N('proj',0,2,r'concat[sp\_a,sp\_t] $\rightarrow$ Linear','spk',span=2),
    N('vib',0,3,r'VIB: $\mu,\log\sigma^2 \rightarrow z$','spk',span=2),N('film',2,3,r'FiLM: $\gamma\odot$listener$+\beta$','fuse'),
    N('res',0,4,'Residual listener + LayerNorm','fuse',span=3),N('cls',0,5,r'Linear $\rightarrow$ 6','out',span=3),
    N('kl',0,6,r'VIB KL$(z\|\mathcal{N}(0,1))\times\beta_{vib}$ (train)','dash',span=2)],
    edges=[E('a','sp'),E('t','sp'),E('v','pl'),E('sp','proj'),E('proj','vib'),E('vib','film',r'$\gamma,\beta$'),
    E('pl','film','anchor'),E('film','res'),E('pl','res','',True),E('res','cls'),E('vib','kl','',True)]),
 'v12: video anchor; speaker nén VIB $\\to$ FiLM điều chế (bất đối xứng).')

DIAG['FER']=(dict(nodes=[N('a',0,0,'WavLM','audio'),N('t',1,0,'RoBERTa','text'),N('v',2,0,'Face crops','video'),
    N('fer',2,1,'HSEmotion enet\\_b2\\_8','video',sub='AffectNet-8, frozen'),
    N('sp',0,2,r'Speaker Bi-Cross-Attn $\rightarrow$ sp\_a, sp\_t','spk',span=2),N('pl',2,2,'FER feat + Query Pool','video'),
    N('fus',0,3,'2-Token Cross-Attn Fusion + Residual (v3)','fuse',span=3),N('cls',0,4,r'Linear $\rightarrow$ 6','out',span=3)],
    edges=[E('a','sp'),E('t','sp'),E('v','fer'),E('fer','pl'),E('sp','fus'),E('pl','fus'),E('fus','cls')]),
 'FER: thay CLIP bằng HSEmotion (AffectNet); phần còn lại = v3.')

DIAG['MLLM']=(dict(nodes=[N('fr',0,0,'8 frame mặt listener','video',sub='crop 112^2'),N('tr',2,0,'Transcript speaker','text'),
    N('vit',0,1,'Qwen ViT (frozen)','video'),N('pr',1,1,'Prompt vai (role-explicit)','neu',span=2),
    N('llm',0,2,'Qwen2.5-VL-7B + LoRA (QLoRA 4-bit)','fuse',span=3),
    N('sft',0,3,r'SFT: target = \{emotion: nhãn\}','out',span=2),N('inf',2,3,'Verbalizer log-prob','out'),
    N('out',0,4,r'softmax 6 nhãn $\rightarrow$ logits $\rightarrow$ submission','out',span=3)],
    edges=[E('fr','vit'),E('vit','llm'),E('tr','pr'),E('pr','llm'),E('llm','sft'),E('llm','inf'),E('inf','out')]),
 'MLLM: Qwen2.5-VL QLoRA fine-tune + verbalizer inference.')

# ---------- sections ----------
A=[
("v1","MemoCMT baseline",
 r"Speaker: cross-attention hai chiều audio$\leftrightarrow$text rồi nối+mean-pool thành một vector; listener: video qua Learnable Query Pooling; fusion Q=listener, K/V=speaker + residual.",
 r"Tái lập MemoCMT làm baseline dyadic: cross-attention cho hai modality bổ sung nhau, listener video làm truy vấn.",
 r"\cite{memocmt,vaswani2017}","baseline",None,'v1'),
("v3","MemoCMT-v3 (best)",
 r"v1 + \textbf{Fix2} (mean-pool a2t và t2a \emph{riêng}) + \textbf{Fix3} (fusion K/V = 2 token [sp\_a, sp\_t]).",
 r"Fix2 sửa mất cân bằng khi số token audio$\neq$text; Fix3 tránh attention suy biến khi chỉ 1 token K/V. Mô hình WAF cao nhất.",
 r"\cite{memocmt,vaswani2017}",r"\textbf{65.78}",'v3','v3'),
("v2","Visual-anchored gated fusion",
 r"Video (BiLSTM) self-attn rồi attend a/t (Q=visual); gate $g\cdot v_a+(1-g)\cdot v_t+h_v$ + residual, rồi pooling.",
 r"Ở test chỉ video (listener) transfer được $\to$ đặt video làm trục, a/t bổ trợ, residual giữ video khi a/t không đáng tin.",
 r"\cite{arevalo2017gmu}","63--64",None,'v2'),
("v4/v5","Full-sequence fusion (fixed / gated)",
 r"Listener query attend \emph{toàn chuỗi} a2t và t2a riêng (frame-level); v4 trộn $0.5/0.5$, v5 gate học per-channel.",
 r"Thử tăng độ mịn frame-level và đảm bảo cân bằng audio/text qua softmax riêng từng modality.",
 r"\cite{vaswani2017}",r"$\sim$64",None,'v4/v5'),
("v6","CM-StEW auxiliary transfer",
 r"Backbone=v3. Train thêm: MLP dịch listener\_feat$\to$sp\_a/sp\_t (detach, L1) + align cosine. Chỉ lúc train.",
 r"Train (Individual) ba modality cùng người $\to$ audio/text \emph{dạy} video encoder mang cảm xúc; test nhánh video giàu hơn transfer.",
 r"\cite{hinton2015}","65.36",'v6','v6'),
("v7","Nonverbal Conflict Exposure",
 r"Backbone=v3. Lúc train, xác suất $p$ tráo audio+text bằng donor \emph{khác nhãn} trong batch (giữ video+nhãn).",
 r"Test a/t thuộc speaker (có thể mâu thuẫn); phơi bày mâu thuẫn để ép model neo vào video.",
 r"\cite{ganin2016dann}","62--63",'v7','v7'),
("v8","Reliability-Guided Complementation",
 r"Video anchor truy hồi tham chiếu speaker (Q=video); head phụ cho $c_v$; $h_v^{*}=h_v+(1-c_v)\Delta$; head trên $[h_v^{*},t_{ref},a_{ref}]$.",
 r"Chỉ để speaker bổ trợ khi video không chắc; video tự tin thì nén speaker $\to$ giảm nhiễu.",
 r"\cite{arevalo2017gmu}","63",'v8','v8'),
("v11","Deep-supervised video branch",
 r"Backbone=v3. Train thêm CE phụ chỉ trên listener\_feat.",
 r"In-domain a/t mạnh nên video ``lười''; deep supervision ép nhánh video tự phân biệt cảm xúc $\to$ anchor mạnh cho test.",
 r"\cite{lee2015dsn}",r"$\sim$65",'v11','v11'),
("v9/v10","FACS Action-Unit features",
 r"Feature video = CLIP$\oplus$AU. v9: hai nhánh gated embedding; v10: CLIP anchor + au\_feat làm token K/V thứ 3.",
 r"AU (FACS) là biểu cảm bất biến danh tính. Thực tế bị observation-shift: speaker nói kích hoạt mouth-AU, listener im lặng thì không.",
 r"\cite{ekman1978facs,cheong2023pyfeat}","60--61",'v9','v9/v10'),
("v12","Asymmetric VIB-FiLM fusion",
 r"Video anchor; speaker nén qua VIB $\to z\to$ FiLM $(\gamma,\beta)$ điều chế listener\_feat. FiLM zero-init; VIB KL train-only.",
 r"Bất đối xứng: video = bằng chứng trực tiếp, a/t = kích thích ngoại lai; VIB lọc đặc trưng speaker không transfer, FiLM để speaker điều chỉnh không lấn át.",
 r"\cite{alemi2017,perez2018}","(đang chạy)",None,'v12'),
("FER","FER-specialised video backbone",
 r"Thay CLIP bằng HSEmotion (EfficientNet-B2, AffectNet-8); v3 giữ nguyên audio/text/fusion.",
 r"Kỳ vọng vision chuyên biểu cảm bắt vi biểu cảm tốt hơn. Thua CLIP: AffectNet posed $\neq$ mặt listener hội thoại (domain gap).",
 r"\cite{savchenko2022,mollahosseini2017}","59.6--61",'v3 (FER)','FER'),
("MLLM","Qwen2.5-VL QLoRA + verbalizer",
 r"Frame mặt + transcript $\to$ prompt vai $\to$ LoRA sinh nhãn; infer log-prob first-token 6 nhãn. Biến thể video-only bỏ transcript.",
 r"Thử reasoning MLLM. Thất bại: vision generic (ViT freeze) trên crop $112^2$ đọc micro-expression yếu; in-domain 0.87 nhưng test 0.50 $\to$ nút thắt là tri giác, không phải reasoning.",
 r"\cite{qwen25vl,dettmers2023,schick2021}","42--50",None,'MLLM'),
("Transductive","Pseudo-label / TMA-BBA",
 r"20k candidate không nhãn: (a) pseudo-label trộn vào train; (b) TMA/BBA: v3 làm bridge, re-init head, train thuần pseudo pool.",
 r"Khai thác target không nhãn. Pseudo-mix +0.2; TMA/BBA $-4.3$ do bẫy thiên kiến xác nhận.",
 r"\cite{liang2020shot,zhu2025bba}","66.0 / 61.47",None,None),
("Label-shift","Post-hoc calibration ($\\tau$ / MLLS)",
 r"$\text{logit}-\tau\log(\text{train\_prior})$ ($\tau=2.5$); hoặc BCTS + MLLS (EM ước lượng prior target trên 20k).",
 r"Phân phối lớp train$\neq$test $\to$ hiệu chỉnh prior. $\tau$ ổn định (+); MLLS có nguyên lý hơn nhưng bị vi phạm giả định label-shift.",
 r"\cite{lipton2018,alexandari2020}","(hậu kỳ)",None,None),
]

BIB=r"""
\begin{thebibliography}{99}
\bibitem{memocmt} \emph{MemoCMT: Multimodal emotion recognition using cross-modal transformer-based feature fusion}, 2025.
\bibitem{vaswani2017} A. Vaswani et al., \emph{Attention Is All You Need}, NeurIPS 2017.
\bibitem{arevalo2017gmu} J. Arevalo et al., \emph{Gated Multimodal Units for Information Fusion}, ICLR-W 2017.
\bibitem{hinton2015} G. Hinton et al., \emph{Distilling the Knowledge in a Neural Network}, 2015.
\bibitem{ganin2016dann} Y. Ganin et al., \emph{Domain-Adversarial Training of Neural Networks}, JMLR 2016.
\bibitem{lee2015dsn} C.-Y. Lee et al., \emph{Deeply-Supervised Nets}, AISTATS 2015.
\bibitem{ekman1978facs} P. Ekman, W. Friesen, \emph{Facial Action Coding System}, 1978.
\bibitem{cheong2023pyfeat} J. Cheong et al., \emph{Py-Feat: Python Facial Expression Analysis Toolbox}, 2023.
\bibitem{alemi2017} A. Alemi et al., \emph{Deep Variational Information Bottleneck}, ICLR 2017.
\bibitem{perez2018} E. Perez et al., \emph{FiLM: Visual Reasoning with a General Conditioning Layer}, AAAI 2018.
\bibitem{savchenko2022} A. Savchenko, \emph{HSEmotion: EfficientNet on AffectNet}, 2022.
\bibitem{mollahosseini2017} A. Mollahosseini et al., \emph{AffectNet}, IEEE TAC 2017.
\bibitem{qwen25vl} Qwen Team, \emph{Qwen2.5-VL Technical Report}, 2025.
\bibitem{dettmers2023} T. Dettmers et al., \emph{QLoRA}, NeurIPS 2023.
\bibitem{schick2021} T. Schick, H. Sch\"utze, \emph{Exploiting Cloze-Questions (PET)}, EACL 2021.
\bibitem{liang2020shot} J. Liang et al., \emph{SHOT: Do We Really Need Source Data?}, ICML 2020.
\bibitem{zhu2025bba} J. Zhu et al., \emph{Bridge Then Begin Anew}, AAAI 2025.
\bibitem{lipton2018} Z. Lipton et al., \emph{Detecting and Correcting for Label Shift (BBSE)}, ICML 2018.
\bibitem{alexandari2020} A. Alexandari et al., \emph{Maximum Likelihood Label Shift (MLLS)}, ICML 2020.
\bibitem{mer2026} \emph{MER 2026: From Discriminative to Generative Emotion Understanding}, arXiv:2604.19417.
\end{thebibliography}
"""

HEAD=r"""% Compile: xelatex report_architectures.tex  (chạy 2 lần cho \cite)
\documentclass[10pt]{article}
\usepackage{fontspec}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,array,float,xcolor,colortbl}
\usepackage{tikz}\usetikzlibrary{arrows.meta}
\usepackage[hidelinks]{hyperref}
\definecolor{cmblue}{RGB}{63,69,168}
\definecolor{audiof}{RGB}{226,241,241}\definecolor{audiol}{RGB}{14,124,134}
\definecolor{textf}{RGB}{246,236,214}\definecolor{textl}{RGB}{169,114,15}
\definecolor{videof}{RGB}{232,232,251}\definecolor{videol}{RGB}{70,76,178}
\definecolor{spkf}{RGB}{239,230,244}\definecolor{spkl}{RGB}{138,83,176}
\definecolor{fusef}{RGB}{251,226,234}\definecolor{fusel}{RGB}{192,68,106}
\definecolor{outf}{RGB}{231,236,244}\definecolor{outl}{RGB}{61,81,112}
\definecolor{neuf}{RGB}{238,240,244}\definecolor{neul}{RGB}{108,118,136}
\tikzset{
 boxbase/.style={draw,rounded corners=2pt,align=center,inner sep=2.5pt,minimum height=0.8cm,font=\scriptsize,line width=0.7pt,text=black},
 audio/.style={boxbase,fill=audiof,draw=audiol},
 txt/.style={boxbase,fill=textf,draw=textl},
 video/.style={boxbase,fill=videof,draw=videol},
 spk/.style={boxbase,fill=spkf,draw=spkl},
 fuse/.style={boxbase,fill=fusef,draw=fusel},
 out/.style={boxbase,fill=outf,draw=outl},
 neu/.style={boxbase,fill=neuf,draw=neul},
 dash/.style={boxbase,fill=none,draw=neul,dashed},
 ar/.style={-{Stealth[length=4.5pt]},gray,line width=0.7pt},
 ard/.style={ar,dashed},
}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{\textbf{MER-Cross (MER2026 Track-1)\\ Tổng hợp kiến trúc: sơ đồ, ý tưởng, cơ sở, kết quả}}
\author{}\date{}
\begin{document}\maketitle
\noindent\textbf{Bài toán.} Dự đoán cảm xúc người \emph{nghe} (listener) trong hội thoại tay đôi.
Train = 9\,395 mẫu Individual (audio/text/video cùng người, đang nói); test = 574 mẫu Interlocutor
(video = listener im lặng, audio/text = speaker khác). Metric = weighted-F1 (WAF), Codabench, nhiễu $\pm3$--4.
Baseline tốt nhất \textbf{v3 = 65.78}~\cite{mer2026}.

\medskip\noindent\textbf{Lưu ý.} Confusion matrix \& metric per-class là \emph{in-domain} (held-out train)
vì test không có nhãn. Màu khối trong sơ đồ: \textcolor{audiol}{audio},
\textcolor{textl}{text}, \textcolor{videol}{video/listener}, \textcolor{spkl}{speaker},
\textcolor{fusel}{fusion}; nét đứt = train-only/phụ.
"""

SUMMARY=r"""
\section*{Bảng tổng hợp kết quả (WAF test)}
\begin{table}[H]\centering\footnotesize
\begin{tabular}{lll}\toprule
\textbf{Kiến trúc} & \textbf{WAF test} & \textbf{Ghi chú} \\ \midrule
Pseudo-label (mix) & 66.0 & transductive, +0.2 \\
v3 ensemble (7-seed) & 65.80 & giảm variance \\
\textbf{v3 (best)} & \textbf{65.78} & Fix2+Fix3 \\
v6 CM-StEW & 65.36 & aux transfer \\
v11 deep-sup video & $\sim$65 & aux CE \\
v4/v5 fusion var. & $\sim$64 & \\
v2 visual-anchor & 63--64 & \\
v8 reliability-gate & 63 & \\
v7 conflict expose & 62--63 & \\
TMA/BBA & 61.47 & confirmation bias \\
FER backbone & 59.6--61 & domain gap \\
v9/v10 AU & 60--61 & observation-shift \\
MLLM SFT (+text) & 49.84 & in-dom 0.87 \\
MLLM video-only & 42.8 & in-dom 0.59 \\
MLLM zero-shot & 42 & \\
\bottomrule\end{tabular}
\caption{WAF Codabench (test, 574 mẫu). Trần thông tin $\sim$66 bền vững mọi trục.}
\end{table}
"""

def sec(a):
    aid,name,idea,why,refs,waf,cmk,dk=a
    parts=[r"\section*{%s \quad\normalsize\textnormal{[%s]} \hfill \small WAF: %s}"%(name,aid,waf)]
    if dk and dk in DIAG:
        spec,cap=DIAG[dk]; parts.append(tikz(spec,cap))
    parts+=[r"\textbf{Ý tưởng.}\ %s"%idea, r"\textbf{Vì sao.}\ %s"%why, r"\textbf{Tham khảo.}\ %s"%refs]
    if cmk and cmk in CMBY:
        d=CMBY[cmk]; parts.append(cm_table(d)); parts.append(per_table(d))
    elif not dk:
        pass
    else:
        parts.append(r"\emph{(Không có confusion matrix in-domain: cv npz không còn / pipeline riêng / đang chạy.)}")
    return "\n\n".join(parts)

doc=HEAD+SUMMARY+"\n"+"\n\n\\clearpage\n".join(sec(a) for a in A)+"\n\n"+BIB+"\n\\end{document}\n"
open('report_architectures.tex','w').write(doc)
print("wrote report_architectures.tex (%d chars)"%len(doc))
