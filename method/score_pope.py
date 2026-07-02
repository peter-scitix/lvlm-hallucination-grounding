import sys, json, re
def yn(t):
    t=t.strip().lower()
    return 1 if (t.startswith("yes") or re.search(r"\byes\b",t)) else 0
tp=fp=tn=fn=0; n=0
for l in open(sys.argv[1]):
    d=json.loads(l)
    labs=d["label"] if isinstance(d["label"],list) else [d["label"]]
    anss=d["ans"] if isinstance(d["ans"],list) else [d["ans"]]
    for gt,a in zip(labs,anss):
        p=yn(a); n+=1
        if gt==1 and p==1: tp+=1
        elif gt==1 and p==0: fn+=1
        elif gt==0 and p==1: fp+=1
        else: tn+=1
acc=(tp+tn)/max(n,1); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
print(f"n={n} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} yes_rate={(tp+fp)/max(n,1):.3f}")
