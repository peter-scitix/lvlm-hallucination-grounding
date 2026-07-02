import sys, json, importlib.util
p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
s=importlib.util.spec_from_file_location("cu",p);cu=importlib.util.module_from_spec(s);s.loader.exec_module(cu)
gt=json.load(open("/volume/exploration/EvolvingLMMs/saliency_repro/cocoid_gt.json"))
res=[]
for l in open(sys.argv[1]):
    d=json.loads(l); cid=str(d["image_id"]); cap=d.get("caption",d.get("text",""))
    if cid not in gt: continue
    _,node,_,_=cu.caption_to_words(cap); res.append({"answer":gt[cid],"pred":node})
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
print(f"n={len(res)} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs/cap={sum(len(x['pred']) for x in res)/len(res):.2f}")
