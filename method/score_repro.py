import sys, json, importlib.util
p="/volume/exploration/EvolvingLMMs/lmms-eval/lmms_eval/tasks/coco_cap_chair/utils.py"
s=importlib.util.spec_from_file_location("cu",p);cu=importlib.util.module_from_spec(s);s.loader.exec_module(cu)
ans_f, q_f = sys.argv[1], sys.argv[2]
gt={json.loads(l)["question_id"]:json.loads(l)["gt_object"] for l in open(q_f)}
res=[]
for l in open(ans_f):
    d=json.loads(l); qid=d.get("question_id", d.get("image_id"))
    cap=d.get("text", d.get("caption",""))
    _,node,_,_=cu.caption_to_words(cap)
    res.append({"answer":gt.get(qid,[]),"pred":node})
cs=cu.coco_cap_chair_aggregate_results_chair_s(res);ci=cu.coco_cap_chair_aggregate_results_chair_i(res);rc=cu.coco_cap_chair_aggregate_results_recall(res)
al=sum(len(x['pred']) for x in res)/len(res)
print(f"n={len(res)} CHAIR_s={cs:.2f} CHAIR_i={ci:.2f} recall={rc:.2f} objs/cap={al:.2f}")
