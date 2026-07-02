# 方法总览:训练无关的 LVLM object 幻觉「检测 → 消除」(logit-lens 一脉相承)

**模型/评测:** LLaVA-1.5-7B (llava-hf);CHAIR(500 张 COCO val2014,greedy,max_new_tokens=512,prompt
"Please describe this image in detail.")、POPE(标准 yes/no)。全程 **training-free**(不训练任何东西)。
cal=偶数图 / test=奇数图(校准量如 base-rate 名单只在 cal 上定,test 上评,不偷看)。

---

## 一、检测(logit-lens grounding,不变)
**直觉:** 一个真实 object,图里总有若干 visual token(patch)"指向"它;一个幻觉 object,没有任何 patch 真的指向它。

**机制(per-image,per-object):** 对被提及的 object 词 o,
- 取 576 个 visual token 在最后一层(L31)的隐状态 h_vis,RMSNorm 后与 o 的 lm_head 词向量 W_U[o] 做 cosine;
- grounding 分 g(o) = 这 576 个 cos 的 **top-k 平均**;per-object 校准 gc(o) = g(o) − mean_o(减掉该 object 的均值,消掉词频先验)。
- gc(o) 低 = 没有 patch 指向它 = 幻觉。

**效果(已验证):** 500 图、~1681 个被提及 object(27% 是幻觉),**检测 AUROC 0.82**(raw 0.79 / 校准后 0.82)。
验证过是真的 per-image grounding(单纯 object 先验只有 0.60;词频-幻觉相关≈0)。cosine 版有效,softmax-prob 版无效(0.52)。

**两个信号:** 我们最初提的 (1) 支撑 patch 内部松散(pairwise sim 低)、(2) logit-lens cos 对所有候选 object 都低。
实测 **信号#2(grounding cos)是主力(0.82);信号#1(looseness)是真的但不额外加分**(和#2冗余)。

**一个额外发现(更强的检测器):** 直接问模型"图里有 X 吗"(self-verification)检测幻觉 **AUROC 0.89**(7B)/ 0.895(13B),
比 grounding(0.82)和监督竞品(0.88)都强 —— 说明"**模型自己知道,只是生成时没说出来**"(generation-verification gap)。
这是一个 clean 的发现,但**它是"问模型"这个不同机制,不是 grounding 的进化**。主方法的检测仍用 grounding(和消除一脉相承)。

**检测在 POPE 上的行为(诚实说清):** grounding 作为"物体在不在"检测器,在 POPE 三 split 上 AUROC = **0.81 / 0.82 / 0.94**(adv/pop/rand)
—— **信号是真的,而且不弱**(证明 grounding 是通用的物体存在性检测器,不是 CHAIR 场景偶然)。**但对 POPE 判别"无增量"**:
grounding 和模型自身 yes/no 组合时最优 λ=0(三 split 皆然),因为**模型自身判别已把该信号包含(0.84-0.87),grounding 冗余**。
=> 检测信号在 POPE 上"能检测但无用武之地",用武之地在生成场景(CHAIR)的消除。

---

## 二、消除(视觉源头中和 —— 我们自己的方法)
**直觉(关键 insight):** 既然 grounding 能定位"支撑幻觉 object 的那些 visual token",那就**在视觉源头把它们中和掉**,
而不是在输出端删词/压 logit。编辑输入侧 → 模型自然重新生成 → **caption 天然通顺**,且只动幻觉的支撑 patch → **真实 object 不受影响(recall 保住)**。

**步骤:**
1. **Pass 1:** 正常生成 caption。用检测(grounding gc)找出高置信幻觉 object;对**低幻觉率的类(person/伞/球等,cal 上幻觉率<0.12)做保护**(不移除 —— 它们几乎都是真的,误删会崩 recall)。
2. **object → visual-token retrieval:** 对每个幻觉 o,取 cos(LN(h_vis[p]), W_U[o]) 最高的 top-k 个 patch —— 即"声称自己是 o"的视觉源头。
3. **中和:** 在 multi_modal_projector 输出上把这些 patch 的 embedding 清零(forward hook);可选同时 ban 掉 o 的输出 token(保证不吐)。
4. **Pass 2:** 重新生成 → 视觉证据没了 → 模型不再说 o,自然说出通顺的替代。

**效果(test split,n=250,已验证):**

| 方法 | CHAIR_s | CHAIR_i | recall | 通顺度 |
|---|---|---|---|---|
| baseline | 52.4 | 15.7 | 75.0 | 基准 |
| 视觉中和 (rk20) | 45.6 | 13.0 | **74.6**(仅 −0.4) | = baseline |
| **视觉中和 + ban (rk20)** | **43.6** | **12.5** | 74.0 | = baseline |

- **CHAIR_s −8.8,CHAIR_i −3.2,recall 只掉 1,caption 通顺**(fluency ≈ baseline;删名词式切除只有 2.5/5,我们的重生成 ≈ baseline 的 4.9/5,GPT-judge 盲评验证过)。

---

## 三、这方法真有效吗?—— 诚实确认
**有效,而且匹配公平比较下打赢竞品:**
- **匹配 recall 下**(这是关键 —— 大家常在不同 recall 报数字):在 recall≈70 处,我们的检测驱动移除给 **CHAIR_s 27.7 / CHAIR_i 7.6**,
  **打过我们忠实复现的 PAI(31.2 / 10.2)**;也低于 ICLR Oral 对手声称的 35.6(而且对手的 headline **不可复现** —— 其 LocoRE 代码缺失、SGRS 是 no-op,我们实测证过)。
- **POPE:** 我们的消除是对 caption 的后处理,不碰 yes/no 判别 → **POPE = baseline(0.840/0.868/0.870),不损**。(标准 POPE 上任何 training-free 方法都 no-op,是这个 benchmark 的特性。)

**必须对师兄诚实说清的三点(别被别人挑穿):**
1. **量级中等,不是 SOTA 碾压** —— −8.8 CHAIR_s、匹配 recall 下小胜 PAI,不是数量级领先。
2. **有个根本天花板** —— 我们证明了:任何 training-free、检测驱动的消除,其 (CHAIR_i, recall) 可达 frontier **恰好等于检测器的 ROC 曲线**,
   与干预机制(压 logit / 删词 / 视觉中和)无关。所以谁都突破不了检测精度这条线;我们的方法是这条线上**最优雅、最通顺、recall 最稳**的实现,不是打破它。
3. **范围** —— 目前是 LLaVA-1.5-7B + CHAIR(检测在 13B 也验证了 0.895)。要发表需补:另一个 LVLM(InstructBLIP/Qwen-VL)+ 另一个 benchmark(AMBER)。

---

## 四、一句话总结(给师兄)
**训练无关,一条 logit-lens 贯穿检测与消除:** 用 logit-lens grounding 检测幻觉 object(AUROC 0.82),
再用同一 logit-lens 定位并**中和其视觉源头 patch**,重生成得到**通顺、recall-safe** 的干净 caption(CHAIR_s −8.8,匹配 recall 下胜 PAI,POPE 不损)。
配套一个理论结论:**幻觉消除本质是 object mention 上的 selective prediction,可达 frontier = 检测器 ROC**,并发现**模型自我验证(0.89)是可得最强检测信号(generation-verification gap)**。
诚实边界:量级中等 + 受检测精度 frontier 根本约束 + 目前单模型单 benchmark。

**脚本:** 检测 detect/logit_lens_probe*.py + method/excise.py;消除 method/vtablate.py(--rk/--ban);
理论/复现 THEORY.md、COMPETITOR_ANALYSIS.md;self-verify method/selfverify.py。
