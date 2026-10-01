# RetailRAG qwen3.7-plus Final Failure Audit

Total non-strict outcomes: **73**.

| Stage | Count | Percentage |
|---|---:|---:|
| FAQ_ROUTING | 20 | 27.40% |
| QUERY_REWRITE | 0 | 0.00% |
| QUERY_CLASSIFIER | 2 | 2.74% |
| STRATEGY_SELECTION | 0 | 0.00% |
| RETRIEVAL | 7 | 9.59% |
| RERANK | 0 | 0.00% |
| CONTEXT_BUDGET | 10 | 13.70% |
| VERSION_FILTER | 0 | 0.00% |
| EVIDENCE_GATE | 19 | 26.03% |
| GENERATION | 15 | 20.55% |
| MULTIPLE_STAGES | 0 | 0.00% |
| SYSTEM_ERROR | 0 | 0.00% |
| OTHER | 0 | 0.00% |

| ID | Type | Classification | Stage | Reason |
|---|---|---|---|---|
| single_001 | single_document | INCORRECT | FAQ_ROUTING | 实际回答未提及ground truth中核心的退款、换货及创建质量事件记录等针对顾客售后的处理措施，而是给出了隔离商品、记录照片及就医升级等无证据且不完全符合顾客售后场景的具体操作。由于缺乏检索证据支持且与ground truth严重不符，判定为INCORRECT。 |
| single_009 | single_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答完整覆盖了ground truth中的核心处理步骤（断电、按走纸方向取纸、不用尖锐工具），但额外添加了无证据支持的具体事实（恢复后打印测试页并扫码核验），不符合CORRECT必须没有无证据具体事实的规则。由于实际检索证据和final context均为空，faithfulness和context_recall均为0.0。 |
| single_013 | single_document | PARTIALLY_CORRECT | GENERATION | 实际回答准确提取了“原交易凭证”和“原因代码”，并补充了操作步骤中的核验信息，但遗漏了 ground truth 中明确指出的“可追溯的审批记录”这一重要材料/条件，因此属于部分正确。 |
| single_017 | single_document | INCORRECT | FAQ_ROUTING | 实际回答未提及基础积分在交易完成后24小时内到账的核心信息，仅说明生日月额外积分到账时间及基础积分时间不变，未给出基础积分的具体到账时间，未能准确回答用户问题。此外，实际检索证据为空，回答内容缺乏上下文支持，存在无证据的具体事实。 |
| single_023 | single_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答准确覆盖了ground truth的核心内容（验收后20分钟内完成入库或上架），但额外补充了“暂存期间不得放在常温通道”这一无证据的具体事实（实际检索证据和上下文均为空）。根据规则，包含无证据的具体事实不能计为严格的CORRECT，故判定为PARTIALLY_CORRECT。由于提供的上下文为空，faithfulness和context_recall均计为0.0。 |
| single_024 | single_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答覆盖了ground truth的核心内容，但额外增加了“修正后再次扫码抽查”这一无证据的具体事实。由于实际检索上下文为空，该额外信息属于幻觉，因此不能计为CORRECT，判定为PARTIALLY_CORRECT。同时，由于缺乏上下文支持，faithfulness和context_recall均为0.0。 |
| single_026 | single_document | PARTIALLY_CORRECT | FAQ_ROUTING | 回答正确指出了0至4摄氏度的温度范围，与ground truth一致，但引入了无证据的额外信息（如'演示温区'和'商品标签要求'），存在幻觉，因此判定为PARTIALLY_CORRECT。由于实际检索证据为空，context_recall为0.0；因回答包含无上下文支持的具体事实，faithfulness为0.5。 |
| single_027 | single_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答与ground truth完全一致，内容实质正确，但实际检索证据和最终上下文均为空，回答中的具体事实缺乏上下文证据支持，属于无证据的具体事实。根据规则1，不满足CORRECT的严格条件；因回答内容无遗漏和事实错误，故评为PARTIALLY_CORRECT。faithfulness和context_recall因上下文为空均计为0.0。 |
| single_034 | single_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答提到了“封袋”和“差异登记”，与 ground truth 中的“封袋并记录差异”部分一致，但遗漏了“双人清点”这一重要条件。同时，回答中提到的“核对 POS 日结、现金退款单和交接记录”等细节在 ground truth 和实际检索证据（context 为空）中均无依据，属于无证据的具体事实。因此，回答包含无证据事实且遗漏重要条件，判定为 PARTIALLY_CORRECT。由于未检索到相关文档且回答内容无上下文支持，faithfulness 和 context_recall 均为 0.0。 |
| keyword_002 | keyword_exact | PARTIALLY_CORRECT | GENERATION | 实际回答正确指出可以使用支付流水末四位查询订单，并提到了结合交易日期和门店等信息查找近90天电子订单，但遗漏了ground truth中明确指出的“会员手机号”这一重要组合条件，仅用“等信息”概括，属于遗漏重要条件，因此判定为PARTIALLY_CORRECT。回答内容均来源于检索到的上下文，且上下文完整覆盖了所需信息。 |
| keyword_016 | keyword_exact | WRONG_REFUSAL | EVIDENCE_GATE | 问题询问金卡会员积分累计口径中‘净实付’的相关内容，期望行为为 answer，且 ground truth 提供了基于净实付的积分累计规则。实际检索的 context 中明确包含了该规则（金卡会员按净实付每满1元累计1.5积分；不足1元部分不累计），但模型以‘未明确定义具体含义’为由直接拒绝回答，未能输出 ground truth 中的有效信息。属于有足够证据却拒答，因此判定为 WRONG_REFUSAL。 |
| keyword_017 | keyword_exact | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答准确回答了核心问题（每单最多抵扣应付金额的20%），与ground truth一致。但回答中包含了“100积分可抵1元”和“积分不能提现、转赠或兑换找零”等无检索证据支持的具体事实，违反了CORRECT必须没有无证据具体事实的规则。因此判定为PARTIALLY_CORRECT。由于实际检索上下文为空，faithfulness和context_recall均为0.0。 |
| paraphrase_002 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 检索到的文档中明确说明了‘不同文件口径不一致时，登记原始现象并转值班经理或业务支持’，这与问题‘手头的资料跟公司制度对不上’及ground truth完全对应。系统有充足证据回答该问题，但实际回答却以‘知识库信息不足’为由拒绝回答，属于有足够证据却拒答，因此判定为WRONG_REFUSAL。 |
| paraphrase_006 | semantic_paraphrase | INCORRECT | FAQ_ROUTING | 实际回答描述的是商品已过期或发生食品安全事件时的应急处理流程（如停止销售、隔离、提交事件等），而ground truth明确指出对于‘快过期’的商品，处理流程应为提前扫描、根据系统建议折价或下架、移至指定区域并更换合规价签。实际回答与ground truth完全不符，且实际检索上下文为空，未能召回相关证据，因此判定为INCORRECT。 |
| paraphrase_008 | semantic_paraphrase | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答正确指出了部分退货通常不退券，覆盖了ground truth的核心内容。但回答遗漏了ground truth中提到的'券已过期'这一重要条件，且补充了'只有整单取消且券仍有效时才按券规则原路退回'这一无证据的具体事实（实际检索context为空）。由于包含无证据内容和遗漏条件，判定为PARTIALLY_CORRECT。因未检索到任何相关文档，context_recall为0。 |
| paraphrase_014 | semantic_paraphrase | PARTIALLY_CORRECT | GENERATION | 回答准确指出了现金差异超过20元需要立即上报，并补充了禁止私下垫付的规定，内容忠实于检索证据且context完全覆盖了所需信息。但回答遗漏了ground truth中提到的另外两个需要立即上报的重要条件：连续出现差异或疑似舞弊的情况，未能完整覆盖所有需立即上报的触发条件，因此计为部分正确。 |
| paraphrase_020 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的上下文中明确包含“开店人员至少两人共同完成重点区域检查”以及“单人开店属于排班异常，应联系值班经理”的信息，足以完整回答该问题。但模型回答称知识库信息不足并拒绝回答，属于有足够证据却拒答，因此判定为 WRONG_REFUSAL。 |
| multi_001 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 回答准确覆盖了换号与账户合并的核验、工单发起、合并规则及处理时限等内容，且未编造事实。但遗漏了ground truth中关于人工补积的具体操作（创建会员调整工单上传原交易凭证）以及补积超500分或涉及等级变更需升级区域运营审批的重要条件，并对补积部分以知识库信息不足为由进行了拒答。因遗漏重要条件，不能计作严格正确，判定为部分正确。 |
| multi_003 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 模型正确回答了第一问（整单取消且新客券在有效期内可原路退回），但未回答第二问（是否可与其他促销叠加），而是以知识库信息不足为由拒绝。根据 ground truth，第二问有明确答案（需以券面‘可叠加’标识为准），且检索到的《member_day_birthday》中也提及了‘按券面说明’执行叠加规则，模型未能有效利用此线索或检索到《promotion_stacking》文档，导致遗漏了重要信息。因此判定为 PARTIALLY_CORRECT。Faithfulness 为 1.0，因为已回答内容均有 context 支持且未编造；Context recall 为 0.5，因为仅覆盖了 ground truth 中关于退回规则的部分，遗漏了叠加规则的部分。 |
| multi_008 | multi_document | PARTIALLY_CORRECT | RETRIEVAL | 模型正确回答了已知的积分退回规则，但未能提供赠品无法退回时的扣减规则以及积分按退货商品分摊比例返还和到期日处理规则，遗漏了重要信息，因此判定为 PARTIALLY_CORRECT。 |
| multi_012 | multi_document | INCORRECT | FAQ_ROUTING | 实际回答未明确回答'能否直接退款'的核心问题，遗漏了'不能凭口述直接退款'的结论以及'无法唯一定位时需登记联系方式并转人工'的关键操作。此外，实际回答中提到的'会员手机号、支付流水末四位、近90天'等细节在给定上下文中无证据支持，且与ground truth不符，包含无证据的具体事实。 |
| multi_013 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 实际回答准确提取了《cash_handover》中关于现金差异超过20元的处理规则，包括升级区域运营、禁止垫付/冲抵及具体操作步骤，内容均有据可查，无幻觉（faithfulness=1.0）。但回答遗漏了ground truth中基于《门店闭店检查清单》的“须在离店前上报”这一针对闭店场景的重要条件，且未提及“值班经理复核”，属于遗漏重要条件，因此判定为PARTIALLY_CORRECT。context_recall因遗漏部分ground truth要点计为0.8。 |
| multi_014 | multi_document | PARTIALLY_CORRECT | GENERATION | 实际回答准确指出了支付系统未结算造成的账面差异不计入现金长短款，且应单独标记，这部分内容忠实于检索到的《cash_handover》文档。但遗漏了ground truth中提到的《门店闭店检查清单》中关于‘保留错误码，不得重复提交造成账务重复’的重要补充条件，因此属于部分正确。 |
| multi_018 | multi_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答正确指出了不能直接修改数量及未经审批不得手工平账，并提及核对相关单据，但遗漏了锁定相关库存、由不同人员二次计数以及达到上报门槛须提交区域运营审批或按事件上报等重要处理条件，因此属于部分正确。回答内容忠实于ground truth，无无证据事实，faithfulness为1.0；因未提供检索上下文，context_recall不适用，返回null。 |
| multi_019 | multi_document | PARTIALLY_CORRECT | GENERATION | 实际回答准确指出了必须使用系统动态盘点模式以及恢复网络后仅提交一次离线数据，但遗漏了 ground truth 中强调的“在登录盘点终端后先核对门店、任务编号和设备时间”这一重要的设备同步前置条件，因此不能计作严格正确。检索到的上下文完整包含了回答问题所需的所有信息，且回答内容均忠实于上下文，无幻觉。 |
| multi_020 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 回答准确覆盖了记录错误码、停止重复提交以及完成未提交单据等关键步骤，且内容均忠实于检索到的上下文。但遗漏了ground truth中强调的“差异复盘后由店长确认提交”以及“不得使用离线表格替代系统记录”等重要条件，因此判定为部分正确。 |
| multi_021 | multi_document | INCORRECT | FAQ_ROUTING | 实际回答仅说明了冷藏温区标准，完全没有回答发现超温后的具体处理流程（如隔离待判、不得上架、停止补货、记录超温时长、评估报损等），遗漏了所有关键处理步骤，且未基于任何检索证据。 |
| multi_022 | multi_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答与ground truth完全一致，内容实质正确。但实际检索证据为空，回答中的具体事实（20分钟、常温通道）缺乏context支持，属于无证据的具体事实，不符合CORRECT必须没有无证据具体事实的严格条件，因此降级为PARTIALLY_CORRECT。由于无检索证据，faithfulness和context_recall均为0.0。 |
| multi_025 | multi_document | PARTIALLY_CORRECT | GENERATION | 实际回答正确指出了处理卡纸的首要步骤（断电、按走纸方向取出、不使用尖锐工具）以及价签未打印时的合规措施（暂停陈列或使用备用流程），但遗漏了卡纸处理后的关键后续步骤（确认标签规格、打印测试页并扫码核验、若仍异常则停用报修），同时也未提及确保商品、货位标签和POS售价一致的核心要求。存在重要条件遗漏，判定为部分正确。回答内容均来源于检索文档，无幻觉，faithfulness为1.0；context_recall因遗漏部分关键步骤和规则，评为0.6。 |
| multi_026 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 实际回答覆盖了 ground truth 中关于“暂停陈列与核验”和“保留设备错误码并联系值班经理”的措施，但遗漏了“先取消错误的打印任务并核对模板，不得重复张贴错误标签”这一重要步骤，因此判定为 PARTIALLY_CORRECT。回答内容均来源于检索到的 context，无无证据的幻觉，faithfulness 为 1.0。检索到的 context 覆盖了 ground truth 中约 2/3 的关键信息，context_recall 为 0.67。 |
| multi_031 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 实际回答正确指出了依据基础制度、《促销叠加与结算顺序》、券面说明和《会员日活动执行通知》等资料，内容均能在检索到的上下文中找到依据。但回答遗漏了ground truth中明确要求的《内部知识维护与冲突处理规范》及其关于限时内部通知覆盖基础制度的原则，属于遗漏重要条件，因此判定为PARTIALLY_CORRECT。 |
| multi_033 | multi_document | PARTIALLY_CORRECT | FAQ_ROUTING | 实际回答正确指出了可获得满100减10券及以系统发放为准，但遗漏了ground truth中‘次月到期’的重要条件。此外，由于实际检索上下文为空，回答中提到的‘3至5月的每月8日’属于无证据的具体事实。因此判定为部分正确。 |
| faq_008 | faq | WRONG_REFUSAL | EVIDENCE_GATE | 实际 final context 中的《member_levels》文档明确包含了等级按月复核、降级执行时间以及临时保级规则，有充足证据回答‘系统自动降级后还能恢复吗’的问题。但模型却以知识库信息不足为由拒绝回答，属于有足够证据却拒答的情况，因此判定为 WRONG_REFUSAL。 |
| faq_010 | faq | PARTIALLY_CORRECT | RETRIEVAL | 实际回答准确提取了检索上下文中的信息，区分了不同会员等级（金卡1.5分，普通/银卡1分）的积分规则，并说明了不足1元不累计的条件，内容完全忠实于提供的上下文，faithfulness为1.0。然而，回答遗漏了ground truth和上下文中明确指出的重要条件，即“礼品卡、运费等排除项不累计积分”，因此不能计为严格正确，判定为PARTIALLY_CORRECT，context_recall为0.8。 |
| faq_011 | faq | PARTIALLY_CORRECT | RETRIEVAL | 实际回答准确提取了不同会员等级的积分累计规则及基础条件（净实付、不足1元不累计），且内容均忠实于检索文档。但遗漏了 ground truth 中明确指出的“礼品卡、运费等排除项不累计”这一重要限制条件，因此判定为 PARTIALLY_CORRECT。 |
| faq_012 | faq | PARTIALLY_CORRECT | RETRIEVAL | 实际回答准确提供了不同会员等级（金卡、普通/银卡）的积分累计规则以及不足1元不累计的条件，内容均来源于检索证据，无幻觉。但遗漏了 ground truth 中强调的“礼品卡、运费等排除项不累计”这一重要限制条件，因此不能计为严格正确，判定为 PARTIALLY_CORRECT。 |
| faq_018 | faq | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的上下文（《points_rules》）中明确包含了积分折算规则（100积分可抵扣1元，每单最多抵扣应付金额的20%，不可提现、转赠或兑换找零），有充足证据回答该问题。但模型却以知识库信息不足为由拒绝回答，属于有足够证据却拒答的情况，因此判定为 WRONG_REFUSAL。由于未利用上下文信息，context_recall 为 0.0；由于未产生幻觉或错误事实，faithfulness 为 1.0。 |
| faq_024 | faq | INCORRECT | FAQ_ROUTING | 实际回答与ground truth完全一致，内容本身正确。但实际检索证据和最终上下文均为空，回答包含了无证据的具体事实，属于模型幻觉，违反了RAG系统必须严格基于给定上下文回答的原则。根据规则1，包含无证据的具体事实不能计为CORRECT，且无遗漏或局部错误也不属于PARTIALLY_CORRECT，故判定为INCORRECT。faithfulness为0.0，context_recall因无上下文不适用返回null。 |
| version_007 | version_conflict | INCORRECT | FAQ_ROUTING | 实际回答未提及现行规则下金卡会员的基础积分倍数（1.5倍），而是回答了生日月活动的积分倍数（2倍），属于答非所问且包含无证据的具体事实，因此判定为 INCORRECT。由于未检索到相关证据且回答内容无上下文支持，faithfulness 和 context_recall 均为 0。 |
| version_017 | version_conflict | PARTIALLY_CORRECT | GENERATION | 回答准确提供了基础积分规则（每1元1.5积分，2026年1月1日起生效），并补充了context中存在的生日月积分规则，内容均有据可查（faithfulness=1.0）。但遗漏了ground truth中提到的'2025年1.0版已被替代'这一重要信息，未能完整覆盖ground truth的所有要点，因此判定为PARTIALLY_CORRECT。 |
| version_020 | version_conflict | PARTIALLY_CORRECT | GENERATION | 模型准确回答了现行金卡积分倍率为每1元累计1.5积分，但未能回答与旧规则的比较或具体变化量，且遗漏了旧版本已被替代的相关信息，未能完整覆盖问题中的比较条件，因此判定为部分正确。 |
| multiturn_002 | multiturn | INCORRECT | FAQ_ROUTING | 实际回答未提及按商品在原订单中分摊后的实付金额计算退款，而是给出了查询订单和提交工单的流程，与ground truth完全不符。且实际检索证据为空，未召回相关文档。 |
| multiturn_003 | multiturn | PARTIALLY_CORRECT | GENERATION | 实际回答准确提取了检索文档中关于“重复账户”的合并规则、操作步骤及例外情况，内容完全忠实于给定证据，且充分召回了文档中关于重复账户的所有相关信息。然而，回答未包含 ground truth 中指定的“无法验证原号时需转人工工单核验”这一关键信息（尽管该信息在文档中实为换号规则），未能完整覆盖 ground truth 的要求，因此判定为 PARTIALLY_CORRECT。 |
| multiturn_006 | multiturn | PARTIALLY_CORRECT | GENERATION | 实际回答提供了超过门槛后的升级审批、禁止手工平账等正确信息，但完全遗漏了 ground truth 中要求的“锁定相关库存，避免继续移动”这一关键操作步骤，因此属于部分正确。检索到的上下文包含了 ground truth 所需信息，且回答内容忠实于上下文。 |
| multiturn_007 | multiturn | WRONG_REFUSAL | QUERY_CLASSIFIER | 期望行为为回答，且ground truth提供了具体的打印机故障处理步骤（断电、按走纸方向取纸、禁用尖锐工具）。实际检索为空，模型以“知识库信息不足”为由拒绝回答具体问题，仅提供通用常识，属于有足够证据（依据ground truth）却拒答的情况，判定为WRONG_REFUSAL。context_recall为0.0，因未召回任何ground truth信息；faithfulness为1.0，因模型未捏造context中的事实。 |
| multiturn_009 | multiturn | PARTIALLY_CORRECT | GENERATION | 实际回答正确指出了按三级事件立即升级并联系区域运营，但遗漏了ground truth中强调的“优先采取人员保护措施”以及“若存在紧急危险，应先报警或联系急救”的重要条件，因此判定为PARTIALLY_CORRECT。检索到的文档包含了所需信息，faithfulness和context_recall均较高。 |
| multiturn_010 | multiturn | WRONG_REFUSAL | QUERY_CLASSIFIER | 期望行为是回答问题，且ground truth中存在明确答案。虽然实际检索上下文为空，但系统未能提供答案而是直接拒答，属于WRONG_REFUSAL。由于未生成具体事实内容，faithfulness为1.0，context_recall为0.0。 |
| scale_single_document_195 | single_document | INCORRECT | FAQ_ROUTING | 实际回答与ground truth完全一致，内容本身正确。但实际检索证据为空（final context为无），回答中的具体事实完全缺乏检索证据支撑，属于无证据的具体事实（幻觉），违反了CORRECT必须没有无证据具体事实的规则，因此判定为INCORRECT。由于context为空，faithfulness和context_recall均为0.0。 |
| scale_keyword_exact_220 | keyword_exact | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的文档《cash_handover》和《closing_checklist》中包含了封袋或交接前需要核对的单据信息（如班结单/POS日结、退款单、优惠凭证、未结订单等），有足够证据回答问题。但模型却以知识库信息不足为由拒绝回答，属于有足够证据却拒答，因此判定为 WRONG_REFUSAL。 |
| scale_semantic_paraphrase_225 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 问题的期望行为为 answer，且 ground truth 中明确给出了答案（已出库在途调拨不计入收货店可售库存，验收确认后才可用）。实际回答以知识库信息不足为由完全拒绝了回答，属于把全部问题拒绝。根据规则4，有足够证据（ground truth）却拒答或把全部问题拒绝属于 WRONG_REFUSAL。同时，实际检索到的 context 中未包含在途调拨库存计算的相关证据，context_recall 为 0.0；因未生成具体事实，faithfulness 不适用返回 null。 |
| scale_semantic_paraphrase_226 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 期望行为为answer且存在明确的ground truth，但模型未能检索到相关调拨运输文档（如transfer_dispatch），导致错误地拒绝了回答，属于WRONG_REFUSAL。由于未生成实质性回答内容，faithfulness不适用返回null，context_recall为0.0。 |
| scale_semantic_paraphrase_231 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索证据中明确包含‘先在系统核验自提码有效且未核销，再核对订单商品，不仅凭截图交货’的业务规则，完全足以回答客人仅凭提货码图片能否拿走货的问题。但实际回答却以知识库信息不足为由拒绝回答，属于有足够证据却拒答的情况，因此判定为WRONG_REFUSAL。 |
| scale_semantic_paraphrase_232 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 问题询问预约取货迟到的处理方式，ground truth 提供了明确答案，期望行为为 answer。实际检索未能召回相关文档（online_pickup），导致系统以知识库信息不足为由拒绝回答。由于问题本身可回答且期望行为为回答，系统的拒答属于 WRONG_REFUSAL。回答未包含幻觉，faithfulness 为 1.0；未召回相关文档，context_recall 为 0.0。 |
| scale_semantic_paraphrase_236 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 期望行为为answer且ground truth有明确答案（禁止抄录完整银行卡号或敏感验证信息，无法安全结算时暂停交易），但模型以知识库信息不足为由直接拒绝回答，未提供实质性解答，属于错误拒答（WRONG_REFUSAL）。 |
| scale_semantic_paraphrase_239 | semantic_paraphrase | PARTIALLY_CORRECT | RETRIEVAL | 回答结论正确，但引入了上下文中未明确提及的“调拨”概念（属于无证据的具体事实），且遗漏了ground truth中“禁止赠送”的重要条件。此外，系统未检索到直接包含完整答案的参考文档（recall_isolation），仅依靠部分相关文档（clearance_return）中的“隔离”规定进行推断，context recall较低。 |
| scale_semantic_paraphrase_240 | semantic_paraphrase | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的上下文中明确包含了“记录在库隔离数、已售可追踪数和已回收数，三类数量分别登记”的信息，足以回答召回统计需要分哪几种数量以及不能只报剩余库存的问题。但模型却以知识库信息不足为由拒绝回答，属于有足够证据却拒答的情况，因此判定为 WRONG_REFUSAL。 |
| scale_hard_confusion_242 | hard_confusion | WRONG_REFUSAL | EVIDENCE_GATE | 期望行为为回答（answer），且存在明确的 ground truth（以当前实物包装标示为准）。但实际检索未能召回相关文档（allergen_label），导致模型以知识库信息不足为由拒绝回答。由于期望行为是回答而非拒绝，此拒答属于 WRONG_REFUSAL。模型回答忠实于其检索到的上下文（未编造事实），但未能召回相关证据。 |
| scale_hard_confusion_245 | hard_confusion | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的上下文中包含了《shelf_life》文档，其中明确规定“生产日期或保质期无法辨认的食品先隔离核验，不自行补写日期后销售”，这足以回答用户关于“袋子日期磨花了，按同箱其他袋补写再上架行吗”的问题。然而，模型却以“知识库信息不足”为由拒绝回答，属于有足够证据却拒答的情况，因此判定为 WRONG_REFUSAL。由于模型未生成实质性回答，faithfulness 为 null，context_recall 为 0.0。 |
| scale_hard_confusion_247 | hard_confusion | PARTIALLY_CORRECT | GENERATION | 实际回答正确引用了‘先到者’的规则，覆盖了ground truth的核心内容。但其直接断言‘按开封后建议三天内吃完的期限执行’属于局部错误，因为原到期日可能比三天更近，不能一概而论。因此判定为PARTIALLY_CORRECT。faithfulness因包含无证据的具体结论而扣分，context_recall为1.0。 |
| scale_hard_confusion_250 | hard_confusion | WRONG_REFUSAL | EVIDENCE_GATE | 期望行为为 answer 且存在明确的 ground truth，但模型以知识库信息不足为由拒绝回答，属于错误的拒答（WRONG_REFUSAL）。实际检索未能召回相关文档（如 serial_trace），导致 final context 中缺乏回答问题所需的证据，context_recall 为 0.0。模型基于缺失的 context 选择拒答，未编造事实，faithfulness 为 1.0。 |
| scale_hard_confusion_251 | hard_confusion | WRONG_REFUSAL | EVIDENCE_GATE | 期望行为为 answer，但实际回答选择了完全拒答。虽然检索到的上下文中缺乏直接回答该问题的具体证据（仅提及需核验附件，未说明无清单时能否参照相似型号），但根据规则，期望回答却把全部问题拒绝属于 WRONG_REFUSAL。 |
| scale_faq_260 | faq | PARTIALLY_CORRECT | RETRIEVAL | 实际回答明确给出了“不能先贴出去”的正确结论，并基于检索文档提供了合理的后续处理建议（如停用报修、暂停陈列），内容忠实于上下文。但遗漏了 ground truth 中提到的具体排查步骤（检查打印浓度、条码完整性和分辨率），因此属于部分正确。 |
| scale_faq_263 | faq | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的《pos_troubleshooting》文档中明确包含‘支付结果不明时先查询原交易状态’、‘不连续重复扣款’等规则，提供了充足的证据来回答‘不能一键全重新收钱，需先查询原交易流水状态’。模型在有足够证据的情况下选择了拒答，属于WRONG_REFUSAL。 |
| scale_faq_264 | faq | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的上下文中包含“不得安装未知软件”以及“切换备用机并提工单”等相关规定，足以支撑“不能安装私人热点工具（属于未知软件）并应提工单使用备用方案”的回答。模型却以知识库信息不足为由拒绝回答，属于有足够证据却拒答，因此判定为 WRONG_REFUSAL。由于未利用上下文中的相关信息，context_recall 为 0.0；回答本身未编造事实，faithfulness 为 1.0。 |
| scale_multi_document_266 | multi_document | PARTIALLY_CORRECT | GENERATION | 实际回答计算出基础积分150分，生日月活动积分300分，合计450分。但根据ground truth，100元合计应为300分（即总积分为基础积分的2倍）。实际回答对文档中“按金卡基础积分的2倍发放”的理解与ground truth存在偏差，导致最终合计结果错误，包含局部错误，因此判定为PARTIALLY_CORRECT。回答内容均基于检索到的文档进行推导，无捏造事实，且检索到了所有相关文档，故faithfulness和context_recall均为1.0。 |
| scale_multi_document_267 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 银卡积分计算正确（100分），金卡基础积分计算正确（150分），但对生日月积分规则理解错误，将文档中的“按金卡基础积分的2倍发放”误解为“额外发放2倍”，导致金卡总积分计算错误（实际回答为450分，正确应为300分）。包含局部事实错误，判定为PARTIALLY_CORRECT。 |
| scale_multi_document_268 | multi_document | PARTIALLY_CORRECT | RETRIEVAL | 回答正确指出了两张订单券不能同时使用，但遗漏了关于积分活动倍数取高不累乘及金卡基础积分按现行口径的重要信息，仅表示知识库信息不足，属于遗漏重要条件，因此判定为PARTIALLY_CORRECT。回答内容忠实于检索到的上下文，未产生幻觉，但检索到的上下文未能覆盖积分倍数叠加的具体规则，导致context_recall较低。 |
| scale_multi_document_282 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 回答准确回答了退款金额的计算方式（按分摊后实付退），与检索到的证据一致。但对于发票调整部分，由于检索到的上下文中缺乏相关证据，模型选择了拒答并建议联系经理。由于期望行为是回答完整问题，且遗漏了发票调整这一重要部分，因此判定为 PARTIALLY_CORRECT。Faithfulness 为 1.0，因为回答内容均忠实于提供的上下文；Context recall 为 0.5，因为上下文仅覆盖了 ground truth 中关于退款的部分，缺失了发票调整的相关信息。 |
| scale_multi_document_284 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 回答明确拒绝了同事的提议，准确引用了禁止员工私下垫付或从下一班现金中冲抵差异的规定，并提供了正确的交接与差异处理步骤（如冻结当前钱箱）。但回答遗漏了ground truth中强调的‘备用金与营业款分记’这一重要财务原则，且未明确点出‘封袋号不符暂停启用’的具体表述，因此判定为部分正确。 |
| scale_multi_document_286 | multi_document | PARTIALLY_CORRECT | CONTEXT_BUDGET | 回答正确指出了秒杀商品不能叠加会员日折扣，核心结论正确且无幻觉。但回答未明确说明‘夏季通知已过期’（仅建议核对），且完全遗漏了ground truth中关于‘秋季试点也沿用该排除’的关键信息，属于遗漏重要条件，因此计为部分正确。 |
| scale_multi_document_290 | multi_document | WRONG_REFUSAL | EVIDENCE_GATE | 实际检索到的上下文《transfer_dispatch》中明确说明该文件是“专题补充”、“不将本文件视为全部流程的授权”，且规定了“非原拣货人复核...不能拣货人自复核”（即经办与审批分离），结合文件状态为active，有足够证据回答补充通知不重置整套制度以及经办与审批分离的要求。模型却以信息不足为由拒绝回答，属于有足够证据却拒答，判定为WRONG_REFUSAL。 |
| scale_version_conflict_297 | version_conflict | PARTIALLY_CORRECT | GENERATION | 模型正确指出了秒杀商品不叠加秋季试点9折和常规95折，提供了实质正确的部分信息。但未能明确回答“不能沿用”的核心结论，也未提及“夏季特例已过期”的关键原因，而是以“无法确认”拒答了核心问题。因此属于部分正确。 |
| scale_hard_confusion_327 | hard_confusion | PARTIALLY_CORRECT | GENERATION | 回答正确指出了每月预约次数为2次，并适当拒绝了无法提供的员工排班信息，但遗漏了ground truth中提到的'每次最长30分钟'这一重要条件，未能完整覆盖可回答部分，因此属于部分正确。 |
