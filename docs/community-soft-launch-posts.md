# Community Soft Launch Posts

These drafts are ready for maintainers to publish manually after checking the
live Dashboard. Do not coordinate votes, cross-post every community at once, or
describe maintainer-operated Kaggle Cells as independent community machines.

## Launch Links

- Website: <https://crowdtensor.24.199.118.54.nip.io>
- Repository: <https://github.com/Ffffffffchopin/CrowdTensor>
- Live Dashboard: <https://crowdtensor.24.199.118.54.nip.io/v1/volunteer/dashboard>
- Founding Beta enrollment: <https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=beta_enrollment.yml>
- Draft 7B RFC: <https://github.com/Ffffffffchopin/CrowdTensor/blob/main/docs/campaigns/qwen25-7b-gsm8k-rfc.md>
- Qwen2.5-7B evidence: <https://github.com/Ffffffffchopin/CrowdTensor/blob/main/docs/qwen7b-gsm8k-elastic-showcase.md>

Before publishing, open every link in a logged-out browser and verify the live
snapshot. Keep the text below unchanged unless a newer checked artifact
supersedes its numbers.

## r/LocalLLaMA

**Suggested title**

> [Project] CrowdTensor: volunteer LoRA training that survives intermittent GPUs (7B proof + live beta)

**Body**

I have been building CrowdTensor around a training-first question: can ordinary
machines move one shared model checkpoint forward without every contributor
remaining online for the whole run?

The unit of work is a Campaign. It pins the model, dataset, training method,
evaluation, and governance. An admitted Cell claims one bounded work unit,
runs a local LoRA update, submits a delta, and can leave. The Coordinator
validates the update, aggregates a quorum, commits checkpoint lineage, and
waits when no eligible compute is present.

The strongest completed systems run used pinned Qwen2.5-7B-Instruct and GSM8K.
Two T4x2 Kernels trained steps 1-128, both were deleted, and two fresh T4x2
Kernels restored four central stage checkpoints and completed steps 129-256
exactly once. Normalized exact match changed from 92/128 (71.875%) to 95/128
(74.219%). The practical +2-point gate passed, but the paired bootstrap
interval included zero, so I am not claiming statistical significance or
broad reasoning improvement.

The public Founding Campaign is now live on SmolLM2-135M/WikiText-2. Its first
round was seeded by two maintainer-operated private Kaggle GPU Cells through
the same public HTTPS invite/Cell path. That is useful live-route evidence, but
it is still Kaggle logical multi-node, not proof of independently administered
physical contributors.

I am opening two things for review:

1. controlled Founding Beta enrollment for people who want to test one bounded
   contribution; and
2. a Draft Qwen2.5-7B GSM8K Campaign RFC covering the stop rule, evaluation,
   hardware boundary, governance, and launch blockers.

Current boundaries are explicit: one controlled Coordinator, private invites,
no permissionless admission, no Sybil or semantic-poisoning resistance, no
secure aggregation, no production SLA, and no physical multi-host claim yet.

Website and live progress: <https://crowdtensor.24.199.118.54.nip.io>

Repository: <https://github.com/Ffffffffchopin/CrowdTensor>

7B RFC: <https://github.com/Ffffffffchopin/CrowdTensor/blob/main/docs/campaigns/qwen25-7b-gsm8k-rfc.md>

Beta access request: <https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=beta_enrollment.yml>

The feedback I need most is whether the 7B pilot's 256-step evaluation stop,
minimum useful work-unit size, and controlled trust model are technically
credible enough for the first independently administered run.

## Hugging Face Forum Or Discord

**Suggested title**

> CrowdTensor volunteer-training Beta and Draft Qwen2.5-7B GSM8K Campaign RFC

**Body**

CrowdTensor is an Apache-2.0 open-source protocol for checkpointed volunteer
model-training Campaigns. A Campaign pins immutable model/data revisions and
gives admitted CPU/GPU/TPU Cells bounded work. Accepted LoRA deltas advance one
auditable checkpoint; contributor disappearance pauses or reassigns work
instead of discarding committed progress.

The completed feasibility run used:

- `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
- `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866`
- 256 real LoRA/SFT optimizer steps and 262,144 non-padding tokens
- complete T4x2 worker replacement at step 128
- normalized exact match `71.875% -> 74.219%`

The practical preregistered threshold passed; the paired bootstrap interval
included zero, so no statistical-significance or broad-capability claim is
made.

A smaller Founding SmolLM2/WikiText-2 Campaign now serves live aggregate
status. Two maintainer-operated private Kaggle GPU Cells seeded its first round
through the public HTTPS contribution path. Enrollment remains controlled, and
the project still lacks independent physical multi-host evidence and
permissionless adversarial safety.

I would value review of the Draft 7B RFC, especially the fresh-holdout design,
the 256-to-1,024-step extension rule, quantized 16 GB GPU work units, delta
validation, and maintainer/rollback policy.

- Live site: <https://crowdtensor.24.199.118.54.nip.io>
- Draft RFC: <https://github.com/Ffffffffchopin/CrowdTensor/blob/main/docs/campaigns/qwen25-7b-gsm8k-rfc.md>
- Evidence: <https://github.com/Ffffffffchopin/CrowdTensor/blob/main/docs/qwen7b-gsm8k-elastic-showcase.md>
- Repository: <https://github.com/Ffffffffchopin/CrowdTensor>
- Controlled Beta enrollment: <https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=beta_enrollment.yml>

## Chinese Long-Form Post

**标题建议**

> CrowdTensor：让普通人的零散算力接力推进同一个开源模型训练

**正文**

我们正在开发 CrowdTensor，一个以“民主化持续训练”为核心目标的开源项目。
它想解决的问题不是如何让一台低配机器独自训练大模型，而是如何让许多不稳定、
随时可能上线或离开的普通机器，各自完成一个有边界的小更新，并持续推进同一个
可以审计和恢复的模型 checkpoint。

在 CrowdTensor 中，训练项目被称为 Campaign。每个 Campaign 会预先固定模型、
数据集 revision、训练方法、评测指标和治理规则。通过审核的贡献设备 Cell 领取
一个小型 work unit，进行本地 LoRA 更新并提交 delta。Coordinator 验证更新、
满足 quorum 后聚合并提交新 checkpoint。没有设备时训练暂停；之后出现新的设备，
就从已提交 checkpoint 继续，而不是从头开始。

目前最强的完整验证使用固定版本的 Qwen2.5-7B-Instruct 和 GSM8K，完成了 256 个
真实 LoRA/SFT optimizer steps。前两个 Kaggle T4x2 Runtime 完成 1-128 步后被
全部删除，系统经历零 Miner 状态，再由两个全新的 T4x2 Runtime 恢复四个中央
stage checkpoint，恰好完成 129-256 步。GSM8K normalized exact match 从
71.875% 提升到 74.219%。这个结果通过了预先设定的实用 +2 个百分点门槛，但
bootstrap 区间包含 0，因此我们不宣称统计显著性，也不把它描述成通用推理能力提升。

当前服务器已经运行一个公开的 Founding Campaign，模型是 SmolLM2-135M，数据是
固定版本的 WikiText-2。首轮由维护者自己的两个私有 Kaggle GPU Cell 通过相同的
公开 HTTPS 邀请和贡献路径启动。它证明真实公网入口可以工作，但仍属于 Kaggle
logical multi-node，不是不同用户物理机器参与的证据。

现在开放的是受控 Beta，而不是无需审核的匿名算力网络。当前没有 Sybil 防护、
语义投毒安全、secure aggregation 或生产 SLA。贡献邀请不会公开，需要先在
GitHub 提交不包含凭证、IP、主机名和账号信息的申请。

我们同时发布了一份简短的 Qwen2.5-7B GSM8K Campaign RFC，希望社区审阅：

- 新的独立 holdout 应该如何设计；
- 首阶段 256 步后是否应该暂停评测；
- 16 GB GPU 上最小且有意义的量化训练 work unit；
- LoRA delta 的异常与投毒检测；
- 谁拥有暂停、回滚和最终发布权限；
- 在什么证据下才能从受控 Beta 转为更开放的 Campaign。

项目网站和实时进度：<https://crowdtensor.24.199.118.54.nip.io>

GitHub：<https://github.com/Ffffffffchopin/CrowdTensor>

7B RFC：<https://github.com/Ffffffffchopin/CrowdTensor/blob/main/docs/campaigns/qwen25-7b-gsm8k-rfc.md>

申请 Founding Beta：<https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=beta_enrollment.yml>

我们现阶段最需要的不是单纯点 Star，而是愿意检查训练协议、评测设计、安全边界，
或者用一台经过授权的机器完成一个 bounded work unit 的早期参与者。

## Chinese Short Post For V2EX Or Developer Forums

**标题建议**

> 做了一个让多台临时 GPU 接力训练同一 checkpoint 的开源项目，招募受控 Beta 测试

**正文**

CrowdTensor 是一个开源的志愿模型训练协议。每台机器只领取一个有边界的 LoRA
work unit，完成后提交 delta；机器离开时，Coordinator 保留已经验证的 checkpoint，
之后由其他设备继续。

我们已经完成一次 Qwen2.5-7B/GSM8K 256 步实跑：旧的两组 T4x2 在第 128 步
全部删除，新的两组 Runtime 从中央 checkpoint 恢复并完成剩余训练。当前网站上
也运行着一个 SmolLM2/WikiText-2 Founding Campaign。

现在是受控工程 Beta，不是 permissionless 网络，也没有投毒/Sybil 安全或生产
SLA。想招募愿意测试一个 work unit、审阅 7B RFC 或检查安全边界的开发者。

- 网站：<https://crowdtensor.24.199.118.54.nip.io>
- GitHub：<https://github.com/Ffffffffchopin/CrowdTensor>
- 7B RFC：<https://github.com/Ffffffffchopin/CrowdTensor/blob/main/docs/campaigns/qwen25-7b-gsm8k-rfc.md>
- Beta 申请：<https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=beta_enrollment.yml>

## Response Boundaries

- **Is it decentralized?** No. The Beta uses one controlled Coordinator and
  admitted Cells.
- **Did unrelated users train the live Campaign?** Not yet. The first seed
  Cells were maintainer-operated Kaggle Kernels.
- **Can any 7B model join now?** No. The 7B ordinary-user runtime is an RFC
  launch blocker; the retained showcase used a controlled private topology.
- **Does the benchmark prove broad improvement?** No. It is one bounded GSM8K
  result, and its paired bootstrap interval includes zero.
- **Can someone contribute without review?** No. Public issues collect a broad
  hardware summary; approved invites are transmitted privately.
