/**
 * OpenCode custom tool `vva_contract` (the file name is the tool name).
 *
 * The model chooses `task` explicitly and passes the task's fields in `params`.
 * This file does NOT validate `params` beyond its shape: the Python request
 * contract (strict Pydantic, extra fields forbidden, no type coercion) is the
 * single source of truth, so the two layers can never disagree.
 *
 * Field reference per task: the SKILL.md of each task in .opencode/skills/.
 */
import { tool } from "@opencode-ai/plugin";
import { resolvePython, runVvaContract } from "../lib/vva_bridge.ts";

const TASKS = [
  "DATASET_AUDIT",
  "DETECTION_EVAL",
  "QUANT_PARITY",
  "RULE_REPLAY",
  "STREAM_PROBE",
  "PIPELINE_BENCH",
] as const;

export default tool({
  description:
    "Run one deterministic, locally computed audit of the video-surveillance project and return its " +
    "validated JSON report. Tasks: DATASET_AUDIT (YOLO labels, leakage, box sizes), DETECTION_EVAL " +
    "(mAP/PR on a labeled split), QUANT_PARITY (FP32 vs INT8/HEF), RULE_REPLAY (alert policies on " +
    "logged frames), STREAM_PROBE (camera fps/drops via ffprobe), PIPELINE_BENCH (throughput, latency, " +
    "temperature). Load the matching vva-* skill first for the exact params. Paths are workspace-relative. " +
    "No data leaves the machine.",
  args: {
    task: tool.schema.enum(TASKS).describe("Which audit to run"),
    params: tool.schema
      .record(tool.schema.string(), tool.schema.unknown())
      .describe("Task fields exactly as documented in the task's skill (do not include 'task')"),
  },
  async execute(args, context) {
    if ("task" in args.params) {
      throw new Error("params must not contain 'task'; pass it as the separate 'task' argument");
    }
    const report = await runVvaContract({
      python: resolvePython(context.worktree),
      workspace: context.directory,
      request: { task: args.task, ...args.params },
      signal: context.abort,
    });
    // Returned verbatim: the agent must repeat it unchanged (see the auditor prompt).
    return JSON.stringify(report);
  },
});
