/**
 * Bridge checks plus real Python round trips using only synthetic example logs.
 * Run from .opencode with `npm test`; install the Python package in .venv first.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  BridgeError,
  filteredEnv,
  resolvePython,
  runVvaContract,
} from "../lib/vva_bridge.ts";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const workspace = join(root, "examples");
const python = resolvePython(root);

test("environment filter drops unrelated secrets and preserves allowed variables", () => {
  const result = filteredEnv({
    PATH: "/usr/bin",
    HOME: "/home/test",
    DEEPSEEK_API_KEY: "synthetic-secret",
    OTHER_SECRET: "synthetic-secret",
    VVA_ALLOW_NETWORK: "1",
    VVA_CAM_DEMO_SUB_URL: "synthetic-url",
    VVA_CAM_DEMO_ONVIF_PASSWORD: "synthetic-secret",
    VVA_PYTHON: "/operator/python",
  });
  assert.deepEqual(result, {
    PATH: "/usr/bin",
    HOME: "/home/test",
    VVA_ALLOW_NETWORK: "1",
    VVA_CAM_DEMO_SUB_URL: "synthetic-url",
  });
});

test("operator can select an absolute Python interpreter", () => {
  assert.equal(resolvePython(root, { VVA_PYTHON: "/operator/python" }), "/operator/python");
});

test("relative Python overrides are rejected", () => {
  assert.throws(
    () => resolvePython(root, { VVA_PYTHON: "./python" }),
    /VVA_PYTHON must be an absolute path/,
  );
});

test("missing virtual environment falls back to python3", () => {
  const empty = mkdtempSync(join(tmpdir(), "vva-bridge-test-"));
  try {
    assert.equal(resolvePython(empty, {}), "python3");
  } finally {
    rmSync(empty, { recursive: true, force: true });
  }
});

test("bridge returns a validated report from real Python and synthetic logs", async () => {
  const report = await runVvaContract({
    python,
    workspace,
    request: {
      task: "PIPELINE_BENCH",
      timings_log: "logs/timings.jsonl",
      system_log: "logs/system.jsonl",
      latency_budget_ms: 400.0,
    },
  });
  assert.equal(report.contract_id, "PipelineBenchReport");
  assert.equal(report.latency_budget_ms, 400.0);
  assert.equal(report.verdict, "PASS");
  assert.deepEqual(report.flags, []);
  const overall = report.overall as Record<string, unknown>;
  assert.equal(overall.n_records, 1700);
  assert.equal(overall.n_processed, 1694);
  assert.equal(overall.n_dropped, 6);
});

test("Python contract errors are rejected, not returned as successful reports", async () => {
  await assert.rejects(
    runVvaContract({
      python,
      workspace,
      request: {
        task: "PIPELINE_BENCH",
        timings_log: "logs/timings.jsonl",
        latency_budget_ms: -1.0,
      },
    }),
    (error: unknown) => {
      assert.ok(error instanceof BridgeError);
      assert.equal(error.exitCode, 2);
      const payload = error.payload as { error: { code: string } };
      assert.equal(payload.error.code, "contract_violation");
      return true;
    },
  );
});

test("missing interpreter is rejected as a launch error", async () => {
  await assert.rejects(
    runVvaContract({
      python: join(root, "__nonexistent_python__"),
      workspace,
      request: { task: "PIPELINE_BENCH", timings_log: "logs/timings.jsonl" },
    }),
    /cannot start Python/,
  );
});

test("child dying before stdin is fully written rejects as BridgeError, not an uncaught stdin error", async () => {
  // process.execPath starts and exits immediately on the unknown '-m' option,
  // without ever reading stdin. A request larger than the OS pipe buffer
  // guarantees Node is still writing when the child dies, so child.stdin emits
  // EPIPE. Without a stdin error handler that unhandled 'error' event throws
  // and tears down the test process instead of rejecting with BridgeError.
  const request = {
    task: "PIPELINE_BENCH",
    timings_log: "logs/timings.jsonl",
    filler: "x".repeat(1 << 20),
  };
  await assert.rejects(runVvaContract({ python: process.execPath, workspace, request }), BridgeError);
});
