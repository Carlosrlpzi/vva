/**
 * Bridge between the OpenCode `vva_contract` tool and the Python CLI.
 *
 * Guarantees (each one is tested in .opencode/tests/vva_bridge.test.ts):
 *  - Python is spawned WITHOUT a shell: request values can never become shell syntax.
 *  - The request travels on stdin, never on the command line (not visible in `ps`).
 *  - The child receives an environment ALLOWLIST: no DEEPSEEK_API_KEY, no
 *    unrelated secrets. Only VVA_ALLOW_* gates and VVA_CAM_*_URL camera URLs pass.
 *  - Only exit code 0 with exactly one JSON object on stdout is a success.
 *    Any other outcome becomes a thrown BridgeError carrying the typed error.
 *  - stdout is capped (8 MiB) and the process has an outer timeout.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { isAbsolute, join } from "node:path";

export const MAX_STDOUT_BYTES = 8 * 1024 * 1024;
export const DEFAULT_TIMEOUT_MS = 20 * 60 * 1000; // dataset hashing can take minutes

/** Variables the Python process may see. Everything else is dropped. */
const ENV_EXACT = new Set(["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "VIRTUAL_ENV"]);
const ENV_PATTERNS = [/^VVA_ALLOW_[A-Z_]+$/, /^VVA_CAM_[A-Z0-9_]+_URL$/];

export class BridgeError extends Error {
  constructor(
    message: string,
    readonly exitCode: number | null,
    readonly payload: unknown,
  ) {
    super(message);
    this.name = "BridgeError";
  }
}

export function filteredEnv(source: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(source)) {
    if (value === undefined) continue;
    if (ENV_EXACT.has(key) || ENV_PATTERNS.some((re) => re.test(key))) env[key] = value;
  }
  return env;
}

/**
 * Interpreter resolution, in order:
 *  1. VVA_PYTHON set by the OPERATOR (absolute path; model arguments cannot set it);
 *  2. the project's virtual environment (POSIX or Windows layout);
 *  3. `python3` from PATH.
 */
export function resolvePython(worktree: string, env: NodeJS.ProcessEnv = process.env): string {
  const override = env.VVA_PYTHON;
  if (override) {
    if (!isAbsolute(override)) throw new BridgeError("VVA_PYTHON must be an absolute path", null, null);
    return override;
  }
  for (const candidate of [join(worktree, ".venv", "bin", "python"), join(worktree, ".venv", "Scripts", "python.exe")]) {
    if (existsSync(candidate)) return candidate;
  }
  return "python3";
}

export interface RunOptions {
  python: string;
  workspace: string;
  request: Record<string, unknown>;
  timeoutMs?: number;
  signal?: AbortSignal;
  env?: NodeJS.ProcessEnv;
}

export function runVvaContract(opts: RunOptions): Promise<Record<string, unknown>> {
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  return new Promise((resolve, reject) => {
    const child = spawn(opts.python, ["-m", "vva_contracts", "run", "--workspace", opts.workspace], {
      shell: false,
      cwd: opts.workspace,
      env: filteredEnv(opts.env ?? process.env),
      stdio: ["pipe", "pipe", "pipe"],
    });

    const chunks: Buffer[] = [];
    let size = 0;
    let stderrTail = "";
    let failure: BridgeError | null = null;

    const kill = (reason: BridgeError): void => {
      failure ??= reason;
      child.kill("SIGKILL");
    };
    const timer = setTimeout(() => kill(new BridgeError(`timed out after ${timeoutMs} ms`, null, null)), timeoutMs);
    const onAbort = (): void => kill(new BridgeError("aborted by the user", null, null));
    opts.signal?.addEventListener("abort", onAbort, { once: true });

    child.stdout.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_STDOUT_BYTES) return kill(new BridgeError("stdout exceeded 8 MiB", null, null));
      chunks.push(chunk);
    });
    // Keep only the tail of stderr (a Python traceback) for diagnostics.
    child.stderr.on("data", (chunk: Buffer) => {
      stderrTail = (stderrTail + chunk.toString("utf8")).slice(-2000);
    });
    child.on("error", (err) => kill(new BridgeError(`cannot start Python: ${err.message}`, null, null)));

    child.on("close", (code) => {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", onAbort);
      if (failure) return reject(failure);

      const stdout = Buffer.concat(chunks).toString("utf8").trim();
      let parsed: unknown = null;
      try {
        parsed = stdout ? JSON.parse(stdout) : null;
      } catch {
        parsed = null;
      }
      const isObject = typeof parsed === "object" && parsed !== null && !Array.isArray(parsed);

      if (code === 0 && isObject && !("error" in (parsed as object))) {
        return resolve(parsed as Record<string, unknown>);
      }
      if (isObject && "error" in (parsed as object)) {
        // Typed failure from the CLI: forward it unchanged.
        return reject(new BridgeError(`vva_contract failed (exit ${code}): ${stdout}`, code, parsed));
      }
      // Exit 1 (bug) or malformed output: never present it as a report.
      reject(new BridgeError(`vva_contract crashed (exit ${code}); stderr tail: ${stderrTail}`, code, null));
    });

    child.stdin.end(JSON.stringify(opts.request));
  });
}
