import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8")) as {
  files?: string[];
  name?: string;
  openclaw?: {
    build?: { openclawVersion?: string };
    compat?: { pluginApi?: string };
    extensions?: string[];
  };
};
const manifest = JSON.parse(readFileSync(resolve(root, "openclaw.plugin.json"), "utf8")) as {
  contracts?: { realtimeTranscriptionProviders?: string[] };
  id?: string;
};

describe("package contract", () => {
  it("ships the built entrypoint declared to OpenClaw", () => {
    expect(packageJson.name).toBe("openclaw-funasr");
    expect(packageJson.openclaw?.extensions).toEqual(["./dist/index.js"]);
    expect(packageJson.files).toContain("dist");
    expect(existsSync(resolve(root, "dist/index.js"))).toBe(true);
  });

  it("declares the provider ownership and host API floor", () => {
    expect(manifest.id).toBe("funasr");
    expect(manifest.contracts?.realtimeTranscriptionProviders).toEqual(["funasr"]);
    expect(packageJson.openclaw?.compat?.pluginApi).toBe(">=2026.7.2");
    expect(packageJson.openclaw?.build?.openclawVersion).toBe("2026.7.2");
  });
});
