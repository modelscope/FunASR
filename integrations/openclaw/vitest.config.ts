import { existsSync, readFileSync, realpathSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { defineConfig } from "vitest/config";

const configuredRoot = process.env.OPENCLAW_ROOT?.trim();
if (!configuredRoot) {
  throw new Error("OPENCLAW_ROOT must point to an OpenClaw source checkout");
}

const openClawRoot = realpathSync(configuredRoot);
const packageJsonPath = resolve(openClawRoot, "package.json");
const realtimeSdkPath = resolve(openClawRoot, "src/plugin-sdk/realtime-transcription.ts");
if (!existsSync(realtimeSdkPath)) {
  throw new Error(`OPENCLAW_ROOT is missing ${realtimeSdkPath}`);
}
const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as { name?: string };
if (packageJson.name !== "openclaw") {
  throw new Error("OPENCLAW_ROOT does not contain the OpenClaw package");
}

const sharedConfigUrl = pathToFileURL(
  resolve(openClawRoot, "test/vitest/vitest.shared.config.ts"),
).href;
const { sharedVitestConfig } = (await import(sharedConfigUrl)) as {
  sharedVitestConfig: { resolve?: { alias?: unknown[] } };
};
const openClawAliases = Array.isArray(sharedVitestConfig.resolve?.alias)
  ? sharedVitestConfig.resolve.alias
  : [];

export default defineConfig({
  resolve: {
    alias: openClawAliases,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    testTimeout: 10_000,
  },
});
