import { rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = resolve(root, "dist");

await rm(outputDirectory, { force: true, recursive: true });
await build({
  bundle: true,
  entryPoints: [resolve(root, "src/index.ts")],
  external: ["openclaw/*"],
  format: "esm",
  legalComments: "none",
  logLevel: "info",
  outfile: resolve(outputDirectory, "index.js"),
  platform: "node",
  sourcemap: true,
  target: "node22",
});
