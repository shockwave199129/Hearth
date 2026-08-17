import { cpSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "..", "assets");
const dest = join(root, "public", "brand");

if (!existsSync(src)) {
  console.error(`Missing brand assets at ${src}`);
  process.exit(1);
}

mkdirSync(dest, { recursive: true });
cpSync(src, dest, { recursive: true });
console.log(`Synced brand assets → public/brand`);
