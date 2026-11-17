import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
const required = ["AutoML Forge", "/api/auth/login", "Start AutoML training"];
const missing = required.filter((text) => !source.includes(text));
if (missing.length) {
  throw new Error(`Frontend smoke test failed: ${missing.join(", ")}`);
}
console.log("Frontend smoke test passed");
