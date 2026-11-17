import { readFile } from "node:fs/promises";

const checks = {
  "../src/App.jsx": ["AutoML Forge", "/api/auth/login", "Start AutoML training", "Analytics", "Explainability", "Data quality score"],
  "../src/pages/Analytics.jsx": ["/api/analytics", "Dataset quality", "Model comparison", "Prediction history"],
  "../src/pages/Explainability.jsx": ["/explain", "Feature importance", "Confusion matrix", "ROC curve", "Residual plot", "/report"],
};
const missing = [];
for (const [file, required] of Object.entries(checks)) {
  const source = await readFile(new URL(file, import.meta.url), "utf8");
  for (const text of required) if (!source.includes(text)) missing.push(`${file}: ${text}`);
}
if (missing.length) {
  throw new Error(`Frontend smoke test failed: ${missing.join(", ")}`);
}
console.log("Frontend smoke test passed");
