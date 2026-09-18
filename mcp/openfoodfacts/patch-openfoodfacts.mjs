// 修复 @jagjeevan/openfoodfacts-mcp@1.1.0 的两个生产问题：
//
// 1. V3 barcode 解析问题：该包把 SDK 返回的 { product: {...} } 外层对象误当成
//    product 本身，导致 getProductByBarcode 返回 "Unknown product" 且缺少
//    nutritionFacts。
//
// 2. searchProducts 直连 world.openfoodfacts.org，既没有 User-Agent，也没有
//    请求超时与重试退避。上游偶发 503/429（限流或后端抖动）时会直接失败。
//    这里补上 User-Agent、请求超时与指数退避重试。
//
// 以上均通过启动前对 node_modules 编译产物做最小字符串补丁实现，不修改 npm 包本身。

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const target = new URL(
  "./node_modules/@jagjeevan/openfoodfacts-mcp/dist/tools/product-search.js",
  import.meta.url,
);

let source = await readFile(target, "utf8");
let changed = false;

// ── 1. barcode 解析补丁 ──────────────────────────────────────────
const barcodeOld = "const product = result.data;";
const barcodeNew = "const product = result.data.product ?? result.data;";

if (source.includes(barcodeNew)) {
  console.log(`[openfoodfacts-mcp] 已包含补丁片段，跳过: ${barcodeOld}`);
} else if (source.includes(barcodeOld)) {
  source = source.replace(barcodeOld, barcodeNew);
  changed = true;
  console.log(`[openfoodfacts-mcp] 已打补丁: ${barcodeOld}`);
} else {
  console.warn(`[openfoodfacts-mcp] 未找到预期补丁位置: ${barcodeOld}`);
}

// ── 2. searchProducts 超时 + 重试退避补丁 ────────────────────────
const retryMarker = "NutriLife search retry";
if (source.includes(retryMarker)) {
  console.log("[openfoodfacts-mcp] 已包含搜索重试补丁，跳过");
} else {
  const originalFetch =
    "        const response = await fetch(searchUrl.toString());";
  const headerFetch =
    '        const response = await fetch(searchUrl.toString(), { headers: { "User-Agent": process.env.OFF_USER_AGENT || "NutriLife/0.1 (Open Food Facts MCP)", "Accept": "application/json" } });';
  const retryFetch = [
    '        const headers = { "User-Agent": process.env.OFF_USER_AGENT || "NutriLife/0.1 (Open Food Facts MCP)", "Accept": "application/json" };',
    "        // NutriLife search retry: 超时 + 指数退避，抵抗 OFF 上游偶发 503/429",
    "        let response = { ok: false, status: 0, statusText: 'search request failed' };",
    "        for (let attempt = 0; attempt <= 3; attempt++) {",
    "            const controller = new AbortController();",
    "            const timeoutId = setTimeout(() => controller.abort(), 15000);",
    "            try {",
    "                response = await fetch(searchUrl.toString(), { headers, signal: controller.signal });",
    "                if (response.ok) break;",
    "            } catch (_error) {",
    "                // 网络错误/超时：保留兜底 response，继续退避重试",
    "            } finally {",
    "                clearTimeout(timeoutId);",
    "            }",
    "            if (attempt < 3) {",
    "                await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt));",
    "            }",
    "        }",
  ].join("\n");

  if (source.includes(headerFetch)) {
    source = source.replace(headerFetch, retryFetch);
    changed = true;
    console.log(
      "[openfoodfacts-mcp] 已打补丁: searchProducts fetch（含 User-Agent 版本）",
    );
  } else if (source.includes(originalFetch)) {
    source = source.replace(originalFetch, retryFetch);
    changed = true;
    console.log(
      "[openfoodfacts-mcp] 已打补丁: searchProducts fetch（原始版本）",
    );
  } else {
    console.warn(
      "[openfoodfacts-mcp] 未找到 searchProducts fetch 段落，无法打重试补丁",
    );
  }
}

if (changed) {
  await writeFile(target, source, "utf8");
  console.log("[openfoodfacts-mcp] product-search.js 补丁写入完成");
}
