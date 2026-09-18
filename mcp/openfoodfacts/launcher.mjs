// NutriLife 使用的 Open Food Facts MCP 启动器。
//
// 说明：@jagjeevan/openfoodfacts-mcp@1.1.0 当前在 registerTools 中
// 重复注册了价格工具，直接启动会抛 “Tool getProductPrices is already
// registered”。这里在动态加载其 startServer 之前，给 McpServer 的
// registerTool 加一层去重保护，从而保留该 npm 包本身，不修改 node_modules。

const { McpServer } = await import("@modelcontextprotocol/sdk/server/mcp.js");

const registeredToolNames = new Set();
const originalRegisterTool = McpServer.prototype.registerTool;

McpServer.prototype.registerTool = function registerTool(name, ...args) {
  if (registeredToolNames.has(name)) {
    console.warn(`[openfoodfacts-mcp] 跳过重复工具注册: ${name}`);
    return;
  }
  registeredToolNames.add(name);
  return originalRegisterTool.call(this, name, ...args);
};

const { startServer } = await import(
  "@jagjeevan/openfoodfacts-mcp/dist/server.js"
);

startServer().catch((error) => {
  console.error("Open Food Facts MCP 服务启动失败:", error);
  process.exit(1);
});
