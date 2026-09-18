FROM node:20-slim

ENV NODE_ENV=production \
    TRANSPORT=http \
    PORT=28375

WORKDIR /app

COPY mcp/openfoodfacts/package.json ./
COPY mcp/openfoodfacts/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY mcp/openfoodfacts/launcher.mjs ./
COPY mcp/openfoodfacts/patch-openfoodfacts.mjs ./

EXPOSE 28375

CMD ["npm", "start"]
