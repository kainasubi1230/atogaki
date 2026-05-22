FROM node:22-alpine

WORKDIR /app

# Install dependencies deterministically from the workspace lockfile.
COPY package.json package-lock.json ./
COPY web/package.json ./web/package.json
RUN npm ci --workspace web

COPY web ./web
WORKDIR /app/web
RUN rm -rf .next

EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]
