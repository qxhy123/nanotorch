# SD-Viz Deployment Guide

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:5173
```

## Production Build

```bash
# Build for production
npm run build

# Output: dist/ directory
# Total size: ~400 KB (gzipped: ~100 KB)
```

## Deployment Options

### Option 1: Static Hosting (Recommended)

Deploy to any static hosting service:

#### Vercel
```bash
npm install -g vercel
vercel deploy dist
```

#### Netlify
```bash
npm install -g netlify-cli
netlify deploy --dir=dist
```

#### GitHub Pages
```bash
# Build to dist/
npm run build

# Or use subtree branch
git subtree push --prefix dist origin gh-pages
```

#### Cloudflare Pages
```bash
# Via dashboard or wrangler
wrangler pages publish dist
```

### Option 2: Node.js Server

#### Simple Express Server

```javascript
const express = require('express');
const path = require('path');

const app = express();
app.use(express.static('dist'));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
```

#### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Option 3: Docker

#### Dockerfile

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY package*.json ./

EXPOSE 3000
CMD ["npm", "run", "preview"]
```

#### Build and Run

```bash
docker build -t sd-viz .
docker run -p 3000:3000 sd-viz
```

## Environment Variables

Create `.env` file for configuration:

```env
# API endpoint (if using backend)
VITE_API_BASE_URL=http://localhost:8000

# Analytics (optional)
VITE_GA_ID=G-XXXXXXXXXX
```

## Performance Optimization

Build output is already optimized:

- Code splitting by route
- Tree-shaking removes unused code
- Minification with Vite
- CSS purged with TailwindCSS

## Troubleshooting

### Build Errors

```bash
# Clear cache and rebuild
rm -rf node_modules dist
npm install
npm run build
```

### Development Issues

```bash
# Check Node version (requires >= 20)
node --version

# Clear npm cache
npm cache clean --force
```

### Port Already in Use

```bash
# Kill process on port 5173
npx kill-port 5173

# Or use different port
npm run dev -- --port 3000
```
