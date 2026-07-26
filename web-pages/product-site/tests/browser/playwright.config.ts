import path from 'node:path';
import { defineConfig } from '@playwright/test';

const siteRoot = path.resolve(__dirname, '../..');
const python = process.env.SITE_PYTHON || 'python3';

export default defineConfig({
  testDir: '.',
  testMatch: 'product-site.spec.ts',
  timeout: 45_000,
  workers: 1,
  reporter: 'line',
  outputDir: '/tmp/funasr-product-site-playwright-results',
  use: {
    baseURL: 'http://127.0.0.1:8770',
    browserName: 'chromium',
    locale: 'zh-CN',
    colorScheme: 'light',
  },
  webServer: {
    command: `${python} build.py --output /tmp/funasr-product-site-browser && ${python} -m http.server 8770 --bind 127.0.0.1 --directory /tmp/funasr-product-site-browser`,
    cwd: siteRoot,
    url: 'http://127.0.0.1:8770/',
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
