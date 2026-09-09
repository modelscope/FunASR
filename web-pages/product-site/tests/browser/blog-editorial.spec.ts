import { expect, test } from '@playwright/test';

for (const prefix of ['', 'en/']) {
  for (const width of [390, 1440]) {
    test(`selected article text contrast ${prefix || 'zh'} at ${width}px`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 });
      for (const slug of ['funclip-v2-2-0-moss-speaker-clipping', 'meeting-transcript-acceptance',
        'fun-asr-nano-transformers', 'self-hosted-openai-whisper-api-alternative', 'funasr-transcribe-long-audio']) {
        await page.goto(`/${prefix}blog/${slug}.html`);
        const ratios = await page.locator('article p, article li').evaluateAll(nodes => {
          const luminance = (color: string) => {
            const rgb = color.match(/[\d.]+/g)!.slice(0, 3).map(Number).map(value => {
              const c = value / 255;
              return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
            });
            return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
          };
          return nodes.filter(node => node.textContent?.trim()).map(node => {
            let parent: Element | null = node;
            let background = 'rgb(255, 255, 255)';
            while (parent) {
              const value = getComputedStyle(parent).backgroundColor;
              if (value !== 'transparent' && value !== 'rgba(0, 0, 0, 0)') { background = value; break; }
              parent = parent.parentElement;
            }
            const foreground = luminance(getComputedStyle(node).color);
            const back = luminance(background);
            return (Math.max(foreground, back) + 0.05) / (Math.min(foreground, back) + 0.05);
          });
        });
        expect(ratios.length).toBeGreaterThan(8);
        expect(Math.min(...ratios), slug).toBeGreaterThanOrEqual(4.5);
        const linkCues = await page.locator('article p a[href], article li a[href]').evaluateAll(nodes =>
          nodes.filter(node => node.textContent?.trim()).map(node => getComputedStyle(node).textDecorationLine));
        expect(linkCues.length).toBeGreaterThan(0);
        expect(linkCues.every(value => value.includes('underline')), `${slug}: visible inline links`).toBeTruthy();
        if (slug === 'funclip-v2-2-0-moss-speaker-clipping') {
          await page.locator('[data-editorial="example"]').scrollIntoViewIfNeeded();
          await page.screenshot({ path: testInfo.outputPath('readable-funclip.png') });
        }
      }
    });

    test(`edited blog ${prefix || 'zh'} at ${width}px`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`/${prefix}blog/`);
      const home = page.locator('[data-blog-view="home"]');
      await expect(home.locator('[data-blog-story]')).toHaveCount(5);
      await expect(home.locator('[data-blog-lead] img')).toBeVisible();
      const layout = await page.evaluate(() => {
        const h1 = document.querySelector('h1')!.getBoundingClientRect();
        const image = document.querySelector('[data-blog-lead] img') as HTMLImageElement;
        const selected = document.querySelector('[data-blog-selected]')!.getBoundingClientRect();
        return { headingTop: h1.top, headingBottom: h1.bottom,
          navBottom: document.querySelector('.site-header')!.getBoundingClientRect().bottom,
          overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          imageWidth: image.naturalWidth, selectedTop: selected.top };
      });
      expect(layout.headingTop).toBeGreaterThanOrEqual(layout.navBottom);
      expect(layout.headingBottom).toBeLessThan(300);
      expect(layout.overflow).toBeLessThanOrEqual(1);
      expect(layout.imageWidth).toBeGreaterThan(100);
      expect(layout.selectedTop).toBeLessThan(900);
      await page.screenshot({ path: testInfo.outputPath('homepage.png'), fullPage: true });
      const lead = home.locator('[data-blog-lead] [data-blog-story]');
      const href = await lead.getAttribute('href');
      await lead.click();
      await expect(page).toHaveURL(new RegExp(href!.replaceAll('.', '\\.')));
      await expect(page.locator('article h1')).toBeVisible();
      for (const category of ['applications', 'selection', 'explanations']) {
        await page.goto(`/${prefix}blog/`);
        await page.locator(`[data-blog-navigation] a[href="/${prefix}blog/${category}/"]`).click();
        const view = page.locator(`[data-blog-view="${category}"]`);
        await expect(view.locator('h1')).toBeVisible();
        expect(await view.locator('[data-blog-story]').count()).toBeGreaterThan(0);
        expect(await view.locator(`[data-blog-story]:not([data-blog-category="${category}"])`).count()).toBe(0);
        expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      }
      await page.locator(`[data-blog-more] a[href="/${prefix}blog/archive/"]`).click();
      const archive = page.locator('[data-blog-view="archive"]');
      await expect(archive.locator('[data-blog-story]')).toHaveCount(35);
      await page.screenshot({ path: testInfo.outputPath('archive.png') });
      await archive.locator(`a[href="/${prefix}blog/self-hosted-deepgram-assemblyai-alternative.html"]`).click();
      await expect(page.locator('article h1')).toBeVisible();
      await page.goto(`/${prefix}blog/`);
      await page.locator(`[data-blog-more] a[href="/${prefix}blog/releases/"]`).click();
      const releases = page.locator('[data-blog-view="releases"]');
      expect(await releases.locator('[data-blog-story]').count()).toBeGreaterThan(0);
      await expect(releases.locator('[data-blog-story]:not([data-blog-category="releases"])')).toHaveCount(0);
    });
  }
}
