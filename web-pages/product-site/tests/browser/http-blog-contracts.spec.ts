import { expect, test } from '@playwright/test';

const slugs = [
  'self-hosted-openai-whisper-api-alternative.html',
  'self-hosted-deepgram-assemblyai-alternative.html',
];

for (const prefix of ['', 'en/']) {
  for (const slug of slugs) {
    for (const width of [390, 1440]) {
      test(`HTTP blog contract ${prefix}${slug} at ${width}px`, async ({ page }, testInfo) => {
        await page.setViewportSize({ width, height: 900 });
        await page.goto(`/${prefix}blog/`);
        if (slug === 'self-hosted-deepgram-assemblyai-alternative.html') {
          await page.locator(`[data-blog-more] a[href="/${prefix}blog/archive/"]`).click();
        }
        const route = `/${prefix}blog/${slug}`;
        await page.locator(`a.post-card[href="${route}"]`).click();
        await expect(page).toHaveURL(new RegExp(slug.replaceAll('.', '\\.')));
        const warning = page.locator('[data-http-contract="security"]');
        await expect(warning).toBeVisible();
        const layout = await page.evaluate(() => {
          const article = document.querySelector('article')!;
          const heading = article.querySelector('h1')!.getBoundingClientRect();
          const nav = document.querySelector('nav')!.getBoundingClientRect();
          const warning = article.querySelector('[data-http-contract="security"]')!.getBoundingClientRect();
          const firstCommand = article.querySelector('pre')!.getBoundingClientRect();
          return {
            overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
            headingTop: heading.top,
            navBottom: nav.bottom,
            warningBottom: warning.bottom,
            commandTop: firstCommand.top,
          };
        });
        expect(layout.overflow).toBeLessThanOrEqual(1);
        expect(layout.headingTop).toBeGreaterThanOrEqual(layout.navBottom);
        expect(layout.warningBottom).toBeLessThanOrEqual(layout.commandTop);
        await page.screenshot({ path: testInfo.outputPath('article-top.png') });
        const formats = page.locator('[data-http-contract="formats"]');
        await formats.scrollIntoViewIfNeeded();
        for (const format of ['json', 'verbose_json', 'text', 'srt', 'vtt']) {
          await expect(formats.locator(`[data-response-format="${format}"]`)).toBeVisible();
        }
        const scroll = await formats.evaluate((table) => {
          const wrapper = table.closest('.table-wrap')!;
          const box = wrapper.getBoundingClientRect();
          wrapper.scrollLeft = wrapper.scrollWidth;
          return {
            left: box.left,
            right: box.right,
            width: wrapper.clientWidth,
            content: wrapper.scrollWidth,
            scrolled: wrapper.scrollLeft,
          };
        });
        expect(scroll.left).toBeGreaterThanOrEqual(0);
        expect(scroll.right).toBeLessThanOrEqual(width);
        if (width === 390) {
          expect(scroll.content).toBeGreaterThan(scroll.width);
          expect(scroll.scrolled).toBeGreaterThan(0);
          expect(scroll.scrolled + scroll.width).toBeGreaterThanOrEqual(scroll.content - 1);
        }
        await page.screenshot({ path: testInfo.outputPath('format-boundary.png') });
        await warning.locator(`a[href="/${prefix}docs/security.html"]`).click();
        await expect(page.locator('[data-source-link]')).toHaveAttribute(
          'href', new RegExp(`/examples/openai_api/SECURITY${prefix ? '' : '_zh'}\\.md$`),
        );
      });
    }
  }
}
