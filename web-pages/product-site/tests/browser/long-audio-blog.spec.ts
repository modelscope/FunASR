import { expect, test } from '@playwright/test';

for (const prefix of ['', 'en/']) {
  for (const width of [390, 1440]) {
    test(`long-audio guide ${prefix || 'zh'} at ${width}px`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 });
      const route = `/${prefix}blog/funasr-transcribe-long-audio.html`;
      await page.goto(`/${prefix}blog/`);
      await page.locator(`a.post-card[href="${route}"]`).click();
      await expect(page).toHaveURL(new RegExp(route.replaceAll('.', '\\.')));
      for (const name of ['window', 'prerequisites', 'recipe', 'resources', 'batching', 'acceptance', 'evidence']) {
        await expect(page.locator(`[data-long-audio-contract="${name}"]`)).toBeVisible();
      }
      const layout = await page.evaluate(() => {
        const article = document.querySelector('article')!;
        const box = (selector: string) => article.querySelector(selector)!.getBoundingClientRect();
        return {
          overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          headingTop: box('h1').top,
          navBottom: document.querySelector('nav')!.getBoundingClientRect().bottom,
          warningBottom: box('[data-editorial="boundary"]').bottom,
          recipeTop: box('[data-long-audio-contract="recipe"]').top,
        };
      });
      expect(layout.overflow).toBeLessThanOrEqual(1);
      expect(layout.headingTop).toBeGreaterThanOrEqual(layout.navBottom);
      expect(layout.warningBottom).toBeLessThanOrEqual(layout.recipeTop);
      await page.screenshot({ path: testInfo.outputPath('article-top.png') });
      const batching = page.locator('[data-long-audio-contract="batching"]');
      await batching.scrollIntoViewIfNeeded();
      await expect(batching.locator('li')).toHaveCount(2);
      for (const value of ['max_single_segment_time=30000', 'batch_size_s=300', 'CPU']) {
        await expect(batching).toContainText(value);
      }
      const boxes = await batching.locator('li').evaluateAll(nodes => nodes.map(node => {
        const rect = node.getBoundingClientRect();
        return { left: rect.left, right: rect.right, width: node.clientWidth, content: node.scrollWidth };
      }));
      for (const box of boxes) {
        expect(box.left).toBeGreaterThanOrEqual(0);
        expect(box.right).toBeLessThanOrEqual(width);
        expect(box.content).toBeLessThanOrEqual(box.width + 1);
      }
      await page.screenshot({ path: testInfo.outputPath('batching-list.png') });
      await page.locator(`article a[href="/${prefix}docs/python-api.html"]`).click();
      await expect(page.locator('[data-source-link]')).toHaveAttribute(
        'href', new RegExp(`/docs/python_api${prefix ? '' : '_zh'}\\.md$`),
      );
    });
  }
}
