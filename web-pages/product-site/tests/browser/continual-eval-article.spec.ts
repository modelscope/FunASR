import { expect, test } from '@playwright/test';

for (const prefix of ['', 'en/']) {
  for (const width of [320, 390, 1440]) {
    test(`continual evaluation story ${prefix || 'zh'} at ${width}px`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(String(error)));
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(`/${prefix}blog/explanations/`);
      await page.locator(`a[data-blog-story][href="/${prefix}blog/sensevoice-finetuning-acceptance.html"]`).click();
      const article = page.locator('article');
      await expect(article.locator('h1')).toBeVisible();
      await expect(article.locator('tbody tr')).toHaveCount(5);
      const tableFits = await article.locator('table').evaluate((table) => {
        const bounds = table.getBoundingClientRect();
        const container = table.parentElement!.getBoundingClientRect();
        return bounds.left >= container.left - 1 && bounds.right <= container.right + 1
          && table.scrollWidth <= table.clientWidth + 1;
      });
      expect(tableFits, 'Every comparison column must fit without horizontal scrolling').toBeTruthy();
      await expect(article.locator('[data-editorial="boundary"]')).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath('opening.png') });
      await article.locator('figure').scrollIntoViewIfNeeded();
      const image = await article.locator('figure img').evaluate((img: HTMLImageElement) => ({
        width: img.naturalWidth, loaded: img.complete,
      }));
      expect(image.loaded).toBeTruthy();
      expect(image.width).toBeGreaterThan(1000);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      await page.screenshot({ path: testInfo.outputPath('example.png') });
      await expect(article.locator('[data-editorial="next-step"] a')).toHaveAttribute('href', /\/blob\/v1\.4\.15\//);
      const peer = prefix ? '' : 'en/';
      await page.locator(`.header-actions a[href="/${peer}blog/sensevoice-finetuning-acceptance.html"]`).click();
      await expect(page).toHaveURL(new RegExp(`/${peer}blog/sensevoice-finetuning-acceptance\\.html$`));
      expect(errors).toEqual([]);
    });
  }
}
