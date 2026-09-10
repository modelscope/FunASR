import { expect, test } from '@playwright/test';

for (const prefix of ['', 'en/']) {
  for (const width of [320, 390, 1440]) {
    test(`SenseVoice tags ${prefix || 'zh'} at ${width}px`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(String(error)));
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(`/${prefix}blog/explanations/`);
      await page.locator(`a[data-blog-story][href="/${prefix}blog/sensevoice-emotion-language-detection.html"]`).click();
      const article = page.locator('article');
      await expect(article.locator('h1')).toBeVisible();
      await expect(article.locator('[data-editorial="boundary"]')).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath('opening.png') });
      await article.locator('figure').scrollIntoViewIfNeeded();
      expect(await article.locator('figure img').evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth >= 1400)).toBeTruthy();
      const raw = article.locator('[data-example="raw-output"]');
      await raw.scrollIntoViewIfNeeded();
      await expect(raw).toContainText('<|zh|>');
      expect(await raw.evaluate(node => node.scrollWidth - node.clientWidth)).toBeLessThanOrEqual(1);
      expect(await raw.evaluate(node => {
        const text = [...node.childNodes].find(child => child.nodeType === Node.TEXT_NODE && child.textContent?.trim());
        const button = node.querySelector('button');
        if (!text || !button) return false;
        const range = document.createRange();
        const start = text.textContent!.search(/\S/);
        range.setStart(text, start);
        range.setEnd(text, start + 1);
        return range.getBoundingClientRect().top >= button.getBoundingClientRect().bottom + 2;
      }), 'Copy control must not overlap the first output line').toBeTruthy();
      await expect(article.locator('[data-example="display-output"]')).not.toContainText('<|');
      await page.screenshot({ path: testInfo.outputPath('readout.png') });
      await expect(article.locator('[data-editorial="next-step"] a')).toHaveAttribute('href', /4482962437ce8ebd1f0ac5b6793d2f82d2e2955d/);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      const peer = prefix ? '' : 'en/';
      await page.locator(`.header-actions a[href="/${peer}blog/sensevoice-emotion-language-detection.html"]`).click();
      await expect(page).toHaveURL(new RegExp(`/${peer}blog/sensevoice-emotion-language-detection\\.html$`));
      expect(errors).toEqual([]);
    });
  }
}
