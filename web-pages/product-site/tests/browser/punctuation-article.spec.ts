import { expect, test } from '@playwright/test';

for (const prefix of ['', 'en/']) {
  for (const width of [320, 390, 1440]) {
    test(`Punctuation candidate ${prefix || 'zh'} at ${width}px`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(String(error)));
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(`/${prefix}blog/explanations/`);
      await page.locator(`a[data-blog-story][href="/${prefix}blog/punctuation-restoration-python.html"]`).click();
      const article = page.locator('article');
      await expect(article.locator('h1')).toBeVisible();
      await expect(article.locator('[data-editorial="opening"]')).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath('opening.png') });
      for (const name of ['punctuate', 'observed']) {
        const example = article.locator(`pre[data-example="${name}"]`);
        await example.scrollIntoViewIfNeeded();
        expect(await example.evaluate(node => node.scrollWidth - node.clientWidth)).toBeLessThanOrEqual(1);
        expect(await example.evaluate(node => {
          const text = [...node.childNodes].find(child => child.nodeType === Node.TEXT_NODE && child.textContent?.trim());
          const button = node.querySelector('button');
          if (!text || !button) return false;
          const range = document.createRange();
          const start = text.textContent!.search(/\S/);
          range.setStart(text, start);
          range.setEnd(text, start + 1);
          return range.getBoundingClientRect().top >= button.getBoundingClientRect().bottom + 2;
        }), `${name} text must not overlap its copy button`).toBeTruthy();
      }
      await expect(article.locator('[data-example="observed"]')).toContainText('人不是石头人有主观价值。');
      await article.locator('[data-editorial="manual-edit"]').scrollIntoViewIfNeeded();
      await expect(article.locator('[data-editorial="manual-edit"]')).toContainText('人不是石头，人有主观价值。');
      await page.screenshot({ path: testInfo.outputPath('review.png') });
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      const peer = prefix ? '' : 'en/';
      await page.locator(`.header-actions a[href="/${peer}blog/punctuation-restoration-python.html"]`).click();
      await expect(page).toHaveURL(new RegExp(`/${peer}blog/punctuation-restoration-python\\.html$`));
      expect(errors).toEqual([]);
    });
  }
}
