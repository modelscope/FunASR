import { expect, test } from '@playwright/test';

for (const prefix of ['', 'en/']) {
  for (const width of [320, 390, 1440]) {
    test(`Subtitle output ${prefix || 'zh'} at ${width}px`, async ({ page }, testInfo) => {
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(String(error)));
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(`/${prefix}blog/applications/`);
      await page.locator(`a[data-blog-story][href="/${prefix}blog/generate-subtitles-srt-vtt-from-audio-video.html"]`).click();
      const article = page.locator('article');
      await expect(article.locator('h1')).toBeVisible();
      await expect(article.locator('[data-editorial="opening"]')).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath('opening.png') });
      await article.locator('figure').scrollIntoViewIfNeeded();
      expect(await article.locator('figure img').evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth >= 1400)).toBeTruthy();
      for (const name of ['transcribe', 'convert', 'srt-output', 'vtt-output']) {
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
      await expect(article.locator('[data-example="transcribe"]')).toContainText('--model paraformer');
      await expect(article.locator('[data-example="vtt-output"]')).toContainText('WEBVTT');
      await page.screenshot({ path: testInfo.outputPath('output.png') });
      await expect(article.locator('[data-editorial="native-boundary"] a')).toHaveAttribute('href', `/${prefix}docs/native-transformers.html`);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      const peer = prefix ? '' : 'en/';
      await page.locator(`.header-actions a[href="/${peer}blog/generate-subtitles-srt-vtt-from-audio-video.html"]`).click();
      await expect(page).toHaveURL(new RegExp(`/${peer}blog/generate-subtitles-srt-vtt-from-audio-video\\.html$`));
      expect(errors).toEqual([]);
    });
  }
}
