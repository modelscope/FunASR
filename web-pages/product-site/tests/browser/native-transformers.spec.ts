import { expect, test } from '@playwright/test';

for (const prefix of ['', 'en/']) {
  for (const width of [390, 1440]) {
    test(`native Transformers article and guide ${prefix || 'zh'} ${width}`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 900 });
      const route = `/${prefix}blog/fun-asr-nano-transformers.html`;
      await page.goto(`/${prefix}blog/`);
      await page.locator(`a.post-card[href="${route}"]`).click();
      await expect(page).toHaveURL(new RegExp(route.replaceAll('.', '\\.')));
      const layout = await page.evaluate(() => ({
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        headingTop: document.querySelector('article h1')!.getBoundingClientRect().top,
        navBottom: document.querySelector('nav')!.getBoundingClientRect().bottom,
      }));
      expect(layout.overflow).toBeLessThanOrEqual(1);
      expect(layout.headingTop).toBeGreaterThanOrEqual(layout.navBottom);
      await page.screenshot({ path: testInfo.outputPath('article-top.png') });
      const waveform = page.locator('article img');
      await waveform.scrollIntoViewIfNeeded();
      await expect(waveform).toBeVisible();
      expect(await waveform.evaluate((node: HTMLImageElement) => node.complete && node.naturalWidth === 1800)).toBeTruthy();
      await page.screenshot({ path: testInfo.outputPath('waveform.png') });
      const formats = page.locator('[data-native-section="formats"]');
      await expect(formats.locator('li')).toHaveCount(4);
      for (const model of ['Fun-ASR-Nano-2512', 'Fun-ASR-Nano-2512-hf', 'Fun-ASR-Nano-2512-vllm', 'GGUF']) {
        await expect(formats).toContainText(model);
      }
      const table = page.locator('[data-native-section="observations"] table');
      await table.scrollIntoViewIfNeeded();
      const scroll = await table.evaluate((tableNode) => {
        const node = tableNode.closest('.table-wrap')!;
        node.scrollLeft = node.scrollWidth;
        const rect = node.getBoundingClientRect();
        return { left: rect.left, right: rect.right, scroll: node.scrollLeft,
          width: node.clientWidth, content: node.scrollWidth };
      });
      expect(scroll.left).toBeGreaterThanOrEqual(0);
      expect(scroll.right).toBeLessThanOrEqual(width);
      if (width === 390) expect(scroll.scroll).toBeGreaterThan(0);
      await page.locator(`article a[href="/${prefix}docs/native-transformers.html"]`).first().click();
      await expect(page.locator('[data-source-link]')).toHaveAttribute(
        'href', new RegExp(`/docs/transformers_native${prefix ? '' : '_zh'}\\.md$`),
      );
      await expect(page.locator('.docs-article')).toContainText('torchaudio==2.10.0+cpu');
      expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
      await page.screenshot({ path: testInfo.outputPath('guide-top.png') });
      await page.goto(`/${prefix}models.html`);
      await page.locator(`a[href="/${prefix}docs/native-transformers.html"]`).click();
      await expect(page).toHaveURL(new RegExp('/docs/native-transformers\\.html$'));
      await page.goto(`/__pages/${prefix ? '' : 'zh/'}native-transformers.html`);
      await expect(page.locator('.docs-article')).toContainText('FunAudioLLM/Fun-ASR-Nano-2512-hf');
      expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    });
  }
}
