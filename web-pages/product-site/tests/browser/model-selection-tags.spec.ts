import { expect, test } from '@playwright/test';

for (const width of [320, 390, 1440]) {
  for (const language of ['zh', 'en']) {
    test(`Model choice preserves distinct runtime journeys: ${language} ${width}px`, async ({ page }, testInfo) => {
      const prefix = language === 'en' ? '/en' : '';
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`${prefix}/docs/model-selection.html`);
      const anchor = page.locator('#vllm-checkpoint-paths');
      await expect(anchor).toHaveCount(1);
      await page.locator('.docs-article a[href="#vllm-checkpoint-paths"]').click();
      await expect(page).toHaveURL(/#vllm-checkpoint-paths$/);
      const heading = anchor.locator('xpath=following::h2[1]');
      await expect(heading).toBeInViewport({ ratio: 1 });
      await expect.poll(() => heading.evaluate(node => {
        const box = node.getBoundingClientRect();
        const painted = document.elementFromPoint(box.left + Math.min(20, box.width / 2), box.top + box.height / 2);
        return painted?.closest('h2') === node;
      })).toBe(true);
      const table = page.locator('.docs-article table').filter({ hasText: 'allendou/Fun-ASR-Nano-2512-vllm' });
      await expect(table).toHaveCount(1);
      await expect(table).toContainText('AutoModelVLLM');
      await expect(table).toContainText('/v1/audio/transcriptions');
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.screenshot({ path: testInfo.outputPath('runtime-choice.png') });
      for (const [slug, marker] of [
        ['official-native-vllm', 'a4362c943d48951f98ca2a62181cc028970270c5'],
        ['vllm', 'AutoModelVLLM'],
        ['native-vllm', 'allendou/Fun-ASR-Nano-2512-vllm'],
      ]) {
        await table.locator(`a[href="${prefix}/docs/${slug}.html"]`).click();
        await expect(page).toHaveURL(new RegExp(`${prefix}/docs/${slug}.html$`));
        await expect(page.locator('.docs-article')).toContainText(marker);
        expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
        await page.goBack();
      }
      await page.locator(`.docs-article a[href="${prefix}/docs/moss-transcribe-diarize.html"]`).first().click();
      await expect(page).toHaveURL(new RegExp(`${prefix}/docs/moss-transcribe-diarize.html$`));
      await expect(page.locator('.docs-article')).toContainText('OpenMOSS');
      expect(errors).toEqual([]);
    });
  }
}

for (const width of [390, 1440]) {
  for (const language of ['zh', 'en']) {
    test(`Model selection to raw tag recipe: ${language} ${width}px`, async ({ page }, testInfo) => {
      const prefix = language === 'en' ? '/en' : '';
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`${prefix}/docs/model-selection.html`);
      const recipe = page.locator(`.docs-article a[href="${prefix}/docs/speaker-emotion.html"]`);
      await expect(recipe).toHaveCount(1);
      await recipe.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      await page.screenshot({ path: testInfo.outputPath('http-aliases.png') });
      await recipe.click();
      await expect(page).toHaveURL(new RegExp(`${prefix}/docs/speaker-emotion.html$`));
      const code = page.locator('.docs-article pre').filter({ hasText: 'raw_tagged_text' });
      await expect(code).toHaveCount(1);
      await code.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      expect(errors).toEqual([]);
    });
  }
}
