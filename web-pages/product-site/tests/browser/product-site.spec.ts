import { expect, test } from '@playwright/test';

const viewports = [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'wide', width: 1920, height: 1080 },
];

for (const viewport of viewports) {
  test(`home layout is stable at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.goto('/');
    await expect(page.locator('.hero-image')).toBeVisible();
    await expect(page.locator('.hero-image')).toHaveJSProperty('complete', true);

    const layout = await page.evaluate(() => {
      const nextSection = document.querySelector('.hero + .section');
      const controls = [...document.querySelectorAll<HTMLAnchorElement | HTMLButtonElement>('a, button')]
        .filter((node) => {
          const style = getComputedStyle(node);
          const rect = node.getBoundingClientRect();
          return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
        })
        .map((node) => {
          const rect = node.getBoundingClientRect();
          return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
        });
      const intersections = [];
      for (let left = 0; left < controls.length; left += 1) {
        for (let right = left + 1; right < controls.length; right += 1) {
          const a = controls[left];
          const b = controls[right];
          const overlapX = Math.min(a.right, b.right) - Math.max(a.left, b.left);
          const overlapY = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
          if (overlapX > 2 && overlapY > 2) intersections.push([left, right]);
        }
      }
      return {
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        nextSectionTop: nextSection?.getBoundingClientRect().top ?? Number.POSITIVE_INFINITY,
        intersections,
      };
    });

    expect(layout.overflow).toBeLessThanOrEqual(1);
    expect(layout.nextSectionTop).toBeLessThan(viewport.height);
    expect(layout.intersections).toEqual([]);
    const hero = await page.locator('.hero').screenshot();
    expect(hero.byteLength).toBeGreaterThan(20_000);
    await page.screenshot({ path: testInfo.outputPath(`home-${viewport.name}.png`), fullPage: true });
  });
}

test('mobile navigation supports keyboard operation and visible focus', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  const menu = page.locator('[data-menu-toggle]');
  await menu.focus();
  const outline = await menu.evaluate((node) => getComputedStyle(node).outlineWidth);
  expect(Number.parseFloat(outline)).toBeGreaterThan(0);
  await page.keyboard.press('Enter');
  await expect(menu).toHaveAttribute('aria-expanded', 'true');
  await expect(page.locator('[data-primary-nav]')).toHaveAttribute('data-open', 'true');
});

test('selector, language peers, copy, and compatibility routes work', async ({ browser }) => {
  const context = await browser.newContext({ permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:8770/');
  await page.locator('[data-selector-group="workload"] [data-value="edge"]').click();
  await page.locator('[data-selector-group="hardware"] [data-value="cpu"]').click();
  await page.locator('[data-selector-group="priority"] [data-value="portability"]').click();
  await expect(page.locator('[data-result-name]')).toContainText('llama.cpp');

  await page.goto('http://127.0.0.1:8770/deploy/llama-cpp.html');
  await expect(page.locator('a[href="/en/deploy/llama-cpp.html"]')).toBeVisible();
  const firstCopy = page.locator('[data-copy-target]').first();
  const target = await firstCopy.getAttribute('data-copy-target');
  await firstCopy.click();
  const expectedCommand = await page.locator(target!).innerText();
  const clipboard = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboard.trim()).toBe(expectedCommand.trim());

  await page.goto('http://127.0.0.1:8770/llama-cpp.html');
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    'href',
    'https://www.funasr.com/deploy/llama-cpp.html',
  );
  await context.close();
});

for (const viewport of [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'desktop', width: 1440, height: 900 },
]) {
  test(`SenseVoice TensorRT deployment is stable at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.goto('/deploy/sensevoice-tensorrt.html');

    await expect(page.locator('h1')).toHaveText('SenseVoice TensorRT / Triton');
    await expect(page.locator('[data-section="commands"] .command-block')).toHaveCount(9);
    await expect(page.locator('[data-section="smoke-test"] .command-block')).toHaveCount(1);
    await expect(page.locator('a[href="https://github.com/modelscope/FunASR/pull/3463"]')).toBeVisible();
    await expect(page.locator('a[href="/en/deploy/sensevoice-tensorrt.html"]')).toBeVisible();

    const layout = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      commandWidths: [...document.querySelectorAll<HTMLElement>('.command-block')].map((node) => ({
        parent: node.parentElement?.getBoundingClientRect().width ?? 0,
        width: node.getBoundingClientRect().width,
      })),
    }));
    expect(layout.overflow).toBeLessThanOrEqual(1);
    expect(layout.commandWidths.every(({ parent, width }) => width <= parent + 1)).toBe(true);

    await page.screenshot({
      path: testInfo.outputPath(`sensevoice-tensorrt-${viewport.name}.png`),
      fullPage: true,
    });
  });
}

test('reduced motion disables smooth scrolling', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/en/');
  const behavior = await page.locator('html').evaluate((node) => getComputedStyle(node).scrollBehavior);
  expect(behavior).toBe('auto');
});
