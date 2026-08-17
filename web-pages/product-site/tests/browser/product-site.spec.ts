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

for (const viewport of [
  { name: 'mobile', width: 390, height: 844, expectedRows: 4 },
  { name: 'desktop', width: 1440, height: 900, expectedRows: 1 },
]) {
  test(`four-project selector is stable at ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const consoleErrors: string[] = [];
    const networkErrors: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('requestfailed', (request) => networkErrors.push(request.url()));

    for (const route of ['/', '/en/']) {
      await page.goto(route);
      const projects = page.locator('#projects [data-project]');
      await expect(projects).toHaveCount(4);
      expect(
        await projects.evaluateAll((nodes) =>
          nodes.map((node) => node.getAttribute('data-project')),
        ),
      ).toEqual(['funasr', 'fun-asr', 'sensevoice', 'funclip']);

      const layout = await projects.evaluateAll((nodes) => ({
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        rows: new Set(nodes.map((node) => Math.round(node.getBoundingClientRect().top))).size,
        visible: nodes.every((node) => {
          const rect = node.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        }),
      }));
      expect(layout.overflow).toBeLessThanOrEqual(1);
      expect(layout.rows).toBe(viewport.expectedRows);
      expect(layout.visible).toBe(true);
    }

    expect(consoleErrors).toEqual([]);
    expect(networkErrors).toEqual([]);
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
  test(`llama.cpp v0.2.0 download matrix is stable at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.goto('/deploy/llama-cpp.html');

    const section = page.locator('[data-section="downloads"]');
    await expect(section.locator('[data-download-asset]')).toHaveCount(9);
    await expect(section.locator('a[href*="runtime-llamacpp-v0.2.0"]')).toHaveCount(9);
    await expect(page.getByText('Windows AMD Vulkan', { exact: false }).first()).toBeVisible();

    await section.evaluate((node) => node.scrollIntoView({ block: 'start' }));

    const layout = await page.evaluate(() => {
      const tableWrap = document.querySelector<HTMLElement>('.download-table');
      const header = document.querySelector<HTMLElement>('.site-header');
      const heading = document.querySelector<HTMLElement>('[data-section="downloads"] .section-heading');
      return {
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        tableClientWidth: tableWrap?.clientWidth ?? 0,
        tableScrollWidth: tableWrap?.scrollWidth ?? 0,
        headerBottom: header?.getBoundingClientRect().bottom ?? 0,
        headingTop: heading?.getBoundingClientRect().top ?? 0,
      };
    });
    expect(layout.overflow).toBeLessThanOrEqual(1);
    expect(layout.tableClientWidth).toBeGreaterThan(0);
    expect(layout.tableScrollWidth).toBeGreaterThanOrEqual(layout.tableClientWidth);
    expect(layout.headingTop).toBeGreaterThanOrEqual(layout.headerBottom + 8);

    await page.screenshot({
      path: testInfo.outputPath(`llama-cpp-downloads-${viewport.name}.png`),
      fullPage: true,
    });
  });
}

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

for (const viewport of [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'desktop', width: 1440, height: 900 },
]) {
  test(`audio.cpp deployment is stable at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.goto('/deploy/audio-cpp.html');

    await expect(page.locator('h1')).toHaveText('audio.cpp 原生 Fun-ASR-Nano 与 SenseVoice');
    await expect(page.getByText('pinned SenseVoice GGUF package', { exact: false })).toBeVisible();
    await expect(page.locator('a[href="https://github.com/0xShug0/audio.cpp/pull/219"]')).toBeVisible();

    const layout = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      evidenceWidths: [...document.querySelectorAll<HTMLElement>('.evidence-list li')].map((node) => ({
        client: node.clientWidth,
        scroll: node.scrollWidth,
      })),
      commandWidths: [...document.querySelectorAll<HTMLElement>('.command-block')].map((node) => ({
        parent: node.parentElement?.getBoundingClientRect().width ?? 0,
        width: node.getBoundingClientRect().width,
      })),
    }));
    expect(layout.overflow).toBeLessThanOrEqual(1);
    expect(layout.evidenceWidths.every(({ client, scroll }) => scroll <= client + 1)).toBe(true);
    expect(layout.commandWidths.every(({ parent, width }) => width <= parent + 1)).toBe(true);

    await page.screenshot({
      path: testInfo.outputPath(`audio-cpp-${viewport.name}.png`),
      fullPage: true,
    });
  });
}

for (const viewport of [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'desktop', width: 1440, height: 900 },
]) {
  test(`MLX Audio ecosystem entry is stable at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);

    for (const route of ['/ecosystem.html', '/en/ecosystem.html']) {
      await page.goto(route);
      const card = page.locator('.card').filter({ hasText: 'MLX Audio' });

      await expect(card).toHaveCount(1);
      await expect(card.locator('.card-tag', { hasText: 'Fun-ASR-Nano' })).toBeVisible();
      await expect(card.locator('a[href="https://github.com/Blaizzy/mlx-audio/pull/885"]')).toBeVisible();
      await expect(card.locator('a[href="/go/fun-asr"]')).toBeVisible();

      const layout = await page.evaluate(() => ({
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      }));
      expect(layout.overflow).toBeLessThanOrEqual(1);
    }

    await page.screenshot({
      path: testInfo.outputPath(`mlx-audio-ecosystem-${viewport.name}.png`),
      fullPage: true,
    });
  });
}

for (const viewport of [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'desktop', width: 1440, height: 900 },
]) {
  test(`OpenMAIC ecosystem entry is stable at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);

    for (const route of ['/ecosystem.html', '/en/ecosystem.html']) {
      await page.goto(route);
      const card = page.locator('.card').filter({ hasText: 'OpenMAIC' });

      await expect(card).toHaveCount(1);
      await expect(card.locator('.card-tag', { hasText: 'Local ASR' })).toBeVisible();
      await expect(card.locator('a[href="https://github.com/THU-MAIC/OpenMAIC/pull/1044"]')).toBeVisible();
      await expect(card.locator('a[href="https://github.com/modelscope/FunASR"]')).toBeVisible();

      const layout = await page.evaluate(() => ({
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      }));
      expect(layout.overflow).toBeLessThanOrEqual(1);
    }

    await page.screenshot({
      path: testInfo.outputPath(`openmaic-ecosystem-${viewport.name}.png`),
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

test('llama.cpp blog heading clears fixed navigation on mobile', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/blog/funasr-llama-cpp-whisper-cpp-alternative.html');

  const layout = await page.evaluate(() => {
    const navigation = document.querySelector<HTMLElement>('nav.nav');
    const heading = document.querySelector<HTMLElement>('h1');
    return {
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      navigationBottom: navigation?.getBoundingClientRect().bottom ?? 0,
      headingTop: heading?.getBoundingClientRect().top ?? 0,
    };
  });

  expect(layout.overflow).toBeLessThanOrEqual(1);
  expect(layout.headingTop).toBeGreaterThanOrEqual(layout.navigationBottom + 16);
  await page.screenshot({
    path: testInfo.outputPath('llama-cpp-blog-mobile.png'),
    fullPage: true,
  });
});

test('legacy comparison pages keep accurate claims and fit mobile', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });

  for (const route of ['/vs-whisper.html', '/en/vs-whisper.html']) {
    await page.goto(route);
    const audit = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      text: document.body.innerText,
      trackedGitHub: document.querySelector('.nav-btn')?.getAttribute('href'),
    }));

    expect(audit.overflow).toBeLessThanOrEqual(1);
    expect(audit.text).not.toMatch(/50\+\s*(?:supported\s+|支持\s*)?(?:languages?|语言|语种)/i);
    expect(audit.text).not.toMatch(/(?:13|15|17|170)[x×]|(?:13|15|17|170)\s*倍/);
    expect(audit.trackedGitHub).toBe('/go/github');
  }

  await page.screenshot({
    path: testInfo.outputPath('legacy-comparison-mobile.png'),
    fullPage: true,
  });
});

test('SenseVoice guides keep mobile navigation clear of the article', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });

  for (const route of [
    '/blog/sensevoice-deployment-guide.html',
    '/en/blog/sensevoice-deployment-guide.html',
  ]) {
    await page.goto(route);
    const layout = await page.evaluate(() => {
      const navigation = document.querySelector<HTMLElement>('nav.nav');
      const heading = document.querySelector<HTMLElement>('h1');
      return {
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        navigationBottom: navigation?.getBoundingClientRect().bottom ?? 0,
        headingTop: heading?.getBoundingClientRect().top ?? 0,
        trackedSenseVoice: document.querySelector('.cta a')?.getAttribute('href'),
      };
    });

    expect(layout.overflow).toBeLessThanOrEqual(1);
    expect(layout.headingTop).toBeGreaterThanOrEqual(layout.navigationBottom + 16);
    expect(layout.trackedSenseVoice).toBe('/go/sensevoice');
  }

  await page.screenshot({
    path: testInfo.outputPath('sensevoice-guide-mobile.png'),
    fullPage: true,
  });
});
