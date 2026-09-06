(() => {
  const language = document.body.dataset.language || 'zh';
  const chinese = language === 'zh';
  let searchData;
  const getSearchData = () => {
    if (!searchData) {
      searchData = fetch(`/search-${language}.json`).then((response) => {
        if (!response.ok) throw new Error('search_unavailable');
        return response.json();
      }).catch((error) => { searchData = null; throw error; });
    }
    return searchData;
  };
  document.querySelectorAll('[data-doc-search]').forEach((form) => {
    const input = form.querySelector('input');
    const results = form.querySelector('.search-results');
    let timer;
    const show = async () => {
      const query = input.value.trim().toLocaleLowerCase();
      if (!query) { results.hidden = true; results.replaceChildren(); return; }
      try {
        const entries = await getSearchData();
        if (query !== input.value.trim().toLocaleLowerCase()) return;
        const terms = query.split(/\s+/);
        const ranked = entries.map((entry) => {
          const title = entry.title.toLocaleLowerCase();
          const text = entry.text.toLocaleLowerCase();
          return { ...entry, score: terms.every((term) => (title + text).includes(term))
            ? terms.reduce((score, term) => score + (title.includes(term) ? 10 : 1), 0) : 0 };
        }).filter((entry) => entry.score).sort((a, b) => b.score - a.score).slice(0, 10);
        results.replaceChildren();
        for (const entry of ranked) {
          const link = document.createElement('a');
          link.href = entry.url;
          const title = document.createElement('strong');
          title.textContent = entry.title;
          const snippet = document.createElement('small');
          const offset = Math.max(0, entry.text.toLocaleLowerCase().indexOf(terms[0]) - 35);
          snippet.textContent = entry.text.slice(offset, offset + 130);
          link.append(title, snippet);
          results.append(link);
        }
        if (!ranked.length) {
          const message = document.createElement('p');
          message.textContent = chinese ? '没有找到相关文档。试试模型名称或部署方式。' : 'No matching documents. Try a model or runtime name.';
          results.append(message);
        }
        results.hidden = false;
      } catch {
        results.replaceChildren();
        const message = document.createElement('p');
        message.textContent = chinese ? '搜索暂时不可用，请从文档目录浏览。' : 'Search is unavailable. Browse the documentation index instead.';
        results.append(message);
        results.hidden = false;
      }
    };
    input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(show, 120); });
    form.addEventListener('submit', (event) => { event.preventDefault(); clearTimeout(timer); show(); });
    form.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') { results.hidden = true; input.focus(); }
      if (event.key === 'ArrowDown' && event.target === input) { event.preventDefault(); results.querySelector('a')?.focus(); }
    });
    document.addEventListener('click', (event) => { if (!form.contains(event.target)) results.hidden = true; });
    const query = new URLSearchParams(location.search).get('q');
    if (query) { input.value = query; show(); }
  });
  const mobileDocs = matchMedia('(max-width: 720px)');
  if (mobileDocs.matches) {
    document.querySelectorAll('.docs-sidebar details').forEach((group) => { group.open = false; });
  }
  mobileDocs.addEventListener('change', ({ matches }) => {
    const navigation = document.querySelector('.docs-navigation');
    if (navigation) navigation.open = !matches;
  });
  document.querySelectorAll('.docs-article pre, .legacy-page article pre').forEach((pre) => {
    if (pre.querySelector('button')) return;
    const text = pre.textContent.trim();
    const button = document.createElement('button');
    button.className = 'icon-button command-copy';
    button.type = 'button';
    const original = chinese ? '复制代码' : 'Copy code';
    button.title = original;
    button.setAttribute('aria-label', original);
    const icon = document.createElement('img');
    icon.src = document.body.dataset.copyIcon;
    icon.alt = '';
    button.append(icon);
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(text);
        button.title = chinese ? '已复制' : 'Copied';
        button.dataset.copied = 'true';
      } catch {
        button.title = chinese ? '复制失败，请选择代码' : 'Copy failed; select the code';
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(pre.querySelector('code') || pre);
        selection.removeAllRanges(); selection.addRange(range);
      }
      button.setAttribute('aria-label', button.title);
      setTimeout(() => { button.title = original; button.setAttribute('aria-label', original); delete button.dataset.copied; }, 2000);
    });
    pre.append(button);
  });
  const menu = document.querySelector('[data-menu-toggle]');
  const navigation = document.querySelector('[data-primary-nav]');
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && menu?.getAttribute('aria-expanded') === 'true') {
      menu.setAttribute('aria-expanded', 'false'); navigation.dataset.open = 'false'; menu.focus();
    }
  });
})();
