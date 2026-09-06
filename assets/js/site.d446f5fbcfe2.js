(() => {
  const menuButton = document.querySelector('[data-menu-toggle]');
  const navigation = document.querySelector('[data-primary-nav]');

  if (menuButton && navigation) {
    menuButton.addEventListener('click', () => {
      const open = menuButton.getAttribute('aria-expanded') !== 'true';
      menuButton.setAttribute('aria-expanded', String(open));
      navigation.dataset.open = String(open);
    });
  }

  const dataNode = document.getElementById('deployment-selector-data');
  const selector = document.querySelector('[data-deployment-selector]');
  const result = document.querySelector('[data-selector-result]');

  if (dataNode && selector && result) {
    const data = JSON.parse(dataNode.textContent);
    const language = document.body.dataset.language || 'zh';
    const selection = {
      workload: 'batch',
      hardware: 'nvidia-gpu',
      priority: 'throughput',
    };

    const score = (entry) => {
      let value = 0;
      value += entry.workloads.includes(selection.workload) ? data.weights.workload : 0;
      value += entry.hardware.includes(selection.hardware) ? data.weights.hardware : 0;
      value += entry.priorities.includes(selection.priority) ? data.weights.priority : 0;
      return value;
    };

    const render = () => {
      const recommendation = [...data.entries].sort((left, right) => {
        const scoreDifference = score(right) - score(left);
        if (scoreDifference !== 0) return scoreDifference;
        if (left.rank !== right.rank) return left.rank - right.rank;
        return left.id.localeCompare(right.id);
      })[0];

      result.querySelector('[data-result-name]').textContent = recommendation.name[language];
      result.querySelector('[data-result-reason]').textContent = recommendation.reason[language];
      result.querySelector('[data-result-limitation]').textContent = recommendation.limitation[language];
      result.querySelector('[data-result-link]').href = recommendation.routes[language];
    };

    selector.querySelectorAll('[data-selector-group]').forEach((group) => {
      group.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-value]');
        if (!button) return;
        group.querySelectorAll('button').forEach((item) => {
          item.setAttribute('aria-pressed', String(item === button));
        });
        selection[group.dataset.selectorGroup] = button.dataset.value;
        render();
      });
    });

    render();
  }

  document.querySelectorAll('[data-copy-target]').forEach((button) => {
    button.addEventListener('click', async () => {
      const target = document.querySelector(button.dataset.copyTarget);
      if (!target) return;
      const original = button.title;
      try {
        await navigator.clipboard.writeText(target.textContent.trim());
        button.dataset.copied = 'true';
        button.title = document.body.dataset.language === 'zh' ? '已复制' : 'Copied';
      } catch {
        button.title = document.body.dataset.language === 'zh' ? '复制失败，请选择代码' : 'Copy failed; select the code';
      }
      button.setAttribute('aria-label', button.title);
      window.setTimeout(() => { delete button.dataset.copied; button.title = original; button.setAttribute('aria-label', original); }, 1400);
    });
  });
})();
