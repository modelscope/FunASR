"""Connect API references to maintained guides without renumbering entries."""

import argparse
from pathlib import Path

from bs4 import BeautifulSoup


def link_api_guides(output: Path, previous: Path) -> None:
    rendered = {}
    for route, prefix, labels in (
        ('api.html', 'en/', ('Native Transformers quickstart', 'Python SDK guide')),
        ('zh/api.html', '', ('原生 Transformers 入门', 'Python SDK 指南')),
    ):
        target = output / route
        # The Chinese reference is retained by keep_files, not generated on main.
        source = target if target.exists() else previous / route
        soup = BeautifulSoup(source.read_text(encoding='utf-8'), 'html.parser')
        welcome = soup.select('#api-welcome')
        existing = soup.select('#api-practical-guides')
        if len(welcome) != 1 or len(existing) > 1:
            raise ValueError(f'{route}: expected one api-welcome and at most one guide entry')
        if existing:
            existing[0].decompose()
        guides = soup.new_tag('p', id='api-practical-guides')
        for index, (slug, label) in enumerate(zip(('native-transformers', 'python-api'), labels)):
            if index:
                guides.append(' | ')
            link = soup.new_tag('a', href=f'https://www.funasr.com/{prefix}docs/{slug}.html')
            link.string = label
            guides.append(link)
        welcome[0].append(guides)
        rendered[target] = str(soup)

    # Validate both inputs before changing the outgoing publication directory.
    for target, content in rendered.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    link_api_guides(args.output, args.previous)
