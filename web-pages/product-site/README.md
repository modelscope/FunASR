# FunASR product website and documentation

This is the source for www.funasr.com. It uses Jinja templates and a static
Python build; no browser framework or runtime application server is required.

## Content ownership

- `data/documentation.json`: task groups, bilingual titles and Markdown sources.
- Repository `docs/`, `model_zoo/`, `runtime/` and `examples/openai_api/`
  Markdown listed in the catalogue: documentation body, shared with GitHub readers.
- `data/deployments.json`: versions, hardware, evidence and operational limits.
- `data/blog.json`: bilingual editorial titles, categories and the five-story
  homepage selection. `blog.py` validates complete article coverage and renders
  reading categories, archive and release history through `templates/blog.html`.
  Published blog indexes are generated; the old `legacy/blog/index.html` and
  English counterpart remain historical snapshots, not the live ordering source.
- `templates/`: generated homepage, deployment manuals and documentation shell.
- `assets/css/experience.css`: shared product and legacy reading experience.
- `legacy/`: preserved public routes, content and demo assets. Do not delete or
  replace this snapshot with a partial output tree.
- Repository `gh-pages-output/index.html` and `zh/index.html`: GitHub Pages entry
  points. Their existing API/tutorial routes remain available.
- `export_docs.py`: generates established GitHub Pages guide URLs from the same
  rendered sources, with canonical navigation to www.funasr.com and a local copy
  of the fingerprinted assets so publication does not depend on cross-host timing.
- Repository `scripts/gen_api_docs.py`: extracts the separate API reference from
  Python source; `gh-pages-output/api-reference.css` owns its responsive layout.

The build emits search indexes, maps source-relative document links, preserves
Unicode heading anchors and fingerprints all product assets. Historical
benchmark evidence must retain its exact tested model/runtime/hardware; new
upstream support is not evidence that old tests covered a new checkpoint.

## Validate

```sh
python -m pip install -r web-pages/product-site/requirements-site.txt
python -m pytest web-pages/product-site/tests -q
python web-pages/product-site/build.py --output /tmp/funasr-site
python web-pages/product-site/validate.py /tmp/funasr-site
python scripts/gen_api_docs.py
python web-pages/product-site/export_docs.py --site /tmp/funasr-site --output gh-pages-output
```

Add a guide by listing both source paths, a unique slug and a known group in the
catalogue. Use repository-relative Markdown links; the renderer maps registered
guides to their local language routes. Keep explicit legacy anchors unique.
Preserve protocol distinctions and mark historical evidence as historical.
Do not hand-edit generated guide bodies in `gh-pages-output/`; the publication
workflow renders them again after the API generator runs. Specialist recipes
and historical notes remain linked from the curated guides rather than being
presented as current, universally supported deployment paths.

Browser tests live in `tests/browser/`. Install their pinned npm dependencies
and Chromium, then run `npm test` from that directory. The suite owns its local
HTTP server and rejects reuse of an occupied port.

Production releases use `web-pages/scripts/deploy-product-site.sh`; see
`docs/operations/funasr-com-site-release.md` for backup, atomic switching and
rollback. Only publish a tested artifact, never a partially built output.
