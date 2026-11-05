# Configuration file for the Sphinx documentation builder.

from __future__ import annotations
import re
from pathlib import Path
# `brisc` is imported from the active environment's site-packages; no
# sys.path mutation needed.

project = "brisc"
author = "Keon Arbabi & Michael Wainberg"
copyright = "2026, Keon Arbabi & Michael Wainberg"
# Suffix of every page's <title> ("Basic Workflow — brisc"). Kept short: the
# longer "brisc documentation" repeated the site name and Google rewrote the
# titles. The navbar text is set separately (html_theme_options["logo"]).
html_title = "brisc"
# Canonical site URL — emits <link rel="canonical"> so search engines treat
# https://brisc.run as authoritative regardless of which host served the page.
html_baseurl = "https://brisc.run/"

templates_path = ["_templates"]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# Allow both .rst and .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST extensions for richer Markdown
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "deflist",
]

# Auto-generate anchors for H1-H3, so method-group headings are linkable
myst_heading_anchors = 3

# -- Autodoc / Autosummary configuration -------------------------------------

# Pull members in source order (matches the order in single_cell.py)
autodoc_member_order = "bysource"

# Move type hints from signature to parameter descriptions (Scanpy style)
autodoc_typehints = "description"

# Document only the constructor (__init__) docstring on each class page, not the
# class-level docstring: the class "description" (e.g. SingleCell's slots
# overview) duplicates the Properties / Data-access tables below. The
# constructor signature stays on the class line (default
# autodoc_class_signature = "mixed").
autoclass_content = "init"

# Don't force one-parameter-per-line wrapping; let CSS handle natural wrapping
maximum_signature_line_length = 10000

# Type alias display is handled by the _semantic_highlight post-processor
autodoc_type_aliases = {}

# Generate stub files from autosummary directives
autosummary_generate = True

# Shorten every autosummary stub filename from the fully-qualified object name
# to just the member: `brisc.SingleCell.hvg` -> `hvg`. With the class dirs
# renamed to `singlecell/` etc. and stubs generated beside the index (no `api/`
# subdir), this yields short URLs like `/singlecell/hvg` instead of
# `/api/single_cell/api/brisc.SingleCell.hvg`. Same basename across classes
# (e.g. `qc`) is fine -- each lands in its own class directory.
def _build_filename_map():
    try:
        from brisc import SingleCell, Pseudobulk, DE
    except Exception:
        # Falls back to full-name stubs (e.g. under a Sphinx-only env that
        # can't import brisc); real builds run in the brisc env.
        return {}
    fmap = {}
    for cls in (SingleCell, Pseudobulk, DE):
        for name in dir(cls):
            if name == "__init__":
                continue  # documented on the class page, not a stub
            # keep public names and dunders; skip _private
            if name.startswith("_") and not (
                    name.startswith("__") and name.endswith("__")):
                continue
            fmap[f"brisc.{cls.__name__}.{name}"] = name
    return fmap

autosummary_filename_map = _build_filename_map()

# Napoleon settings (Google-style docstrings with Args:, Returns:, Note:)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_favicon = "_static/images/favicon.ico"

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = [
    "js/benchmark-data.js",
    "js/benchmark.js",
    "js/theme-fix.js",
    "js/api-scrollspy.js",
    "js/header-reveal.js",
    "js/details-anchor.js",
]
html_show_sourcelink = False
# Don't ship the raw .md/.rst sources under _sources/: nothing links to them
# (no source link above), and they are duplicate copies of every page for
# search engines to index.
html_copy_source = False

html_theme_options = {
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/briscverse/brisc",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_version_warning_banner": False,
    # "On this page" secondary sidebar: rendered as a page TOC, but the CSS
    # (body:not(.tutorial-page)) shows it only on tutorial content pages and
    # hides it elsewhere (API pages navigate via the left sidebar).
    "secondary_sidebar_items": ["page-toc"],
    "logo": {
        "image_dark": "_static/images/runner_logo_dark.svg",
        "image_light": "_static/images/runner_logo_light.svg",
        "text": "brisc documentation",
        "alt_text": "brisc documentation",
    },
}

# sidebar-nav-bs provides the full collapsible toctree navigation. The
# Installation page and every Tutorials page instead share one unified "guide"
# sidebar (Installation + the Tutorials tree), rendered by
# _templates/guide-nav.html. _GUIDE_TUTORIALS is the single source of truth for
# the tutorial entries: handed to the template via html_context and used to
# build the per-page html_sidebars overrides below.
_GUIDE_TUTORIALS = [
    ("tutorials/basic_workflow", "Basic Workflow"),
    ("tutorials/integration_and_label_transfer", "Integration and Label Transfer"),
    ("tutorials/differential_expression", "Differential Expression"),
    ("tutorials/interoperability", "Interoperability"),
    ("tutorials/data_manipulation", "Data manipulation"),
]
html_context = {"guide_tutorials": _GUIDE_TUTORIALS}

html_sidebars = {"**": ["sidebar-nav-bs"]}
for _guide_page in ["installation", "tutorials/index",
                    *(doc for doc, _ in _GUIDE_TUTORIALS)]:
    html_sidebars[_guide_page] = ["guide-nav"]

# Strip prompt prefixes from code copy
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Fix up Markdown-isms in docstrings on the fly -------------------------

# Match Markdown links in docstrings -- tolerates whitespace (including
# a line break) between `]` and `(`, and URLs without an http(s):// prefix.
# Group 1 is the optional backtick pair (a `[`code`](url)` link vs plain
# `[text](url)`); group 2 the label, group 3 the URL.
_md_link_re = re.compile(
    r"\[(`?)([^\]]+?)\1\]\s*\(([^\s\)]+)\)",
    re.DOTALL,
)

def _md_link_to_rst(m):
    """Convert one Markdown link match to RST. A backtick-wrapped label
    ([`code`](url)) keeps its code formatting by becoming a `codelink` role
    (an inline-code hyperlink); a plain label becomes a normal RST hyperlink."""
    code, label, url = m.group(1), m.group(2), m.group(3)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if code:
        return f":codelink:`{label} <{url}>`"
    return f"`{label} <{url}>`_"

# Markdown fenced code blocks (```lang ... ```) are not valid RST -- autodoc
# parses docstrings as reStructuredText, so the fence renders as literal
# "`lang" text. Convert them to RST literal blocks. (brisc's docstrings use
# Markdown fences throughout; fixing them here avoids editing the source.)
_fence_open_re = re.compile(r"^(\s*)```(\w*)\s*$")
_fence_close_re = re.compile(r"^\s*```\s*$")

def _md_fences_to_rst(lines):
    out = []
    i, n = 0, len(lines)
    while i < n:
        m = _fence_open_re.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, lang = m.group(1), m.group(2)
        # Gather the fenced body up to the closing ``` (or end of docstring).
        body, j = [], i + 1
        while j < n and not _fence_close_re.match(lines[j]):
            body.append(lines[j])
            j += 1
        # Emit an RST code block, re-indenting the body 4 spaces past the
        # directive while preserving each line's relative indentation.
        out.append(f"{indent}.. code-block::{(' ' + lang) if lang else ''}")
        out.append("")
        for b in body:
            if not b.strip():
                out.append("")
            else:
                stripped = b[len(indent):] if b.startswith(indent) else b.lstrip()
                out.append(f"{indent}    {stripped}")
        out.append("")
        i = j + 1  # skip the closing fence
    return out

def _md_to_rst_links(app, what, name, obj, options, lines):
    # 1) Markdown code fences -> RST literal blocks.
    fenced = _md_fences_to_rst(lines)
    if fenced != lines:
        lines[:] = fenced
    # 2) Markdown links -> RST links.
    text = "\n".join(lines)
    new_text = _md_link_re.sub(_md_link_to_rst, text)
    if new_text != text:
        lines[:] = new_text.split("\n")

# Drop the __init__ opening summary that would duplicate the separately-rendered
# class description (autoclass_content="init" shows only the __init__ docstring;
# the class docstring is emitted via the classdescription directive). Stripped at
# build time so the brisc source docstrings stay untouched; each constructor's
# parameters are preserved.
_DROP_CONSTRUCTOR_SUMMARIES = {
    "brisc.DE": "Initialize the DE object.",
    "brisc.Pseudobulk":
        "Load a saved Pseudobulk dataset, or create one from an in-memory "
        "count matrix + metadata for each cell type.",
}

def _drop_constructor_summary(app, what, name, obj, options, lines):
    # `name` is the class fullname when autoclass merges in the __init__
    # docstring, or the constructor itself; handle both.
    cls_name = name[:-9] if name.endswith(".__init__") else name
    summary = _DROP_CONSTRUCTOR_SUMMARIES.get(cls_name)
    if not summary:
        return
    target = summary.split()
    # Find the leading paragraph (run of non-blank lines) whose whitespace-
    # normalized text equals `summary`, and delete it plus one trailing blank.
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        j, words = i, []
        while j < len(lines) and lines[j].strip():
            words += lines[j].split()
            j += 1
        if words == target:
            if j < len(lines) and not lines[j].strip():
                j += 1
            del lines[i:j]
            return
        i = j

# Fold SingleCell's long constructor `Examples:` and `Note:` sections into two
# collapsible sphinx-design dropdowns, so the class page stays scannable. Runs
# BEFORE napoleon (priority 400 in setup()) so it operates on the raw Google-
# style sections; only SingleCell's constructor has these.
def _section_span(lines, header, start=0):
    """Return [i, j) covering a Google `header` line and its more-indented
    body (blank lines included), or None if not found."""
    for i in range(start, len(lines)):
        if lines[i].strip() == header:
            base = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            while j < len(lines):
                s = lines[j]
                if s.strip() and (len(s) - len(s.lstrip())) <= base:
                    break
                j += 1
            return i, j
    return None

def _strip_trailing_blanks(block):
    while block and not block[-1].strip():
        block.pop()
    return block

def _constructor_dropdowns(app, what, name, obj, options, lines):
    cls_name = name[:-9] if name.endswith(".__init__") else name
    if cls_name != "brisc.SingleCell":
        return
    ex = _section_span(lines, "Examples:")
    if ex is None:
        return
    ex_s, ex_e = ex
    # Collect the consecutive `Note:` sections that follow Examples.
    notes, k = [], ex_e
    while True:
        nspan = _section_span(lines, "Note:", k)
        if nspan is None or nspan[0] != k:
            break
        notes.append(nspan)
        k = nspan[1]
    region_end = notes[-1][1] if notes else ex_e
    # Examples dropdown (body keeps its existing indentation, valid as content).
    block = [".. dropdown:: Examples", ""] + \
        _strip_trailing_blanks(lines[ex_s + 1:ex_e])
    # One Notes dropdown holding every Note paragraph.
    if notes:
        block += ["", ".. dropdown:: Notes", ""]
        for idx, (ns, ne) in enumerate(notes):
            if idx:
                block.append("")
            block += _strip_trailing_blanks(lines[ns + 1:ne])
    lines[ex_s:region_end] = block

# Render a class's docstring as standalone prose, so the class "description" can
# sit between the page title and the constructor signature in index.rst
# (autoclass always renders the docstring *after* the signature). Single source
# of truth: the class docstring itself, not a hand-copied blurb.
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
import inspect as _inspect

class _ClassDescription(Directive):
    required_arguments = 1  # class name in `brisc`, e.g. SingleCell

    def run(self):
        import brisc
        cls = getattr(brisc, self.arguments[0])
        doc_lines = _md_fences_to_rst(
            _inspect.cleandoc(cls.__doc__ or "").split("\n"))
        text = "\n".join(doc_lines)
        text = _md_link_re.sub(_md_link_to_rst, text)
        container = nodes.container(classes=["class-description"])
        self.state.nested_parse(
            StringList(text.split("\n"), source="<class-description>"),
            self.content_offset, container)
        return [container]

# `codelink` role: renders `:codelink:`label <url>`` as an inline-code
# hyperlink (a <reference> wrapping a <literal>), so backtick-wrapped Markdown
# links in docstrings -- [`os.cpu_count()`](url) -- keep their code formatting
# instead of collapsing to plain prose-link text. RST can't nest inline markup
# inside a hyperlink, so we build the node tree directly.
from docutils.utils import unescape as _unescape

_codelink_target_re = re.compile(r"^(?P<label>.*?)\s*<(?P<url>[^>]+)>$", re.DOTALL)

def _codelink_role(name, rawtext, text, lineno, inliner,
                   options=None, content=None):
    m = _codelink_target_re.match(_unescape(text))
    label, url = (m.group("label"), m.group("url")) if m else (text, text)
    ref = nodes.reference(rawtext, "", refuri=url)
    ref += nodes.literal("", label)
    return [ref], []

def _build_api_links():
    """Build a dict of method_name → relative URL for all documented methods."""
    from brisc import SingleCell, Pseudobulk, DE
    links = {}
    # class names → their index pages
    links['SingleCell'] = 'singlecell/index.html'
    links['Pseudobulk'] = 'pseudobulk/index.html'
    links['DE'] = 'de/index.html'
    # methods and properties: each stub lives at <class-dir>/<member>.html
    for cls, prefix in [(SingleCell, 'singlecell/'),
                        (Pseudobulk, 'pseudobulk/'),
                        (DE, 'de/')]:
        for name in dir(cls):
            if name.startswith('_') and name != '__init__':
                continue
            url = f'{prefix}{name}.html'
            links[(cls.__name__, name)] = url
            # bare name → SingleCell takes priority, but don't overwrite class names
            if name not in links or cls is SingleCell:
                links[name] = url
    return links

# Built lazily on first use (see _get_api_links) to keep Sphinx startup fast —
# importing single_cell pulls in h5py/numpy/polars/pyarrow/scipy and takes ~2min.
_api_links = None

def _get_api_links():
    global _api_links
    if _api_links is None:
        _api_links = _build_api_links()
    return _api_links

# Pattern: <span class="n">NAME</span><span class="o">=</span>  → keyword arg
_kwarg_re = re.compile(
    r'<span class="n">([^<]+)</span>'
    r'(<span class="o">=</span>)')

# Receiver-variable conventions and method return types, so qualified/chained
# calls resolve to the right class (e.g. pb.qc -> Pseudobulk.qc, not
# SingleCell.qc) when a method name is shared across classes.
_VAR_CLASS = {
    'sc': 'SingleCell', 'sc_ref': 'SingleCell', 'sc_query': 'SingleCell',
    'sc_data': 'SingleCell', 'single_cell': 'SingleCell',
    'pb': 'Pseudobulk', 'de': 'DE',
}
_CLASSES = {'SingleCell', 'Pseudobulk', 'DE'}
_RETURNS = {
    ('SingleCell', 'pseudobulk'): 'Pseudobulk',
    ('Pseudobulk', 'de'): 'DE',
}
# A call site: optional `RECV.` receiver, then NAME, then an opening paren.
_link_re = re.compile(
    r'(?:<span class="n">(?P<recv>[A-Za-z_]\w*)</span>)?'
    r'(?P<dot><span class="o">\.</span>)?'
    r'<span class="n">(?P<name>[A-Za-z_]\w*)</span>'
    r'(?P<paren><span class="p">\([^<]*</span>)')

def _linked(name, url, depth):
    if url:
        rel = '../' * depth + url
        return (f'<a href="{rel}" class="api-link" '
                f'style="text-decoration:none;color:inherit">'
                f'<span class="nf">{name}</span></a>')
    return f'<span class="nf">{name}</span>'

def _link_calls(text, depth):
    """Link calls to API docs, resolving the receiving class from the receiver
    variable or the method chain so shared names (qc, filter_obs, DE) point at
    the correct class page."""
    links = _get_api_links()
    state = {'cur': None}  # class the next chained `.method()` is called on

    def repl(m):
        recv, dot, name, paren = (m.group('recv'), m.group('dot'),
                                  m.group('name'), m.group('paren'))
        if dot:  # method call: [recv].name(
            if recv is not None:
                cls = _VAR_CLASS.get(recv) or (recv if recv in _CLASSES
                                               else None)
            else:
                cls = state['cur']  # chained call off the previous result
            url = (links.get((cls, name)) if cls else None) or links.get(name)
            if cls and (cls, name) in _RETURNS:
                state['cur'] = _RETURNS[(cls, name)]
            elif cls and (cls, name) in links:
                state['cur'] = cls
            recv_span = f'<span class="n">{recv}</span>' if recv else ''
            return recv_span + dot + _linked(name, url, depth) + paren
        if name in _CLASSES:  # bare call: constructor anchors the chain
            state['cur'] = name
        return _linked(name, links.get(name), depth) + paren

    return _link_re.sub(repl, text)

# Inline `code` references in docstrings render as <cite> (RST's default
# title-reference role for single backticks). Link the ones that name an API
# method or class -- qualified (Class.method), a bare call (method()), or a bare
# class name -- to their reference page, without touching the docstrings. Bare
# words and parameter names (no parens, not a class) are left alone.
_CITE_RE = re.compile(r'<cite>([^<]+)</cite>')
_CITE_REF_RE = re.compile(
    r'^(?:(SingleCell|Pseudobulk|DE)\.)?([A-Za-z_]\w*)(\(\s*[^)]*\s*\))?$')

def _link_inline_refs(text, depth, current_class):
    links = _get_api_links()
    classes = {'SingleCell', 'Pseudobulk', 'DE'}
    def repl(m):
        pm = _CITE_REF_RE.match(m.group(1).strip())
        if not pm:
            return m.group(0)
        cls_prefix, name, call = pm.group(1), pm.group(2), pm.group(3)
        if cls_prefix:                    # Class.method (qualified)
            url = links.get((cls_prefix, name))
        elif call:                        # bare method() -- resolve in page context
            url = (links.get((current_class, name)) if current_class else None) \
                  or links.get(name)
        elif name in classes:             # bare class name
            url = links.get(name)
        else:
            url = None
        if not url:
            return m.group(0)
        rel = '../' * depth + url
        return (f'<a href="{rel}" class="api-link" '
                f'style="text-decoration:none;color:inherit">{m.group(0)}</a>')
    return _CITE_RE.sub(repl, text)

# Within an API page, the parameter list (built by the Scanpy-style _param_repl
# pass below) is the set of names a docstring might reference. Give each
# parameter an id anchor, then turn any bare `name` cite that matches a parameter
# into a same-page jump to its description.
_PARAM_HEADER_RE = re.compile(r'<li><p class="param-header"><strong>(\w+)</strong>')

def _link_param_refs(text):
    params = set(_PARAM_HEADER_RE.findall(text))
    if not params:
        return text
    text = _PARAM_HEADER_RE.sub(
        lambda m: f'<li id="param-{m.group(1)}">'
                  f'<p class="param-header"><strong>{m.group(1)}</strong>',
        text)
    def repl(m):
        if m.group(1) not in params:
            return m.group(0)
        return (f'<a href="#param-{m.group(1)}" class="api-link" '
                f'style="text-decoration:none;color:inherit">{m.group(0)}</a>')
    return re.sub(r'<cite>(\w+)</cite>', repl, text)

_SIDEBAR_TITLE_MAP = [
    ("singlecell/", "SingleCell"),
    ("pseudobulk/", "Pseudobulk"),
    ("de/",         "DE"),
]
_SIDEBAR_TITLE_RE = re.compile(
    r'<p class="bd-links__title"[^>]*>Section Navigation</p>'
)
_SIDEBAR_ENTRY_RE = re.compile(
    r'>brisc\.(SingleCell|Pseudobulk|DE)\.([A-Za-z_][A-Za-z0-9_]*)<'
)

# Autosummary tables list each member by its qualified `Class.method` name.
# Drop the `Class.` prefix so the method column reads `find_doublets`, not
# `SingleCell.find_doublets`. Scoped to the generic-object xref (`py-obj`) that
# autosummary emits for its entries, so inline prose cross-references (rendered
# as `py-attr`/`py-meth`, e.g. `SingleCell.uns` in the typedefs prose) keep
# their qualifier. The link href and hover title stay fully qualified.
_AUTOSUMMARY_QUALIFIED_RE = re.compile(
    r'(<code class="xref py py-obj[^"]*"><span class="pre">)'
    r'(?:SingleCell|Pseudobulk|DE)\.'
    r'([A-Za-z_]\w*)'
    r'(</span></code>)'
)

# The "Return type" field's <dd>, so CSS can render its whole value as a
# monospace type expression (Sphinx leaves the operators/brackets as plain
# text, not <em>). Group 1 is the dt + dd open up to the class list.
_RTYPE_DD_RE = re.compile(
    r'(<dt class="field-(?:odd|even)">Return type<span class="colon">:</span>'
    r'</dt>\s*<dd class="field-(?:odd|even))(">)'
)

# Category basename (matches the rst filename) → Sphinx-generated heading
# anchor on the class's index page.
_CATEGORY_ANCHORS = {
    "constructor":           "constructor",
    "io":                    "i-o",
    "properties":            "properties",
    "data_access":           "data-access",
    "manipulation":          "manipulation",
    "structural":            "structural",
    "analysis":              "analysis",
    "utility":               "utility",
    "dictionary_interface":  "dictionary-interface",
    "transformation":        "transformation",
}
_CAT_NAMES = "|".join(re.escape(k) for k in _CATEGORY_ANCHORS)
# Matches sidebar toctree-l1 category entries so their href can be
# rewritten to point at the anchor on the class's index page.
_SIDEBAR_CAT_RE = re.compile(
    r'(<li class="toctree-l1[^"]*"><a class="reference internal" href=")'
    r'((?:\.\./)*)'
    rf'({_CAT_NAMES})\.html"'
)

def _semantic_highlight(app, exception=None):
    """Post-process HTML to add semantic classes and API links."""
    if exception is not None:
        return
    outdir = Path(app.builder.outdir)
    for html_file in outdir.rglob("*.html"):
        text = html_file.read_text()
        original = text
        rel = html_file.relative_to(outdir)
        depth = len(rel.parts) - 1  # e.g. tutorials/foo.html → depth 1
        rel_str = str(rel).replace("\\", "/")

        # -- Sidebar customization (runs on every page) --
        # Rename "Section Navigation" to the active class/module name.
        new_title = None
        for prefix, label in _SIDEBAR_TITLE_MAP:
            if rel_str.startswith(prefix):
                new_title = label
                break
        if new_title:
            text = _SIDEBAR_TITLE_RE.sub(
                f'<p class="bd-links__title" role="heading" aria-level="1">{new_title}</p>',
                text,
            )
        # Strip "single_cell.<Class>." prefix from sidebar entry labels
        # (and anywhere else that shows the fully-qualified dotted name).
        text = _SIDEBAR_ENTRY_RE.sub(r'>\2<', text)

        # Strip the "Class." prefix from autosummary method-table entries
        # (SingleCell.find_doublets → find_doublets); href/title stay qualified.
        text = _AUTOSUMMARY_QUALIFIED_RE.sub(r'\1\2\3', text)

        # Tag the "Return type" field's <dd> so CSS can render the whole type
        # expression as code. Sphinx wraps only the known type names in links
        # and leaves the operators/brackets (` | tuple[ … ]`) as plain text, so
        # the class lets us colour the plain-text tokens too, not just the <em>s.
        text = _RTYPE_DD_RE.sub(r'\1 rtype-field\2', text)

        # Rewrite sidebar category links to jump to the anchored section on
        # the class index page (instead of loading a dedicated category
        # page).  e.g. href="constructor.html"  →  href="index.html#constructor"
        # On the class index page itself, use a bare `#anchor` so it scrolls in
        # place: the page is served at the clean URL /<class>/, so an
        # `index.html#...` href is a *different* document to the browser and
        # triggers a full cross-document load instead of an in-page scroll.
        is_class_index = rel_str in (
            "singlecell/index.html", "pseudobulk/index.html", "de/index.html")
        def _cat_repl(m):
            prefix, upward, cat = m.group(1), m.group(2), m.group(3)
            anchor = _CATEGORY_ANCHORS.get(cat, cat)
            if is_class_index:
                return f'{prefix}#{anchor}"'
            return f'{prefix}{upward}index.html#{anchor}"'
        text = _SIDEBAR_CAT_RE.sub(_cat_repl, text)

        # API pages: mark the body so CSS can hide the right "On this page"
        # secondary sidebar -- the left-hand sidebar is the sole navigation
        # there, with category entries jumping to the anchored section on
        # the class index page (rewritten above).
        if rel_str.startswith(("singlecell/", "pseudobulk/", "de/")):
            text = text.replace('<body ', '<body class="api-page" ', 1)
            # Sphinx prefixes staticmethod signatures with a C++-style
            # `static ` keyword. Replace it with Python's `@staticmethod`
            # decorator, rendered (via CSS) on its own line above the
            # signature.
            text = text.replace(
                '<span class="property"><span class="k">'
                '<span class="pre">static</span></span>'
                '<span class="w"> </span></span>',
                '<span class="staticmethod-decorator">'
                '<span class="pre">@staticmethod</span></span>',
            )
        # Tutorial content pages (not the index, which has no sections): mark
        # the body so CSS shows the "On this page" secondary sidebar.
        elif rel_str.startswith("tutorials/") and rel_str != "tutorials/index.html":
            text = text.replace('<body ', '<body class="tutorial-page" ', 1)
        # Installation: tag the body so CSS can drop its "previous: brisc"
        # prev-next link (the landing page isn't a step in the guide flow).
        elif rel_str == "installation.html":
            text = text.replace('<body ', '<body class="installation-page" ', 1)

        # Link inline <cite> references (single-backtick code in docstrings) to
        # their API pages where they name a method or class. Runs on every page;
        # only docstring-derived pages contain <cite> elements.
        current_class = None
        if rel_str.startswith("singlecell/"):
            current_class = "SingleCell"
        elif rel_str.startswith("pseudobulk/"):
            current_class = "Pseudobulk"
        elif rel_str.startswith("de/"):
            current_class = "DE"
        text = _link_inline_refs(text, depth, current_class)

        # Cross-reference links to an API object (the {class}/{meth} roles,
        # autosummary entries, signature/type links, viewcode "back" links) point
        # at the object's definition anchor (#brisc.X). On cross-page links the
        # fragment is redundant -- it's the target page's primary object -- and
        # makes for ugly URLs, so drop it: the link then lands at the page top
        # (its title). Only strip when a path precedes the "#"; same-page anchors
        # (the ¶ permalinks and the "On this page" TOC, href="#brisc.X") are kept.
        text = re.sub(
            r'(<a [^>]*href="[^"#]+)#brisc\.[\w.]*"', r'\1"', text)

        # -- Code-block semantic highlighting (only where pygments ran) --
        if '<span class="n">' not in text:
            if text != original:
                html_file.write_text(text)
            continue

        # function/method calls → green + API links (receiver/chain-aware, so
        # names shared across classes resolve to the right class page)
        text = _link_calls(text, depth)

        # keyword args: name= → orange
        text = _kwarg_re.sub(
            r'<span class="na">\1</span>\2', text)

        # Scanpy-style params: split "name (type) – desc" into two lines.
        # Handles both the form with a description (em-dash present) and
        # the form without (just name + type). A method with several params
        # renders each as `<li><p><strong>...`; a method with a *single* param
        # renders it as a bare `<dd class="field-odd"><p><strong>...` (no
        # <ul>/<li>). Match either opening and re-emit it (group 1) so single-
        # parameter methods get the same styled header + description.
        def _param_repl(m):
            opening, name, types, sep = m.groups()
            header = (
                f'{opening}<p class="param-header"><strong>{name}</strong>: '
                f'{types}</p>'
            )
            if '–' in sep:
                return header + '<p class="param-desc">'
            return header
        text = re.sub(
            r'(<li>|<dd class="field-(?:odd|even)">)'
            r'<p><strong>([^<]+)</strong>\s*'
            r'\(([^)]*(?:\([^)]*\))*[^)]*)\)'
            r'(\s*–\s*|</p>)',
            _param_repl,
            text)

        # Link bare-name cites that match a parameter on the page to that
        # parameter's description (id anchors are added here too).
        text = _link_param_refs(text)

        # Strip module qualifiers (and Sphinx's `~` short-name marker) from
        # rendered type tokens, keeping only the final component: `pl.DataFrame`
        # → `DataFrame`, `np.integer` → `integer`, and fully-qualified forms like
        # `~polars.expr.expr.Expr` → `Expr`, `~typing.Callable` → `Callable`,
        # `~brisc.pseudobulk.Pseudobulk` → `Pseudobulk`. This also lets the
        # type-alias collapses below match types autodoc rendered with the long
        # qualified names (e.g. the Pseudobulk.DE `group` union → PseudobulkColumn).
        text = re.sub(
            r'(<em>)~?(?:\w+\.)+(\w+)(</em>)', r'\1\2\3', text)
        # `Literal[False]` → `False` (any single-argument `Literal[...]`).
        text = re.sub(
            r'<em>Literal</em><em>\[</em><em>([^<]*)</em><em>\](\s*)</em>',
            r'<em>\1\2</em>', text)

        # Simplify types: remove np.integer/np.floating/np.bool_ duplicates
        # "int | integer" → "int", "float | floating" → "float"
        for numpy_t, python_t in [('integer', 'int'), ('floating', 'float'),
                                   ('bool_', 'bool')]:
            # remove "| integer" or "integer | " patterns
            text = re.sub(
                rf'<em>\s*\|\s*</em><em>\s*{numpy_t}\s*</em>', '', text)
            text = re.sub(
                rf'<em>\s*{numpy_t}\s*</em><em>\s*\|\s*</em>', '', text)

        # Collapse SingleCellColumn union to typedef name
        _scc_pattern = (
            r'<em>str</em><em> \| </em><em>Expr</em><em> \| </em>'
            r'<em>Series</em><em> \| </em><em>ndarray</em><em> \| </em>'
            r'<em>Callable</em><em>\[</em><em>\[</em>'
            r'.*?SingleCell.*?'
            r'<em>\]</em><em>,\s*</em><em>Series</em><em> \| </em>'
            r'<em>ndarray</em><em>\](\s*)</em>'
        )
        # Re-emit (\g<1>) exactly the whitespace autodoc placed inside the
        # union's closing `]` em -- a space when a separator follows
        # (`... | Sequence`), nothing when a bracket closes right after
        # (`Sequence[SingleCellColumn]`) -- rather than hardcoding a space,
        # which would render `SingleCellColumn ]`.
        _scc_link = (
            f'<a href="{"../" * depth}singlecell/typedefs.html'
            f'#singlecellcolumn" style="text-decoration:none">'
            f'<em>SingleCellColumn</em></a>' r'\g<1>'
        )
        text = re.sub(_scc_pattern, _scc_link, text)

        # Collapse the PseudobulkColumn union (the SingleCellColumn twin, whose
        # callable takes a Pseudobulk + cell type) to its typedef name.
        _pbc_pattern = (
            r'<em>str</em><em> \| </em><em>Expr</em><em> \| </em>'
            r'<em>Series</em><em> \| </em><em>ndarray</em><em> \| </em>'
            r'<em>Callable</em><em>\[</em><em>\[</em>'
            r'.*?Pseudobulk.*?'
            r'<em>\]</em><em>,\s*</em><em>Series</em><em> \| </em>'
            r'<em>ndarray</em><em>\](\s*)</em>'
        )
        _pbc_link = (  # \g<1>: preserve the closing `]` whitespace (see _scc_link)
            f'<a href="{"../" * depth}singlecell/typedefs.html'
            f'#pseudobulkcolumn" style="text-decoration:none">'
            f'<em>PseudobulkColumn</em></a>' r'\g<1>'
        )
        text = re.sub(_pbc_pattern, _pbc_link, text)

        # Collapse the Scalar-style union (str | int | float | Decimal | date
        # | time | datetime | timedelta | bool | bytes | Expr | Series | ...)
        # into the Scalar typedef link. Also handles the Iterable[...] variant.
        _scalar_inner = (
            r'<em>str</em><em> \| </em>'
            r'<em>int</em><em> \| </em>'
            r'<em>float</em><em> \| </em>'
            r'<em>Decimal</em><em> \| </em>'
            r'<em>date</em><em> \| </em>'
            r'<em>time</em><em> \| </em>'
            r'<em>datetime</em><em> \| </em>'
            r'<em>timedelta</em><em> \| </em>'
            r'<em>bool</em><em> \| </em>'
            r'<em>bytes</em><em> \| </em>'
            r'<em>Expr</em><em> \| </em>'
            r'<em>Series</em>'
        )
        _scalar_full = (
            _scalar_inner +
            r'(?:<em> \| </em><em>Iterable</em><em>\[</em>' +
            _scalar_inner +
            r'<em>\]</em>)?'
        )
        # No trailing space: _scalar_full ends at <em>Series</em> / the
        # Iterable `]` and consumes no separator whitespace, so the following
        # token (` | `, `]`, `,`) already carries the correct spacing. A
        # hardcoded space here would render `Scalar ]` inside a Sequence[...].
        _scalar_link = (
            f'<a href="{"../" * depth}singlecell/typedefs.html'
            f'#scalar" style="text-decoration:none">'
            f'<em>Scalar</em></a>'
        )
        text = re.sub(_scalar_full, _scalar_link, text)

        # Link bare type-alias names to the (orphan) typedefs page. Skip any
        # already wrapped above (the union collapses) — detected by a trailing
        # </a>.
        for _alias in ('SingleCellColumn', 'PseudobulkColumn', 'Scalar',
                       'UnsDict', 'UnsItem', 'Color', 'Indexer'):
            text = re.sub(
                rf'<em>{_alias}</em>(?!</a>)',
                f'<a href="{"../" * depth}singlecell/typedefs.html'
                f'#{_alias.lower()}" style="text-decoration:none">'
                f'<em>{_alias}</em></a>',
                text)

        if text != original:
            html_file.write_text(text)

# Keep the Tutorials and the API classes as LINEAR, self-contained prev/next
# sequences that do not wrap end-to-end. By default the global toctree chains
# everything together, so the last tutorial's "next" leaks into the API (Data
# manipulation → SingleCell) and each class index's "next" drops into its own
# Analysis/I-O subsections while its "prev" is pulled from the previous class's
# method stubs (or the tutorials). The Tutorials chain is otherwise correct, so
# we only drop the last tutorial's "next"; the class indexes get an explicit
# SingleCell → Pseudobulk → DE chain. None suppresses a link; no section wraps
# (the first has no cross-section "prev", the last has no "next").
_NAV_OVERRIDES = {
    "tutorials/data_manipulation": {"next": None},
    "singlecell/index":  {"prev": None,               "next": "pseudobulk/index"},
    "pseudobulk/index":  {"prev": "singlecell/index", "next": "de/index"},
    "de/index":          {"prev": "pseudobulk/index", "next": None},
}

def _fix_prev_next(app, pagename, templatename, context, doctree):
    spec = _NAV_OVERRIDES.get(pagename)
    if spec is None:
        return
    for key, doc in spec.items():
        if doc is None:
            context[key] = None
        else:
            context[key] = {"link": context["pathto"](doc),
                            "title": app.env.titles[doc].astext()}
    # Clear pydata's optional title overrides so our titles show.
    context["prev_title"] = context["next_title"] = ""

# -- SEO: canonical URLs, descriptions, social cards, sitemap ----------------
import html as _html
import json as _json

# Cloudflare Pages serves clean URLs: it strips `.html`, maps `index.html` to
# its directory, and 308-redirects the suffixed forms. Sphinx builds each
# page's canonical URL from the suffixed filename, so every canonical pointed
# at a redirect. Build them in the served form instead (also used for og:url
# and the sitemap).
def _clean_url(pagename):
    if pagename == "index":
        return html_baseurl
    if pagename.endswith("/index"):
        return html_baseurl + pagename[:-len("index")]
    return html_baseurl + pagename

# Pages search engines should skip: the search page and global indexes, the
# viewcode source dumps, and the per-category API pages (their autosummary
# tables duplicate the class index page, which the sidebar links to instead).
def _noindex(pagename):
    if pagename in ("search", "genindex", "py-modindex"):
        return True
    if pagename.startswith("_modules/"):
        return True
    parts = pagename.split("/")
    return (len(parts) == 2 and parts[0] in ("singlecell", "pseudobulk", "de")
            and parts[1] in _CATEGORY_ANCHORS)

# A hand-written `description` from the page's front matter (MyST) or
# top-of-file field list (RST), else the page's first paragraph (for the
# autosummary stubs, the method's docstring summary), cut to the ~160
# characters search engines display.
def _page_description(app, pagename, doctree):
    text = app.env.metadata.get(pagename, {}).get("description")
    if not text and doctree is not None:
        for node in doctree.findall(nodes.paragraph):
            if node.astext().strip():
                text = node.astext()
                break
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) > 160:
        text = text[:157].rsplit(" ", 1)[0] + "…"
    return text

_SITE_NAME = "brisc"
_HOME_TITLE = "brisc: fast, reproducible single-cell analysis in Python"
_SOCIAL_CARD = "_static/images/social_card.png"  # 1200 x 630

def _json_ld(description):
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "brisc",
        "description": description,
        "url": html_baseurl,
        "codeRepository": "https://github.com/briscverse/brisc",
        "programmingLanguage": "Python",
        "runtimePlatform": "Python 3.9+",
        "license": "https://opensource.org/license/mit",
        "keywords": ["single-cell", "scRNA-seq", "single-cell RNA-seq",
                     "bioinformatics", "AnnData", "Scanpy", "Seurat"],
        "author": [{"@type": "Person", "name": "Keon Arbabi"},
                   {"@type": "Person", "name": "Michael Wainberg"}],
        "sameAs": ["https://github.com/briscverse/brisc",
                   "https://pypi.org/project/brisc/",
                   "https://anaconda.org/conda-forge/brisc"],
        "citation": {
            "@type": "ScholarlyArticle",
            "@id": "https://doi.org/10.64898/2026.11.03.764120",
            "name": "brisc: a multithreaded CPU toolkit for single-cell "
                    "analysis",
            "author": [{"@type": "Person", "name": name} for name in (
                "Keon Arbabi", "Tain Luquez", "Jaiz Jeeson", "Mingyau Lam",
                "Hamed Heydari", "Gary D. Bader", "Vilas Menon",
                "Michael Wainberg")],
            "datePublished": "2026-11-04",
            "publisher": {"@type": "Organization", "name": "bioRxiv"},
        },
    }
    return ('<script type="application/ld+json">'
            + _json.dumps(data).replace("</", "<\\/") + "</script>")

def _seo_page_context(app, pagename, templatename, context, doctree):
    url = _clean_url(pagename)
    context["pageurl"] = url
    if pagename == "index":
        # The landing page's <title> was "brisc — brisc documentation"; make
        # it say what brisc is (this is the text of the search result link).
        # The H1 and everything in the body are untouched; the " — brisc
        # documentation" suffix is dropped in _fix_home_title.
        context["title"] = _HOME_TITLE
    tags = []
    if _noindex(pagename):
        tags.append('<meta name="robots" content="noindex" />')
    else:
        desc = _page_description(app, pagename, doctree)
        if pagename == "index":
            og_title = _HOME_TITLE
        else:
            title = (app.env.titles[pagename].astext()
                     if pagename in app.env.titles else context.get("title", ""))
            og_title = f"{title.removeprefix('brisc.')} — {_SITE_NAME}"
        if desc:
            d = _html.escape(desc, quote=True)
            tags += [f'<meta name="description" content="{d}" />',
                     f'<meta property="og:description" content="{d}" />']
        tags += [
            f'<meta property="og:title" '
            f'content="{_html.escape(og_title, quote=True)}" />',
            '<meta property="og:type" content="website" />',
            f'<meta property="og:url" content="{url}" />',
            f'<meta property="og:site_name" content="{_SITE_NAME}" />',
            f'<meta property="og:image" content="{html_baseurl}{_SOCIAL_CARD}" />',
            '<meta property="og:image:width" content="1200" />',
            '<meta property="og:image:height" content="630" />',
            '<meta property="og:image:alt" content="brisc" />',
            '<meta name="twitter:card" content="summary_large_image" />',
        ]
        if pagename == "index":
            tags.append(_json_ld(desc))
    context["metatags"] = context.get("metatags", "") + "".join(
        f"    {tag}\n" for tag in tags)

# Rewrite <title> tags only; nothing in any page body changes.
#  - homepage: drop the " — brisc" suffix, leaving _HOME_TITLE.
#  - API member pages: "brisc.SingleCell.pca — brisc" -> "SingleCell.pca — brisc",
#    matching the signature shown on the page. Google was rewriting the
#    mismatched titles into things like "SingleCell. - pca".
_API_TITLE_RE = re.compile(r"<title>brisc\.((?:SingleCell|Pseudobulk|DE)\.[\w.]+)")

def _fix_titles(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    outdir = Path(app.builder.outdir)
    index = outdir / "index.html"
    text = index.read_text()
    suffixed = f"<title>{_HOME_TITLE} &#8212; {html_title}</title>"
    if suffixed in text:
        index.write_text(text.replace(suffixed, f"<title>{_HOME_TITLE}</title>"))
    for html_file in outdir.rglob("*.html"):
        rel = html_file.relative_to(outdir).as_posix()
        if not rel.startswith(("singlecell/", "pseudobulk/", "de/")):
            continue
        text = html_file.read_text()
        new = _API_TITLE_RE.sub(r"<title>\1", text, count=1)
        if new != text:
            html_file.write_text(new)

# sitemap.xml at the site root, listing every indexable page by its clean URL.
def _write_sitemap(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    urls = sorted(_clean_url(d) for d in app.env.found_docs if not _noindex(d))
    Path(app.builder.outdir, "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{_html.escape(u)}</loc></url>\n" for u in urls)
        + "</urlset>\n")

def setup(app):
    app.add_directive("classdescription", _ClassDescription)
    app.add_role("codelink", _codelink_role)
    app.connect("autodoc-process-docstring", _constructor_dropdowns, priority=400)
    app.connect("autodoc-process-docstring", _md_to_rst_links)
    app.connect("autodoc-process-docstring", _drop_constructor_summary)
    app.connect("html-page-context", _fix_prev_next)
    app.connect("html-page-context", _seo_page_context)
    app.connect("build-finished", _semantic_highlight, priority=901)
    app.connect("build-finished", _write_sitemap)
    app.connect("build-finished", _fix_titles)
