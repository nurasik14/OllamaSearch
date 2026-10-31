#!/usr/bin/env python3
"""Build the decorative README images in docs/img/readme/ (all original artwork).

    python3 scripts/make_readme_art.py

Writes SVG sources and renders PNGs with `rsvg-convert` (librsvg), so the
fonts (JetBrains Mono, Noto Sans) only need to exist on the machine that runs
this script, not on the reader's.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img" / "readme"

THEMES = {
    "light": {
        "bg": "#fbfaf7",
        "card": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "line": "#d8dee4",
        "wool": "#fbf7ef",
        "chip": "#f1efe9",
    },
    "dark": {
        "bg": "#10141a",
        "card": "#161b22",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "line": "#30363d",
        "wool": "#2a2f37",
        "chip": "#21262d",
    },
}
AMBER = "#e8a33d"
AMBER_DEEP = "#c98a2b"
TEAL = "#2bb3a3"
MONO = "JetBrains Mono, Fira Code, monospace"
SANS = "Noto Sans, DejaVu Sans, sans-serif"


def llama(ink: str, wool: str, x: float = 0, y: float = 0, scale: float = 1.0) -> str:
    """Original side-view llama inspecting things through a magnifying glass (viewBox 240x206)."""
    return f"""
<g transform="translate({x} {y}) scale({scale})" fill="none" stroke="{ink}" stroke-width="5"
   stroke-linecap="round" stroke-linejoin="round">
  <path d="M40 118 C30 110 32 98 44 96 C48 86 62 86 66 92 C72 84 86 84 90 92 C96 84 110 84 114 92
           C118 80 128 76 134 78 L136 40 C136 32 142 28 150 28 L162 28 C170 28 176 34 176 42 L176 46
           C176 52 172 56 166 56 L158 56 L156 110 C164 116 162 130 152 132 C148 140 136 142 130 136
           C122 144 108 144 102 136 C94 144 80 144 74 136 C66 144 52 142 50 134 C40 134 34 126 40 118 Z"
        fill="{wool}"/>
  <path d="M144 30 C140 18 142 10 148 8 C152 14 154 22 152 29"/>
  <path d="M40 106 C30 102 26 110 30 116"/>
  <path d="M58 138 L58 182 M80 140 L80 184 M122 140 L122 184 M146 136 L146 182"/>
  <path d="M52 184 L60 184 M74 186 L82 186 M116 186 L124 186 M140 184 L148 184" stroke-width="6"/>
  <circle cx="160" cy="39" r="3.2" fill="{ink}" stroke="none"/>
  <path d="M166 51 C169 52 172 51 174 49" stroke-width="3.5"/>
  <path d="M206 78 L226 102" stroke="{AMBER_DEEP}" stroke-width="9"/>
  <circle cx="194" cy="62" r="21" stroke="{AMBER}" stroke-width="6" fill="{AMBER}" fill-opacity="0.14"/>
  <path d="M182 56 C184 50 189 47 195 46" stroke="{AMBER}" stroke-width="3" opacity="0.75"/>
</g>"""


def banner(t: dict[str, str]) -> str:
    w, h = 1600, 420
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" rx="28" fill="{t["bg"]}"/>
  <rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="28" fill="none" stroke="{t["line"]}" stroke-width="2"/>
  {llama(t["ink"], t["wool"], x=96, y=88, scale=1.42)}
  <text x="560" y="168" font-family="{MONO}" font-weight="800" font-size="96" fill="{t["ink"]}">Ollama<tspan fill="{AMBER}">Search</tspan></text>
  <text x="564" y="222" font-family="{SANS}" font-size="31" fill="{t["muted"]}">Private web search for local Ollama models,</text>
  <text x="564" y="262" font-family="{SANS}" font-size="31" fill="{t["muted"]}">and a benchmark that checks whether they really use it.</text>
  <!-- mock query bar -->
  <rect x="564" y="300" width="900" height="64" rx="32" fill="{t["card"]}" stroke="{t["line"]}" stroke-width="2"/>
  <circle cx="604" cy="331" r="11" fill="none" stroke="{AMBER}" stroke-width="4"/>
  <path d="M612 339 L622 349" stroke="{AMBER}" stroke-width="4" stroke-linecap="round"/>
  <text x="640" y="342" font-family="{MONO}" font-size="27" fill="{t["ink"]}">who built the Kalvenna Bridge?</text>
  <rect x="1180" y="314" width="132" height="36" rx="18" fill="{t["chip"]}"/>
  <text x="1246" y="339" text-anchor="middle" font-family="{MONO}" font-size="20" fill="{TEAL}">search ×2</text>
  <rect x="1322" y="314" width="126" height="36" rx="18" fill="{t["chip"]}"/>
  <text x="1385" y="339" text-anchor="middle" font-family="{MONO}" font-size="20" fill="{TEAL}">cited [1]</text>
</svg>"""


def corner_llama(t: dict[str, str]) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="240" height="206" viewBox="0 0 240 206">
  {llama(t["ink"], t["wool"])}
</svg>"""


def sticky(lines: list[str], tint: str, rotate: float, ink: str = "#1f2328") -> str:
    """A small rotated sticky note; same colours in both themes (it is 'paper')."""
    w, h = 340, 210
    body = "".join(
        f'<text x="26" y="{68 + i * 44}" font-family="{MONO}" font-size="{30 if i == 0 else 23}" '
        f'font-weight="{800 if i == 0 else 500}" fill="{ink}">{line}</text>'
        for i, line in enumerate(lines)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w + 40}" height="{h + 40}" viewBox="0 0 {w + 40} {h + 40}">
  <g transform="rotate({rotate} {(w + 40) / 2} {(h + 40) / 2}) translate(20 20)">
    <rect x="4" y="6" width="{w}" height="{h}" rx="6" fill="#000" opacity="0.18"/>
    <rect width="{w}" height="{h}" rx="6" fill="{tint}"/>
    <rect x="{w / 2 - 50}" y="-10" width="100" height="26" rx="3" fill="#ffffff" opacity="0.55"/>
    {body}
  </g>
</svg>"""


def render(svg: str, name: str, width: int) -> None:
    src = OUT / f"{name}.svg"
    src.write_text(svg, encoding="utf-8")
    png = OUT / f"{name}.png"
    subprocess.run(["rsvg-convert", "-w", str(width), str(src), "-o", str(png)], check=True)
    src.unlink()  # keep only the PNG in the repo; this script is the source
    print(f"{png.relative_to(ROOT)}  {png.stat().st_size // 1024} KB")


def main() -> None:
    if not shutil.which("rsvg-convert"):
        raise SystemExit("rsvg-convert (librsvg) is required")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        render(banner(t), f"banner-{name}", 1600)
        render(corner_llama(t), f"llama-{name}", 240)
    render(
        sticky(["closed-book: 0%", "the llama was", "guessing. confidently."], "#ffe58a", -4),
        "note-closed-book",
        380,
    )
    render(
        sticky(["works on 6 GB", "of VRAM (and", "a lot of patience)"], "#bfe8df", 3),
        "note-vram",
        380,
    )


if __name__ == "__main__":
    main()
