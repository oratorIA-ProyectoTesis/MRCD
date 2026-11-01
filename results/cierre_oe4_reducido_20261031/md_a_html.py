"""Minimal Markdown -> printable HTML (no external libraries), same approach as the OE1 sheet.

Usage: python -I md_a_html.py APROBACION_ASESOR_OE4.md APROBACION_ASESOR_OE4.html "Title"
"""

import html
import re
import sys
from pathlib import Path


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def to_html(md):
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if (
            l.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[\s:|-]+\|$", lines[i + 1])
        ):
            head = [c.strip() for c in l.strip("|").split("|")]
            out.append(
                "<table><thead><tr>"
                + "".join("<th>%s</th>" % inline(c) for c in head)
                + "</tr></thead><tbody>"
            )
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append(
                    "<tr>" + "".join("<td>%s</td>" % inline(c) for c in cells) + "</tr>"
                )
                i += 1
            out.append("</tbody></table>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", l)
        if m:
            n = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (n, inline(m.group(2)), n))
        elif l.startswith("> "):
            out.append("<blockquote>%s</blockquote>" % inline(l[2:]))
        elif l.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append("<li>%s</li>" % inline(lines[i][2:]))
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif l.strip() == "---":
            out.append("<hr>")
        elif l.strip():
            out.append("<p>%s</p>" % inline(l))
        i += 1
    return "\n".join(out)


CSS = """
body{font-family:Georgia,'Times New Roman',serif;font-size:10.5pt;line-height:1.35;margin:1.6cm;color:#111;background:#fff}
h1{font-size:17pt}h2{font-size:13pt;border-bottom:1px solid #444;padding-bottom:2px;margin-top:1.2em}h3{font-size:11pt}
table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:8.5pt}th,td{border:1px solid #666;padding:3px 5px;vertical-align:top;word-break:break-word}
th{background:#eee}code{font-family:Consolas,monospace;font-size:8pt;word-break:break-all}
blockquote{border-left:3px solid #999;margin:.5em 0;padding-left:.7em;color:#333}
.pagebreak{page-break-before:always}
@media print{body{margin:0}@page{size:A4;margin:1.5cm}tr{page-break-inside:avoid}h2,h3{page-break-after:avoid}}
"""

if __name__ == "__main__":
    src, dst, title = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    body = to_html(src.read_text(encoding="utf8"))
    body = body.replace(
        "<h2>7. Bloque de firma (LISTO PARA FIRMAR)</h2>",
        '<h2 class="pagebreak">7. Bloque de firma (LISTO PARA FIRMAR)</h2>',
    )
    dst.write_text(
        '<!doctype html><html lang="es"><head><meta charset="utf-8"><title>%s</title><style>%s</style></head><body>%s</body></html>'
        % (html.escape(title), CSS, body),
        encoding="utf8",
    )
    print("ok", dst)
