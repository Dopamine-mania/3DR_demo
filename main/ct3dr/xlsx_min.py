from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import zipfile


_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _col_name(idx0: int) -> str:
    n = idx0 + 1
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def _cell_ref(r1: int, c1: int) -> str:
    return f"{_col_name(c1 - 1)}{r1}"


def _xml_cell(r1: int, c1: int, v) -> str:
    ref = _cell_ref(r1, c1)
    if v is None:
        return f'<c r="{ref}"/>'
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f'<c r="{ref}"><v>{v}</v></c>'
    text = html.escape(str(v))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_xlsx_simple(path: str | Path, sheet_name: str, headers: list[str], rows: list[list[object]]) -> None:
    """
    Minimal .xlsx writer (single sheet) with inline strings and numbers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>
"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""

    workbook = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{html.escape(sheet_name)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""

    wb_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""

    # Build sheet XML.
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "  <sheetData>",
    ]
    r = 1
    lines.append(f'    <row r="{r}">')
    for c, h in enumerate(headers, start=1):
        lines.append("      " + _xml_cell(r, c, h))
    lines.append("    </row>")
    for row in rows:
        r += 1
        lines.append(f'    <row r="{r}">')
        for c, v in enumerate(row, start=1):
            lines.append("      " + _xml_cell(r, c, v))
        lines.append("    </row>")
    lines.extend(["  </sheetData>", "</worksheet>"])
    sheet1 = "\n".join(lines) + "\n"

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet1)

