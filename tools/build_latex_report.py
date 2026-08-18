"""Generate the editable standalone LaTeX report from REPORT.md."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REPORT.md"
OUT_DIR = ROOT.parent / "output" / "latex"
OUT_TEX = OUT_DIR / "NBQSS_2026_Tomography_Report.tex"


PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[letterpaper,margin=1in,headheight=15pt]{geometry}
\usepackage{fontspec}
\setmainfont[
  Path=/System/Library/Fonts/Supplemental/,
  UprightFont=Arial.ttf,
  BoldFont=Arial Bold.ttf,
  ItalicFont=Arial Italic.ttf,
  BoldItalicFont=Arial Bold Italic.ttf
]{Arial}
\setsansfont[
  Path=/System/Library/Fonts/Supplemental/,
  UprightFont=Arial.ttf,
  BoldFont=Arial Bold.ttf,
  ItalicFont=Arial Italic.ttf,
  BoldItalicFont=Arial Bold Italic.ttf
]{Arial}
\setmonofont[Path=/System/Library/Fonts/]{Menlo.ttc}
\usepackage{microtype}
\usepackage{amsmath,amssymb,mathtools}
\usepackage[table]{xcolor}
\usepackage{booktabs,array,tabularx,xltabular}
\usepackage{ragged2e}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{tcolorbox}
\usepackage{graphicx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage{caption}
\usepackage{float}
\usepackage{fvextra}
\usepackage{needspace}
\usepackage{hyperref}

\definecolor{Navy}{HTML}{14344E}
\definecolor{Teal}{HTML}{0F7173}
\definecolor{Gold}{HTML}{B38120}
\definecolor{Ink}{HTML}{20262D}
\definecolor{Muted}{HTML}{5D6974}
\definecolor{Pale}{HTML}{F5F7F9}
\definecolor{Light}{HTML}{EAF0F4}
\definecolor{Rule}{HTML}{CED7DE}
\definecolor{Steel}{HTML}{4F718C}

\hypersetup{
  colorlinks=true,
  linkcolor=Navy,
  urlcolor=Teal,
  citecolor=Teal,
  pdftitle={Building a Hardware-Agnostic Quantum State Tomography and Denoising Suite},
  pdfsubject={NBQSS 2026 tomography challenge technical report},
  pdfauthor={NBQST project team},
  pdfkeywords={quantum state tomography, denoising, Array API, MLE, low rank}
}
\urlstyle{same}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.60em}
\setlength{\emergencystretch}{3em}
\setlist[itemize]{leftmargin=1.5em,itemsep=0.25em,topsep=0.25em}
\setlist[enumerate]{leftmargin=1.8em,itemsep=0.35em,topsep=0.35em}
\renewcommand{\arraystretch}{1.22}
\setlength{\tabcolsep}{4pt}
\renewcommand{\contentsname}{Contents}
\setcounter{tocdepth}{2}

\titleformat{\section}
  {\needspace{5\baselineskip}\Large\bfseries\color{Navy}}
  {\thesection.}{0.55em}{}
\titlespacing*{\section}{0pt}{1.8em}{0.65em}
\titleformat{\subsection}
  {\needspace{4\baselineskip}\large\bfseries\color{Teal}}
  {\thesubsection}{0.5em}{}
\titlespacing*{\subsection}{0pt}{1.35em}{0.45em}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\small\color{Muted}NBQSS 2026 \enspace|\enspace Hardware-Agnostic QST}
\fancyfoot[R]{\small\color{Muted}\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\newcolumntype{Y}{>{\RaggedRight\arraybackslash}X}
\newcommand{\tableheader}[1]{\textbf{\color{Navy}#1}}
\captionsetup{font=small,labelfont={bf,color=Navy},textfont={it,color=Muted}}

\begin{document}
\begin{titlepage}
\thispagestyle{empty}
\centering
\vspace*{1.0in}
{\small\bfseries\color{Gold}NBQSS 2026 \;|\; TECHNICAL REPORT\par}
\vspace{0.35in}
{\fontsize{28}{34}\selectfont\bfseries\color{Navy}
Building a Hardware-Agnostic\\[0.12em]
Quantum State Tomography\\[0.12em]
and Denoising Suite\par}
\vspace{0.35in}
{\large\color{Teal}Methods, implementation, benchmarks, and a scalable roadmap beyond attention\par}
\vfill
\begin{tcolorbox}[
  width=0.88\textwidth,
  colback=Light,
  colframe=Rule,
  boxrule=0.7pt,
  arc=1.5mm,
  left=5mm,right=5mm,top=4mm,bottom=4mm]
\centering
\textbf{\color{Navy}Core recommendation}\\[0.3em]
Start with physical likelihood and validated structure. Use attention only when repeatable device noise justifies learned global corrections.
\end{tcolorbox}
\vfill
{\small\color{Muted}Version 1.1 \;|\; 18 August 2026\\[0.25em]
Reference implementation: \texttt{nbqst 0.1.0}\par}
\end{titlepage}

\pagenumbering{roman}
\tableofcontents
\clearpage
\pagenumbering{arabic}
"""


POSTAMBLE = r"""
\end{document}
"""


QUALITY_CHART = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.92\linewidth,
  height=8.2cm,
  xbar,
  xmin=0,xmax=0.56,
  xlabel={Mean Hilbert--Schmidt distance (lower is better)},
  ytick={0,1,2,3,4,5,6,7,8},
  yticklabels={Mixed / MLE,Mixed / Rank-1,Mixed / PSD,Product / MLE,Product / Rank-1,Product / PSD,Haar / MLE,Haar / Rank-1,Haar / PSD},
  yticklabel style={font=\scriptsize},
  xticklabel style={font=\scriptsize},
  bar width=6pt,
  axis line style={draw=Rule},
  tick style={draw=Rule},
  xmajorgrids=true,
  grid style={draw=Rule!65},
  legend style={at={(0.5,-0.19)},anchor=north,legend columns=3,draw=none,font=\scriptsize},
]
\addplot[fill=Teal,draw=Teal] coordinates {
  (0.0680,0) (0.0375,3) (0.0416,6)};
\addlegendentry{MLE}
\addplot[fill=Gold,draw=Gold] coordinates {
  (0.5191,1) (0.0548,4) (0.0535,7)};
\addlegendentry{Rank-1 projection}
\addplot[fill=Steel,draw=Steel] coordinates {
  (0.0836,2) (0.0616,5) (0.0593,8)};
\addlegendentry{PSD projection}
\end{axis}
\end{tikzpicture}
\caption{Two-qubit physical-estimator error at 500 shots per setting. Rank one is appropriate for pure targets and deliberately misspecified for full-rank mixed targets.}
\label{fig:quality}
\end{figure}
"""


RUNTIME_CHART = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.88\linewidth,
  height=7.0cm,
  ybar,
  ymode=log,
  ymin=0.08,ymax=120,
  ylabel={Mean reconstruction time (ms, log scale)},
  symbolic x coords={Linear,PSD,Rank-1,Shrinkage,MLE},
  xtick=data,
  xticklabel style={font=\small},
  point meta=explicit symbolic,
  nodes near coords,
  nodes near coords style={font=\scriptsize},
  bar width=20pt,
  enlarge x limits=0.13,
  axis line style={draw=Rule},
  tick style={draw=Rule},
  ymajorgrids=true,
  grid style={draw=Rule!65},
]
\addplot[fill=Teal,draw=Teal] coordinates {
  (Linear,0.331) [0.331]
  (PSD,0.136) [0.136]
  (Rank-1,0.121) [0.121]
  (Shrinkage,0.186) [0.186]
  (MLE,62.0) [62.0]};
\end{axis}
\end{tikzpicture}
\caption{Eager NumPy reconstruction time averaged across two-qubit benchmark conditions.}
\label{fig:runtime}
\end{figure}
"""


DISPLAY_MATH = {
    "p_s = diag(U_s^dagger rho U_s).": r"\[\mathbf{p}_s=\operatorname{diag}(U_s^\dagger\rho U_s).\]",
    "c_s ~ Multinomial(N, p_s).": r"\[\mathbf{c}_s\sim\operatorname{Multinomial}(N,\mathbf{p}_s).\]",
    "rho_LI = (1 / 2^n) sum over P in {I,X,Y,Z}^n of <P> P.": r"\[\rho_{\mathrm{LI}}=\frac{1}{2^n}\sum_{P\in\{I,X,Y,Z\}^{\otimes n}}\langle P\rangle P.\]",
    "L(rho) = - sum over settings s and outcomes b of c_(s,b) log p_(s,b)(rho).": r"\[\mathcal{L}(\rho)=-\sum_{s,b}c_{s,b}\log p_{s,b}(\rho).\]",
    "rho(T) = T^dagger T / Tr(T^dagger T)": r"\[\rho(T)=\frac{T^\dagger T}{\operatorname{Tr}(T^\dagger T)}\]",
    "rho_NN = L L^dagger / Tr(L L^dagger).": r"\[\rho_{\mathrm{NN}}=\frac{LL^\dagger}{\operatorname{Tr}(LL^\dagger)}.\]",
}


def escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def protect_urls(text: str):
    tokens = {}

    def replace(match):
        url = match.group(0)
        trailing = ""
        while url and url[-1] in ".,;":
            trailing = url[-1] + trailing
            url = url[:-1]
        key = f"@@URL{len(tokens)}@@"
        tokens[key] = rf"\url{{{url}}}{trailing}"
        return key

    return re.sub(r"https?://[^\s]+", replace, text), tokens


def protect_math(text: str):
    tokens = {}
    patterns = [
        (r"rho = T\^dagger T / Tr\(T\^dagger T\)", r"\(\rho=T^\dagger T/\operatorname{Tr}(T^\dagger T)\)"),
        (r"rho = T\^dagger T", r"\(\rho=T^\dagger T\)"),
        (r"rho_alpha = alpha rho_hat \+ \(1-alpha\) I/d", r"\(\rho_\alpha=\alpha\widehat\rho+(1-\alpha)I/d\)"),
        (r"C C\^dagger", r"\(CC^\dagger\)"),
        (r"d = 2\^n", r"\(d=2^n\)"),
        (r"\{X,Y,Z\}\^n", r"\(\{X,Y,Z\}^{\otimes n}\)"),
        (r"\{I,X,Y,Z\}\^n", r"\(\{I,X,Y,Z\}^{\otimes n}\)"),
        (r"(?<![A-Za-z0-9])([XYZ])\^n", None),
        (r"(?<![A-Za-z0-9])([0-9]+)\^([0-9n])", None),
        (r"(?<![A-Za-z0-9])([234])\^n", None),
        (r"O\(d\^3\)", r"\(O(d^3)\)"),
        (r"O\(d\^2\)", r"\(O(d^2)\)"),
        (r"O\(r d\^2\)", r"\(O(rd^2)\)"),
        (r"O\(r d\)", r"\(O(rd)\)"),
        (r"<X tensor Y>", r"\(\langle X\otimes Y\rangle\)"),
        (r"<P>", r"\(\langle P\rangle\)"),
    ]
    for pattern, replacement in patterns:
        def repl(match, replacement=replacement):
            if replacement:
                value = replacement
            elif len(match.groups()) == 2:
                value = rf"\({match.group(1)}^{{{match.group(2)}}}\)"
            elif match.group(1) in "XYZ":
                value = rf"\({match.group(1)}^{{\otimes n}}\)"
            else:
                value = rf"\({match.group(1)}^n\)"
            key = f"@@MATH{len(tokens)}@@"
            tokens[key] = value
            return key

        text = re.sub(pattern, repl, text)
    return text, tokens


def inline_latex(text: str) -> str:
    pieces = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text)
    result = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith("`"):
            result.append(r"\texttt{" + escape_plain(piece[1:-1]) + "}")
            continue
        bold = piece.startswith("**")
        if bold:
            piece = piece[2:-2]
        piece, url_tokens = protect_urls(piece)
        piece, math_tokens = protect_math(piece)
        escaped = escape_plain(piece)
        for key, value in {**url_tokens, **math_tokens}.items():
            escaped = escaped.replace(escape_plain(key), value)
        if bold:
            escaped = r"\textbf{" + escaped + "}"
        result.append(escaped)
    return "".join(result)


def table_spec(column_count: int) -> str:
    specs = {
        2: r">{\RaggedRight\arraybackslash}p{0.22\textwidth}Y",
        3: r">{\RaggedRight\arraybackslash}p{0.16\textwidth}YY",
        4: r">{\RaggedRight\arraybackslash}p{0.16\textwidth}>{\RaggedRight\arraybackslash}p{0.13\textwidth}YY",
        5: r">{\RaggedRight\arraybackslash}p{0.15\textwidth}>{\RaggedRight\arraybackslash}p{0.12\textwidth}Y>{\RaggedRight\arraybackslash}p{0.13\textwidth}>{\RaggedRight\arraybackslash}p{0.13\textwidth}",
        6: r">{\RaggedRight\arraybackslash}p{0.12\textwidth}>{\RaggedRight\arraybackslash}p{0.11\textwidth}>{\RaggedRight\arraybackslash}p{0.13\textwidth}>{\RaggedRight\arraybackslash}p{0.09\textwidth}YY",
    }
    return specs.get(column_count, "Y" * column_count)


def latex_table(rows: list[list[str]]) -> str:
    def table_cell(cell: str) -> str:
        converted = inline_latex(cell)
        converted = converted.replace("-", r"-\allowbreak{}")
        converted = converted.replace("/", r"/\allowbreak{}")
        return converted.replace("Substantial", r"Sub\-stantial")

    headers = rows[0]
    body = rows[1:]
    columns = len(headers)
    size = r"\scriptsize" if columns >= 5 else r"\footnotesize"
    header = " & ".join(r"\tableheader{" + table_cell(cell) + "}" for cell in headers) + r" \\"
    lines = [
        r"\begingroup",
        size,
        r"\sloppy",
        r"\rowcolors{2}{white}{Pale}",
        rf"\begin{{xltabular}}{{\textwidth}}{{{table_spec(columns)}}}",
        r"\toprule",
        r"\rowcolor{Light}",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\rowcolor{Light}",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in body:
        lines.append(" & ".join(table_cell(cell) for cell in row) + r" \\")
    lines.extend([r"\end{xltabular}", r"\endgroup"])
    return "\n".join(lines)


def heading_title(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text).strip()


def convert() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "## Abstract")
    output = [PREAMBLE]
    list_mode = None
    in_code = False

    def close_list():
        nonlocal list_mode
        if list_mode:
            output.append(rf"\end{{{list_mode}}}")
            list_mode = None

    i = start
    while i < len(lines):
        raw = lines[i]
        text = raw.strip()
        if text.startswith("```"):
            close_list()
            if in_code:
                output.append(r"\end{Verbatim}")
                in_code = False
            else:
                output.append(r"\begin{Verbatim}[fontsize=\small,frame=single,breaklines=true,breakanywhere=true,rulecolor=\color{Rule}]")
                in_code = True
            i += 1
            continue
        if in_code:
            output.append(raw)
            i += 1
            continue
        if not text:
            close_list()
            output.append("")
            i += 1
            continue
        if text.startswith("|"):
            close_list()
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
                i += 1
            output.append(latex_table([table_rows[0]] + table_rows[2:]))
            continue
        if text.startswith("### "):
            close_list()
            raw_title = text[4:]
            title = heading_title(raw_title)
            output.append(r"\subsection{" + inline_latex(title) + "}")
            if raw_title.startswith("9.1"):
                output.append(QUALITY_CHART)
            elif raw_title.startswith("9.3"):
                output.append(RUNTIME_CHART)
            i += 1
            continue
        if text.startswith("## "):
            close_list()
            raw_title = text[3:]
            title = heading_title(raw_title)
            if raw_title == "Abstract":
                output.append(r"\section*{Abstract}\addcontentsline{toc}{section}{Abstract}")
            else:
                output.append(r"\section{" + inline_latex(title) + "}")
            i += 1
            continue
        numbered = re.match(r"^\d+\.\s+(.*)$", text)
        if numbered:
            if list_mode != "enumerate":
                close_list()
                output.append(r"\begin{enumerate}")
                list_mode = "enumerate"
            output.append(r"\item " + inline_latex(numbered.group(1)))
            i += 1
            continue
        if text.startswith("- "):
            if list_mode != "itemize":
                close_list()
                output.append(r"\begin{itemize}")
                list_mode = "itemize"
            output.append(r"\item " + inline_latex(text[2:]))
            i += 1
            continue
        close_list()
        if text in DISPLAY_MATH:
            output.append(DISPLAY_MATH[text])
        else:
            output.append(inline_latex(text) + "\n")
        i += 1
    close_list()
    output.append(POSTAMBLE)
    return "\n".join(output)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(convert(), encoding="utf-8")
    print(OUT_TEX)


if __name__ == "__main__":
    main()
