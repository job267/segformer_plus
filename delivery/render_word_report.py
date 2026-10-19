"""Use the canonical DOCX rasterizer with a PDF exported by native Microsoft Word."""

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--pdf",type=Path,required=True)
    parser.add_argument("--renderer",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not args.pdf.is_file() or args.pdf.stat().st_size==0:raise FileNotFoundError(args.pdf)
    temporary=args.output.parent/"tmp";temporary.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temporary.resolve())
    spec=importlib.util.spec_from_file_location("canonical_docx_renderer",args.renderer)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    def native_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose):
        if Path(doc_path).resolve()!=args.input.resolve():raise ValueError("Unexpected document path")
        destination=Path(convert_tmp_dir)/(stem+".pdf");shutil.copyfile(args.pdf,destination)
        return str(destination),"Microsoft Word ExportAsFixedFormat used; no LibreOffice binary is bundled on this Windows host."
    renderer.convert_to_pdf=native_pdf
    images=renderer.rasterize(str(args.input),str(args.output),160,False,False)
    print(f"Rendered {len(images)} pages with the canonical render_docx.py rasterizer and native Word PDF export.")


if __name__=="__main__":main()
