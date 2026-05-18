.PHONY: all run extract equations video plates validate pptx

all: run extract equations video plates validate

run:
	python scripts/00_run_notebook.py

extract:
	python scripts/01_extract_notebook_outputs.py

equations:
	python scripts/02_render_equations.py

video:
	python scripts/03_convert_gifs_to_mp4.py

plates:
	python scripts/04_build_slide_plates.py

validate:
	python scripts/05_validate_assets.py

pptx:
	python scripts/06_make_pptx_skeleton.py
