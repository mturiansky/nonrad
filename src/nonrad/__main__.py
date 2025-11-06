import cyclopts

from nonrad.cli import generate_ccd, process, setup_elph

app = cyclopts.App(help_format="md")

for func in (generate_ccd, setup_elph, process):
    app.command()(func)

if __name__ == "__main__":
    app()
