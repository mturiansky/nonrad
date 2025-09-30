import cyclopts

from nonrad.cli import generate_ccd, process_ccd

app = cyclopts.App(help_format="md")

for func in (generate_ccd, process_ccd):
    app.command()(func)

if __name__ == "__main__":
    app()
