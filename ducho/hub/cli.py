import click
from browser import search_models, DUCHO_TASK_MAP

@click.command()
@click.option("--modality", "-m", required=True,
              type=click.Choice(list(DUCHO_TASK_MAP.keys())),
              help="Ducho modality to search for")
@click.option("--query", "-q", default=None, help="Search keyword (e.g. 'clip', 'bert')")
@click.option("--library", "-l", default=None, help="Backend library (transformers, sentence-transformers)")
@click.option("--limit", default=10, show_default=True)
@click.option("--sort", default="downloads", type=click.Choice(["downloads","likes","lastModified"]))
@click.option("--show-config", is_flag=True, help="Print Ducho YAML snippet for top result")
def search(modality, query, library, limit, sort, show_config):
    """Search HuggingFace Hub for models compatible with a Ducho modality."""
    results = search_models(modality, query, library, limit, sort)
    
    if not results:
        click.echo("No models found. Try a different query or modality.")
        return
    
    for r in results:
        click.echo(f"{r['model_id']:60s}  ⬇ {r['downloads']:>10,}  ♥ {r['likes']:>5}  [{r['task']}]")
        if show_config:
            click.echo(r['ducho_config_snippet'])
    click.echo(f"\n{len(results)} model(s) found.")
    
if __name__ == "__main__":
    search()