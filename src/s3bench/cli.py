"""Typer CLI for S3 provider benchmarking."""

import csv
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from s3bench import config as cfg
from s3bench.config import ProviderType
from s3bench.benchmark import run_benchmark, DEFAULT_ITERATIONS, BenchmarkResult
from s3bench.providers import create_provider
from s3bench.features import run_feature_tests, FEATURE_TESTS, FeatureStatus
from s3bench.rclone import create_rclone_provider, check_rclone_installed


def get_results_dir() -> Path:
    """Get the results storage directory."""
    results_dir = Path.home() / ".config" / "s3bench" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_results(result: BenchmarkResult) -> Path:
    """Save benchmark results to JSON file."""
    results_dir = get_results_dir()
    filename = f"benchmark_{result.timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = results_dir / filename

    data = {
        "timestamp": result.timestamp.isoformat(),
        "providers": {},
    }
    for provider_name, provider_result in result.provider_results.items():
        data["providers"][provider_name] = {
            "size_results": {
                str(size): {
                    "size_bytes": sr.size_bytes,
                    "upload_throughputs": sr.upload_throughputs,
                    "download_throughputs": sr.download_throughputs,
                    "latencies": sr.latencies,
                }
                for size, sr in provider_result.size_results.items()
            },
            "errors": provider_result.errors,
        }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def get_machine_info() -> dict:
    """Get current machine information."""
    hostname = socket.gethostname()

    # Get local IP by connecting to a remote address (doesn't actually send data)
    ip_address = "unknown"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
    except Exception:
        try:
            ip_address = socket.gethostbyname(hostname)
        except socket.gaierror:
            pass

    return {"hostname": hostname, "ip_address": ip_address}


def get_csv_path() -> Path:
    """Get the CSV results file path."""
    return get_results_dir() / "benchmarks.csv"


def save_results_csv(result: BenchmarkResult) -> Path:
    """Append benchmark results to CSV file in tidy/long format with individual iterations."""
    csv_path = get_csv_path()
    machine_info = get_machine_info()
    file_exists = csv_path.exists()

    # Collect all rows in tidy format (one row per metric per iteration)
    rows = []
    for provider_name, provider_result in result.provider_results.items():
        # Parse provider name and method
        if provider_name.endswith("_rclone"):
            base_provider = provider_name[:-7]  # Remove "_rclone"
            method = "rclone"
        else:
            base_provider = provider_name
            method = "sdk"

        for size_bytes, size_result in provider_result.size_results.items():
            base = {
                "date": result.timestamp.strftime("%Y-%m-%d"),
                "timestamp": result.timestamp.isoformat(),
                "hostname": machine_info["hostname"],
                "ip_address": machine_info["ip_address"],
                "provider": base_provider,
                "method": method,
                "size_bytes": size_result.size_bytes,
                "size_label": size_result.size_label,
            }
            # Individual upload iterations
            for i, val in enumerate(size_result.upload_throughputs, 1):
                rows.append({**base, "iteration": i, "metric": "upload_mbps", "value": round(val, 3)})
            # Individual download iterations
            for i, val in enumerate(size_result.download_throughputs, 1):
                rows.append({**base, "iteration": i, "metric": "download_mbps", "value": round(val, 3)})
            # Individual latency iterations
            for i, val in enumerate(size_result.latencies, 1):
                rows.append({**base, "iteration": i, "metric": "latency_sec", "value": round(val, 6)})

    if not rows:
        return csv_path

    fieldnames = [
        "date", "timestamp", "hostname", "ip_address", "provider", "method",
        "size_bytes", "size_label", "iteration", "metric", "value"
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    return csv_path


app = typer.Typer(help="S3 Provider Benchmark Tool")
provider_app = typer.Typer(help="Manage S3 providers")
app.add_typer(provider_app, name="provider")

console = Console()


@provider_app.command("add")
def provider_add(
    name: Annotated[str, typer.Argument(help="Provider name")],
    bucket: Annotated[str, typer.Option("--bucket", "-b", help="Bucket/container name", prompt=True)],
    access_key: Annotated[str, typer.Option("--access-key", "-a", help="Access key (or Azure storage account name)", prompt=True)],
    secret_key: Annotated[
        str, typer.Option("--secret-key", "-s", help="Secret key", prompt=True, hide_input=True)
    ],
    endpoint_url: Annotated[
        Optional[str], typer.Option("--endpoint", "-e", help="Custom endpoint URL")
    ] = None,
    region: Annotated[Optional[str], typer.Option("--region", "-r", help="Region")] = None,
    provider_type: Annotated[
        ProviderType, typer.Option("--type", "-t", help="Provider type (s3 or azure)")
    ] = ProviderType.S3,
) -> None:
    """Add a new storage provider."""
    cfg.add_provider(
        name=name,
        bucket=bucket,
        access_key=access_key,
        secret_key=secret_key,
        endpoint_url=endpoint_url,
        region=region,
        provider_type=provider_type,
    )
    console.print(f"Provider [bold green]{name}[/] ({provider_type.value}) added successfully.")


@provider_app.command("list")
def provider_list() -> None:
    """List all configured providers."""
    config = cfg.load_config()

    if not config.providers:
        console.print("No providers configured. Use [bold]s3bench provider add[/] to add one.")
        return

    table = Table(title="Configured Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Endpoint")
    table.add_column("Region")
    table.add_column("Bucket", style="green")

    for name, provider in config.providers.items():
        table.add_row(
            name,
            provider.provider_type.value,
            provider.endpoint_url or "(default)",
            provider.region or "-",
            provider.bucket,
        )

    console.print(table)


@provider_app.command("remove")
def provider_remove(
    name: Annotated[str, typer.Argument(help="Provider name to remove")],
) -> None:
    """Remove a provider."""
    if cfg.remove_provider(name):
        console.print(f"Provider [bold red]{name}[/] removed.")
    else:
        console.print(f"Provider [bold]{name}[/] not found.", style="red")
        raise typer.Exit(1)


@provider_app.command("test")
def provider_test(
    name: Annotated[str, typer.Argument(help="Provider name to test")],
) -> None:
    """Test connection to a provider."""
    provider_config = cfg.get_provider(name)

    if not provider_config:
        console.print(f"Provider [bold]{name}[/] not found.", style="red")
        raise typer.Exit(1)

    provider = create_provider(name, provider_config)

    with console.status(f"Testing connection to {name}..."):
        if provider.test_connection():
            console.print(f"[bold green]Success![/] Connected to {name}")
        else:
            console.print(f"[bold red]Failed![/] Could not connect to {name}")
            raise typer.Exit(1)


@provider_app.command("cleanup")
def provider_cleanup(
    name: Annotated[str, typer.Argument(help="Provider name to clean up")],
) -> None:
    """Remove leftover benchmark files from a provider."""
    provider_config = cfg.get_provider(name)

    if not provider_config:
        console.print(f"Provider [bold]{name}[/] not found.", style="red")
        raise typer.Exit(1)

    provider = create_provider(name, provider_config)

    with console.status(f"Cleaning up benchmark files from {name}..."):
        count = provider.cleanup_benchmark_objects()

    if count > 0:
        console.print(f"Deleted [bold]{count}[/] benchmark files from {name}")
    else:
        console.print(f"No benchmark files found in {name}")


@app.command("run")
def run(
    providers: Annotated[
        Optional[list[str]],
        typer.Option("--provider", "-p", help="Provider(s) to benchmark"),
    ] = None,
    all_providers: Annotated[
        bool, typer.Option("--all", "-a", help="Run against all providers")
    ] = False,
    sizes: Annotated[
        str, typer.Option("--sizes", "-s", help="Size categories: small,medium,large")
    ] = "small,medium,large",
    small_iter: Annotated[
        int, typer.Option("--small-iter", help="Iterations for small files")
    ] = DEFAULT_ITERATIONS["small"],
    medium_iter: Annotated[
        int, typer.Option("--medium-iter", help="Iterations for medium files")
    ] = DEFAULT_ITERATIONS["medium"],
    large_iter: Annotated[
        int, typer.Option("--large-iter", help="Iterations for large files")
    ] = DEFAULT_ITERATIONS["large"],
    method: Annotated[
        str, typer.Option("--method", "-m", help="Benchmark method: sdk, rclone, or both")
    ] = "both",
) -> None:
    """Run benchmarks against providers."""
    config = cfg.load_config()

    if not config.providers:
        console.print("No providers configured. Use [bold]s3bench provider add[/] first.")
        raise typer.Exit(1)

    # Validate method
    valid_methods = {"sdk", "rclone", "both"}
    if method not in valid_methods:
        console.print(f"Invalid method: {method}. Use: sdk, rclone, or both", style="red")
        raise typer.Exit(1)

    # Check rclone if needed
    if method in ("rclone", "both") and not check_rclone_installed():
        console.print("rclone is not installed. Install from https://rclone.org/install/", style="red")
        raise typer.Exit(1)

    # Determine which providers to run
    if all_providers:
        provider_names = list(config.providers.keys())
    elif providers:
        provider_names = providers
    else:
        console.print("Specify providers with --provider or use --all")
        raise typer.Exit(1)

    # Validate providers exist and create instances
    storage_providers = []
    rclone_providers = []
    for name in provider_names:
        provider_config = config.providers.get(name)
        if not provider_config:
            console.print(f"Provider [bold]{name}[/] not found.", style="red")
            raise typer.Exit(1)
        if method in ("sdk", "both"):
            storage_providers.append(create_provider(name, provider_config))
        if method in ("rclone", "both"):
            rclone_providers.append(create_rclone_provider(f"{name}_rclone", provider_config))

    # Parse size categories
    categories = [s.strip() for s in sizes.split(",")]
    valid_categories = {"small", "medium", "large"}
    for cat in categories:
        if cat not in valid_categories:
            console.print(f"Invalid size category: {cat}", style="red")
            raise typer.Exit(1)

    iterations = {
        "small": small_iter,
        "medium": medium_iter,
        "large": large_iter,
    }

    # Combine all providers for benchmarking
    all_benchmark_providers = storage_providers + rclone_providers

    method_label = {"sdk": "SDK", "rclone": "rclone", "both": "SDK + rclone"}[method]
    console.print(f"Running benchmarks for: [bold]{', '.join(provider_names)}[/]")
    console.print(f"Method: [bold]{method_label}[/]")
    console.print(f"Size categories: [bold]{', '.join(categories)}[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=100)

        def update_progress(msg: str, current: int, total: int) -> None:
            progress.update(task, completed=(current / total) * 100, description=msg)

        result = run_benchmark(
            providers=all_benchmark_providers,
            categories=categories,
            iterations=iterations,
            progress_callback=update_progress,
        )

    # Cleanup rclone temp files
    for rp in rclone_providers:
        rp.cleanup()

    # Save and display results
    results_file = save_results(result)
    csv_file = save_results_csv(result)
    console.print(f"\nResults saved to: [dim]{results_file}[/]")
    console.print(f"CSV appended to: [dim]{csv_file}[/]\n")
    _display_results(result, categories)


def _display_results(result, categories: list[str]) -> None:
    """Display benchmark results as tables."""
    from s3bench.benchmark import get_sizes_for_category

    # Get all sizes tested
    all_sizes = []
    for cat in categories:
        all_sizes.extend(get_sizes_for_category(cat))

    # Upload throughput table
    upload_table = Table(title="Upload Throughput (MB/s)")
    upload_table.add_column("Size", style="cyan")
    for provider_name in result.provider_results:
        upload_table.add_column(provider_name, justify="right")

    for size in all_sizes:
        row = [size.name.replace("_", " ")]
        for provider_name, provider_result in result.provider_results.items():
            size_result = provider_result.size_results.get(size.value)
            if size_result:
                row.append(f"{size_result.avg_upload_throughput:.2f}")
            else:
                row.append("-")
        upload_table.add_row(*row)

    console.print(upload_table)
    console.print()

    # Download throughput table
    download_table = Table(title="Download Throughput (MB/s)")
    download_table.add_column("Size", style="cyan")
    for provider_name in result.provider_results:
        download_table.add_column(provider_name, justify="right")

    for size in all_sizes:
        row = [size.name.replace("_", " ")]
        for provider_name, provider_result in result.provider_results.items():
            size_result = provider_result.size_results.get(size.value)
            if size_result:
                row.append(f"{size_result.avg_download_throughput:.2f}")
            else:
                row.append("-")
        download_table.add_row(*row)

    console.print(download_table)
    console.print()

    # Latency table
    latency_table = Table(title="Latency / TTFB (seconds)")
    latency_table.add_column("Size", style="cyan")
    for provider_name in result.provider_results:
        latency_table.add_column(provider_name, justify="right")

    for size in all_sizes:
        row = [size.name.replace("_", " ")]
        for provider_name, provider_result in result.provider_results.items():
            size_result = provider_result.size_results.get(size.value)
            if size_result:
                row.append(f"{size_result.avg_latency:.4f}")
            else:
                row.append("-")
        latency_table.add_row(*row)

    console.print(latency_table)

    # Show errors if any
    for provider_name, provider_result in result.provider_results.items():
        if provider_result.errors:
            console.print(f"\n[bold red]Errors for {provider_name}:[/]")
            for error in provider_result.errors:
                console.print(f"  - {error}")


@app.command("results")
def results(
    last: Annotated[int, typer.Option("--last", "-n", help="Show last N results")] = 5,
) -> None:
    """List past benchmark results."""
    results_dir = get_results_dir()
    files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)

    if not files:
        console.print("No benchmark results found.")
        return

    table = Table(title="Benchmark Results")
    table.add_column("Date", style="cyan")
    table.add_column("Providers")
    table.add_column("File")

    for f in files[:last]:
        with open(f) as fp:
            data = json.load(fp)
        providers = ", ".join(data.get("providers", {}).keys())
        timestamp = data.get("timestamp", "unknown")
        table.add_row(timestamp[:19], providers, f.name)

    console.print(table)
    console.print(f"\nResults stored in: [dim]{results_dir}[/]")


@app.command("features")
def features(
    providers: Annotated[
        Optional[list[str]],
        typer.Option("--provider", "-p", help="Provider(s) to test"),
    ] = None,
    all_providers: Annotated[
        bool, typer.Option("--all", "-a", help="Test all providers")
    ] = False,
    feature_list: Annotated[
        Optional[str], typer.Option("--features", "-f", help="Comma-separated features to test")
    ] = None,
    output_json: Annotated[
        bool, typer.Option("--json", help="Output as JSON")
    ] = False,
) -> None:
    """Test S3 API feature support across providers."""
    config = cfg.load_config()

    if not config.providers:
        console.print("No providers configured. Use [bold]s3bench provider add[/] first.")
        raise typer.Exit(1)

    # Determine which providers to test
    if all_providers:
        provider_names = list(config.providers.keys())
    elif providers:
        provider_names = providers
    else:
        console.print("Specify providers with --provider or use --all")
        raise typer.Exit(1)

    # Parse features
    features_to_test = None
    if feature_list:
        features_to_test = [f.strip() for f in feature_list.split(",")]
        for f in features_to_test:
            if f not in FEATURE_TESTS:
                console.print(f"Unknown feature: {f}", style="red")
                console.print(f"Available: {', '.join(FEATURE_TESTS.keys())}")
                raise typer.Exit(1)

    # Validate providers and create instances
    storage_providers = []
    for name in provider_names:
        provider_config = config.providers.get(name)
        if not provider_config:
            console.print(f"Provider [bold]{name}[/] not found.", style="red")
            raise typer.Exit(1)
        storage_providers.append(create_provider(name, provider_config))

    console.print(f"Testing features for: [bold]{', '.join(provider_names)}[/]")

    # Run feature tests
    all_results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for provider in storage_providers:
            task = progress.add_task(f"Testing {provider.name}...", total=None)

            def update_progress(feature: str, current: int, total: int) -> None:
                progress.update(task, description=f"{provider.name}: {feature}")

            results = run_feature_tests(provider, features_to_test, update_progress)
            all_results[provider.name] = results
            progress.update(task, description=f"{provider.name}: Done")

    # Output results
    if output_json:
        json_output = {}
        for provider_name, provider_results in all_results.items():
            json_output[provider_name] = {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                }
                for name, result in provider_results.results.items()
            }
        console.print(json.dumps(json_output, indent=2))
    else:
        _display_feature_results(all_results)


def _display_feature_results(all_results: dict) -> None:
    """Display feature test results as a table."""
    # Get all feature names from results
    feature_names = set()
    for provider_results in all_results.values():
        feature_names.update(provider_results.results.keys())
    feature_names = sorted(feature_names)

    # Create table
    table = Table(title="Feature Support Matrix")
    table.add_column("Feature", style="cyan")

    provider_names = list(all_results.keys())
    for name in provider_names:
        table.add_column(name, justify="center")

    # Add rows
    for feature_name in feature_names:
        row = [feature_name]
        for provider_name in provider_names:
            provider_results = all_results[provider_name]
            result = provider_results.results.get(feature_name)
            if result:
                if result.status == FeatureStatus.SUPPORTED:
                    cell = "[green]Yes[/]"
                elif result.status == FeatureStatus.NOT_SUPPORTED:
                    cell = "[red]No[/]"
                elif result.status == FeatureStatus.NOT_APPLICABLE:
                    cell = "[dim]N/A[/]"
                else:
                    cell = f"[yellow]Error[/]"
            else:
                cell = "-"
            row.append(cell)
        table.add_row(*row)

    console.print()
    console.print(table)

    # Show any error messages
    has_errors = False
    for provider_name, provider_results in all_results.items():
        for feature_name, result in provider_results.results.items():
            if result.status == FeatureStatus.ERROR and result.message:
                if not has_errors:
                    console.print("\n[bold yellow]Errors:[/]")
                    has_errors = True
                console.print(f"  {provider_name}/{feature_name}: {result.message}")


if __name__ == "__main__":
    app()
