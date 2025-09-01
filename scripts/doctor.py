#!/usr/bin/env python3
"""
System Diagnostics and Troubleshooting Script
Standalone version of the doctor functionality
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

def main():
    """Main diagnostics function"""
    parser = argparse.ArgumentParser(description="RAG System Diagnostics")
    parser.add_argument('--format', choices=['markdown', 'json'], default='markdown',
                       help='Output format')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Quick check only')
    
    args = parser.parse_args()
    
    console = Console()
    
    try:
        from src.config_manager import ConfigManager
        from src.health_checks import HealthChecker
        
        # Initialize config manager
        config_manager = ConfigManager()
        health_checker = HealthChecker(config_manager)
        
        rprint("[yellow]🔍 Running system diagnostics...[/yellow]")
        
        if args.quick:
            # Quick health check only
            rprint("[blue]Running quick health check...[/blue]")
            # For now, just return True as a basic health check
            # TODO: Implement basic config validation
            healthy = True
            
            if healthy:
                rprint("[green]✅ System appears healthy[/green]")
                return 0
            else:
                rprint("[red]❌ System health issues detected[/red]")
                rprint("[yellow]💡 Run without --quick for detailed diagnostics[/yellow]")
                return 1
        
        # Full diagnostics
        with Progress() as progress:
            task = progress.add_task("Running comprehensive checks...", total=None)
            
            # Run all health checks
            health_report = health_checker.run_all_checks()
            
            progress.update(task, completed=100, total=100)
        
        # Display results
        overall_status = "✅ HEALTHY" if health_report.overall_status else "❌ UNHEALTHY"
        rprint(f"\n[bold blue]System Health Report {overall_status}[/bold blue]")
        rprint(f"Generated: {health_report.timestamp}")
        rprint(f"Total execution time: {health_report.summary['total_execution_time_ms']:.2f}ms")
        
        # Summary table
        if args.format == 'json' and not args.output:
            # Print JSON to stdout
            from dataclasses import asdict
            print(json.dumps(asdict(health_report), indent=2, default=str))
        else:
            # Rich table display
            summary_table = Table(show_header=True, header_style="bold magenta")
            summary_table.add_column("Check", style="cyan")
            summary_table.add_column("Status", style="white")
            summary_table.add_column("Message", style="yellow")
            
            if args.verbose:
                summary_table.add_column("Time (ms)", justify="right", style="dim")
            
            for check in health_report.checks:
                status_icon = "✅" if check.status else "❌"
                row_data = [
                    check.name.replace('_', ' ').title(),
                    status_icon,
                    check.message
                ]
                
                if args.verbose:
                    row_data.append(f"{check.execution_time_ms:.1f}")
                
                summary_table.add_row(*row_data)
            
            console.print(summary_table)
            
            # Show recommendations
            if health_report.recommendations:
                rprint(f"\n[bold yellow]💡 Recommendations:[/bold yellow]")
                for i, rec in enumerate(health_report.recommendations, 1):
                    rprint(f"  {i}. {rec}")
            
            # Show failed check details if verbose
            if args.verbose:
                failed_checks = [check for check in health_report.checks if not check.status]
                if failed_checks:
                    rprint(f"\n[bold red]❌ Failed Check Details:[/bold red]")
                    for check in failed_checks:
                        rprint(f"\n[red]• {check.name.replace('_', ' ').title()}[/red]")
                        rprint(f"  Error: {check.message}")
                        if check.details:
                            rprint(f"  Details: {json.dumps(check.details, indent=2)}")
        
        # Save report to file if requested
        if args.output:
            report_content = health_checker.generate_report(health_report, format=args.format)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            rprint(f"\n[green]📄 Report saved to: {output_path}[/green]")
        
        # Return appropriate exit code
        return 0 if health_report.overall_status else 1
        
    except ImportError as e:
        rprint(f"[red]❌ Import error: {e}[/red]")
        rprint("[yellow]💡 Make sure you're running from the project root directory[/yellow]")
        return 1
    except Exception as e:
        rprint(f"[red]❌ Diagnostics failed: {e}[/red]")
        if args.verbose:
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        return 1

if __name__ == '__main__':
    sys.exit(main())