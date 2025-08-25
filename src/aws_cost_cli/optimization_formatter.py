"""Formatting utilities for cost optimization reports."""

from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.tree import Tree

from .cost_optimizer import (
    OptimizationReport, OptimizationRecommendation, CostAnomaly, 
    BudgetVariance, OptimizationType, SeverityLevel
)


class OptimizationFormatter:
    """Formatter for cost optimization reports."""
    
    def __init__(self, console: Console = None):
        """Initialize formatter."""
        self.console = console or Console()
    
    def format_optimization_report(self, report: OptimizationReport) -> str:
        """Format complete optimization report."""
        output = []
        
        # Report header
        output.append(self._format_report_header(report))
        
        # Executive summary
        output.append(self._format_executive_summary(report))
        
        # Recommendations by type
        if report.recommendations:
            output.append(self._format_recommendations_by_type(report.recommendations))
        
        # Cost anomalies
        if report.anomalies:
            output.append(self._format_anomalies(report.anomalies))
        
        # Budget variances
        if report.budget_variances:
            output.append(self._format_budget_variances(report.budget_variances))
        
        # Action items summary
        output.append(self._format_action_items(report))
        
        return "\n\n".join(output)
    
    def _format_report_header(self, report: OptimizationReport) -> str:
        """Format report header."""
        header = Panel(
            Text("AWS Cost Optimization Report", style="bold blue", justify="center"),
            border_style="blue",
            padding=(1, 2)
        )
        
        with self.console.capture() as capture:
            self.console.print(header)
            self.console.print(f"📅 Report Date: {report.report_date.strftime('%Y-%m-%d %H:%M:%S')}")
            self.console.print(f"📊 Analysis Period: {report.analysis_period.start.strftime('%Y-%m-%d')} to {report.analysis_period.end.strftime('%Y-%m-%d')}")
            self.console.print(f"💰 Total Potential Savings: ${report.total_potential_savings.amount:,.2f}/month")
        
        return capture.get()
    
    def _format_executive_summary(self, report: OptimizationReport) -> str:
        """Format executive summary."""
        with self.console.capture() as capture:
            self.console.print(Panel(
                Text("Executive Summary", style="bold green"),
                border_style="green"
            ))
            
            # Summary statistics
            total_recommendations = len(report.recommendations)
            high_priority = len([r for r in report.recommendations if r.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]])
            total_anomalies = len(report.anomalies)
            budget_issues = len([b for b in report.budget_variances if b.is_over_budget])
            
            summary_table = Table(show_header=False, box=None)
            summary_table.add_column("Metric", style="bold")
            summary_table.add_column("Value", style="cyan")
            
            summary_table.add_row("📋 Total Recommendations", str(total_recommendations))
            summary_table.add_row("🚨 High Priority Items", str(high_priority))
            summary_table.add_row("⚠️  Cost Anomalies", str(total_anomalies))
            summary_table.add_row("💸 Budget Overruns", str(budget_issues))
            summary_table.add_row("💰 Monthly Savings Potential", f"${report.total_potential_savings.amount:,.2f}")
            
            self.console.print(summary_table)
            
            # Key insights
            insights = self._generate_key_insights(report)
            if insights:
                self.console.print("\n🔍 Key Insights:")
                for insight in insights:
                    self.console.print(f"   • {insight}")
        
        return capture.get()
    
    def _format_recommendations_by_type(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Format recommendations grouped by type."""
        with self.console.capture() as capture:
            self.console.print(Panel(
                Text("Cost Optimization Recommendations", style="bold yellow"),
                border_style="yellow"
            ))
            
            # Group recommendations by type
            by_type = {}
            for rec in recommendations:
                if rec.type not in by_type:
                    by_type[rec.type] = []
                by_type[rec.type].append(rec)
            
            # Sort by potential savings (highest first)
            for rec_type in by_type:
                by_type[rec_type].sort(key=lambda x: x.potential_savings.amount, reverse=True)
            
            # Display each type
            for rec_type, recs in by_type.items():
                type_savings = sum(r.potential_savings.amount for r in recs)
                
                self.console.print(f"\n{self._get_type_icon(rec_type)} {self._get_type_title(rec_type)}")
                self.console.print(f"   💰 Potential Savings: ${type_savings:,.2f}/month")
                self.console.print(f"   📊 {len(recs)} recommendation(s)")
                
                # Show top recommendations for this type
                for i, rec in enumerate(recs[:3]):  # Show top 3
                    severity_icon = self._get_severity_icon(rec.severity)
                    self.console.print(f"   {severity_icon} {rec.title}")
                    self.console.print(f"      💵 ${rec.potential_savings.amount:,.2f}/month")
                    self.console.print(f"      📝 {rec.description}")
                    if rec.action_required:
                        self.console.print(f"      🔧 Action: {rec.action_required}")
                
                if len(recs) > 3:
                    self.console.print(f"   ... and {len(recs) - 3} more recommendations")
        
        return capture.get()
    
    def _format_anomalies(self, anomalies: List[CostAnomaly]) -> str:
        """Format cost anomalies."""
        with self.console.capture() as capture:
            self.console.print(Panel(
                Text("Cost Anomalies Detected", style="bold red"),
                border_style="red"
            ))
            
            # Sort by severity and cost impact
            sorted_anomalies = sorted(
                anomalies, 
                key=lambda x: (x.severity.value, x.actual_cost.amount), 
                reverse=True
            )
            
            anomaly_table = Table()
            anomaly_table.add_column("Date", style="cyan")
            anomaly_table.add_column("Service", style="green")
            anomaly_table.add_column("Impact", style="red")
            anomaly_table.add_column("Variance", style="yellow")
            anomaly_table.add_column("Severity", style="bold")
            
            for anomaly in sorted_anomalies:
                severity_style = self._get_severity_style(anomaly.severity)
                anomaly_table.add_row(
                    anomaly.anomaly_date.strftime('%Y-%m-%d'),
                    anomaly.service,
                    f"${anomaly.actual_cost.amount:,.2f}",
                    f"{anomaly.variance_percentage:+.1f}%",
                    Text(anomaly.severity.value.upper(), style=severity_style)
                )
            
            self.console.print(anomaly_table)
            
            # Show root cause analysis for significant anomalies
            significant_anomalies = [a for a in sorted_anomalies if a.actual_cost.amount > 100]
            if significant_anomalies:
                self.console.print("\n🔍 Root Cause Analysis:")
                for anomaly in significant_anomalies[:3]:  # Show top 3
                    self.console.print(f"   • {anomaly.service}: {anomaly.root_cause_analysis or 'Investigation needed'}")
        
        return capture.get()
    
    def _format_budget_variances(self, variances: List[BudgetVariance]) -> str:
        """Format budget variances."""
        with self.console.capture() as capture:
            self.console.print(Panel(
                Text("Budget Variance Analysis", style="bold magenta"),
                border_style="magenta"
            ))
            
            # Sort by variance percentage (highest first)
            sorted_variances = sorted(variances, key=lambda x: abs(x.variance_percentage), reverse=True)
            
            variance_table = Table()
            variance_table.add_column("Budget", style="cyan")
            variance_table.add_column("Budgeted", style="green")
            variance_table.add_column("Actual", style="yellow")
            variance_table.add_column("Variance", style="bold")
            variance_table.add_column("Status", style="bold")
            
            for variance in sorted_variances:
                variance_style = "red" if variance.is_over_budget else "green"
                status_text = "OVER" if variance.is_over_budget else "UNDER"
                
                variance_table.add_row(
                    variance.budget_name,
                    f"${variance.budgeted_amount.amount:,.2f}",
                    f"${variance.actual_amount.amount:,.2f}",
                    f"{variance.variance_percentage:+.1f}%",
                    Text(status_text, style=variance_style)
                )
            
            self.console.print(variance_table)
            
            # Budget recommendations
            over_budget = [v for v in variances if v.is_over_budget]
            if over_budget:
                self.console.print("\n💡 Budget Recommendations:")
                for variance in over_budget[:3]:  # Show top 3
                    self.console.print(f"   • {variance.budget_name}: Review spending patterns and adjust budget or implement cost controls")
        
        return capture.get()
    
    def _format_action_items(self, report: OptimizationReport) -> str:
        """Format prioritized action items."""
        with self.console.capture() as capture:
            self.console.print(Panel(
                Text("Prioritized Action Items", style="bold blue"),
                border_style="blue"
            ))
            
            # Collect all actionable items
            action_items = []
            
            # High-priority recommendations
            high_priority_recs = [r for r in report.recommendations if r.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
            for rec in high_priority_recs:
                action_items.append({
                    'type': 'recommendation',
                    'priority': 1 if rec.severity == SeverityLevel.CRITICAL else 2,
                    'title': rec.title,
                    'action': rec.action_required,
                    'savings': rec.potential_savings.amount,
                    'effort': rec.estimated_effort
                })
            
            # Critical anomalies
            critical_anomalies = [a for a in report.anomalies if a.severity == SeverityLevel.CRITICAL]
            for anomaly in critical_anomalies:
                action_items.append({
                    'type': 'anomaly',
                    'priority': 1,
                    'title': f"Investigate {anomaly.service} anomaly",
                    'action': "Analyze root cause and implement corrective measures",
                    'savings': anomaly.actual_cost.amount,
                    'effort': 'medium'
                })
            
            # Budget overruns
            budget_overruns = [b for b in report.budget_variances if b.is_over_budget and b.variance_percentage > 20]
            for variance in budget_overruns:
                action_items.append({
                    'type': 'budget',
                    'priority': 2,
                    'title': f"Address {variance.budget_name} budget overrun",
                    'action': "Review spending and implement cost controls",
                    'savings': variance.variance_amount.amount,
                    'effort': 'high'
                })
            
            # Sort by priority and potential impact
            action_items.sort(key=lambda x: (x['priority'], -x['savings']))
            
            if action_items:
                action_table = Table()
                action_table.add_column("Priority", style="bold")
                action_table.add_column("Item", style="cyan")
                action_table.add_column("Action Required", style="yellow")
                action_table.add_column("Impact", style="green")
                action_table.add_column("Effort", style="magenta")
                
                for i, item in enumerate(action_items[:10], 1):  # Show top 10
                    priority_text = "🔴 CRITICAL" if item['priority'] == 1 else "🟡 HIGH"
                    effort_text = (item.get('effort') or 'unknown').upper()
                    
                    action_table.add_row(
                        priority_text,
                        item['title'],
                        item['action'] or "Review and take appropriate action",
                        f"${item['savings']:,.2f}",
                        effort_text
                    )
                
                self.console.print(action_table)
            else:
                self.console.print("✅ No critical action items identified")
        
        return capture.get()
    
    def _generate_key_insights(self, report: OptimizationReport) -> List[str]:
        """Generate key insights from the report."""
        insights = []
        
        # Savings potential insight
        if report.total_potential_savings.amount > 1000:
            insights.append(f"Significant savings opportunity: ${report.total_potential_savings.amount:,.2f}/month potential savings identified")
        
        # Unused resources insight
        unused_recs = [r for r in report.recommendations if r.type == OptimizationType.UNUSED_RESOURCES]
        if unused_recs:
            unused_savings = sum(r.potential_savings.amount for r in unused_recs)
            insights.append(f"${unused_savings:,.2f}/month can be saved by removing {len(unused_recs)} unused resources")
        
        # Rightsizing insight
        rightsizing_recs = [r for r in report.recommendations if r.type == OptimizationType.RIGHTSIZING]
        if rightsizing_recs:
            rightsizing_savings = sum(r.potential_savings.amount for r in rightsizing_recs)
            insights.append(f"${rightsizing_savings:,.2f}/month potential savings from rightsizing {len(rightsizing_recs)} resources")
        
        # Anomaly insight
        if report.anomalies:
            total_anomaly_impact = sum(a.actual_cost.amount for a in report.anomalies)
            insights.append(f"${total_anomaly_impact:,.2f} in unexpected costs detected across {len(report.anomalies)} anomalies")
        
        # Budget insight
        over_budget = [b for b in report.budget_variances if b.is_over_budget]
        if over_budget:
            insights.append(f"{len(over_budget)} budget(s) exceeded - review spending controls")
        
        return insights
    
    def _get_type_icon(self, rec_type: OptimizationType) -> str:
        """Get icon for recommendation type."""
        icons = {
            OptimizationType.UNUSED_RESOURCES: "🗑️",
            OptimizationType.RIGHTSIZING: "📏",
            OptimizationType.RESERVED_INSTANCES: "🏦",
            OptimizationType.SAVINGS_PLANS: "💳",
            OptimizationType.COST_ANOMALY: "⚠️",
            OptimizationType.BUDGET_VARIANCE: "📊"
        }
        return icons.get(rec_type, "📋")
    
    def _get_type_title(self, rec_type: OptimizationType) -> str:
        """Get title for recommendation type."""
        titles = {
            OptimizationType.UNUSED_RESOURCES: "Unused Resources",
            OptimizationType.RIGHTSIZING: "Rightsizing Opportunities",
            OptimizationType.RESERVED_INSTANCES: "Reserved Instance Recommendations",
            OptimizationType.SAVINGS_PLANS: "Savings Plan Opportunities",
            OptimizationType.COST_ANOMALY: "Cost Anomalies",
            OptimizationType.BUDGET_VARIANCE: "Budget Variances"
        }
        return titles.get(rec_type, "Other Recommendations")
    
    def _get_severity_icon(self, severity: SeverityLevel) -> str:
        """Get icon for severity level."""
        icons = {
            SeverityLevel.LOW: "🟢",
            SeverityLevel.MEDIUM: "🟡",
            SeverityLevel.HIGH: "🟠",
            SeverityLevel.CRITICAL: "🔴"
        }
        return icons.get(severity, "⚪")
    
    def _get_severity_style(self, severity: SeverityLevel) -> str:
        """Get Rich style for severity level."""
        styles = {
            SeverityLevel.LOW: "green",
            SeverityLevel.MEDIUM: "yellow",
            SeverityLevel.HIGH: "orange3",
            SeverityLevel.CRITICAL: "red"
        }
        return styles.get(severity, "white")
    
    def format_recommendations_summary(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Format a summary of recommendations."""
        if not recommendations:
            return "No optimization recommendations found."
        
        with self.console.capture() as capture:
            total_savings = sum(r.potential_savings.amount for r in recommendations)
            high_priority = len([r for r in recommendations if r.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]])
            
            self.console.print(f"💰 Total Potential Savings: ${total_savings:,.2f}/month")
            self.console.print(f"📋 {len(recommendations)} recommendations ({high_priority} high priority)")
            
            # Show top 3 recommendations
            sorted_recs = sorted(recommendations, key=lambda x: x.potential_savings.amount, reverse=True)
            self.console.print("\n🔝 Top Recommendations:")
            
            for i, rec in enumerate(sorted_recs[:3], 1):
                severity_icon = self._get_severity_icon(rec.severity)
                self.console.print(f"   {i}. {severity_icon} {rec.title}")
                self.console.print(f"      💵 ${rec.potential_savings.amount:,.2f}/month")
        
        return capture.get()