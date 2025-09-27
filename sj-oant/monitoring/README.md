# Research Analytics and Monitoring

This directory contains the research analytics and performance monitoring system for TMMA.

## Components

### `research_analytics.py`
Comprehensive analytics system providing:
- **Performance Monitoring**: Real-time system performance tracking
- **Memory Analytics**: Memory usage, tier distribution, and efficiency metrics
- **Verification Analytics**: Truth verification accuracy and contradiction detection rates
- **Evaluation Analytics**: Benchmark performance and false memory prevention metrics
- **System Health**: Overall system status and component health monitoring

## Analytics Features

### Performance Metrics
- **Response Time**: End-to-end processing latency
- **Memory Efficiency**: Memory usage and retrieval performance
- **Verification Speed**: Truth verification and contradiction detection timing
- **Throughput**: Conversations processed per unit time

### Memory Analytics
- **Tier Distribution**: Content distribution across L1, L2, L3, and FLAGGED tiers
- **Confidence Distribution**: Confidence score distribution and trends
- **Eviction Patterns**: Memory eviction and promotion patterns
- **Contradiction Rates**: False memory detection and quarantine statistics

### Verification Analytics
- **Truth Score Distribution**: Verification confidence across all content
- **Contradiction Detection**: False memory detection accuracy and patterns
- **Risk Assessment**: Risk score distribution and threshold effectiveness
- **Verification Latency**: Time required for truth verification

### Evaluation Analytics
- **Benchmark Performance**: Performance across MultiWOZ, SGD, and Taskmaster
- **False Memory Prevention**: FMR, MEL, DAR, and CDR metrics tracking
- **Baseline Comparison**: Performance relative to baseline systems
- **Trend Analysis**: Performance trends over time and across evaluations

## Monitoring Dashboard

### Real-Time Metrics
- **System Status**: Overall system health and component status
- **Active Conversations**: Current conversation processing status
- **Memory Usage**: Real-time memory tier utilization
- **Verification Queue**: Pending verification tasks and processing status

### Historical Analysis
- **Performance Trends**: Long-term performance analysis
- **Memory Evolution**: Memory system behavior over time
- **Verification Patterns**: Truth verification accuracy trends
- **Evaluation Results**: Historical benchmark performance

## Usage

### Basic Monitoring
```python
from sj_oant.monitoring import research_analytics

# Initialize analytics
analytics = research_analytics.ResearchAnalytics()

# Start monitoring
analytics.start_monitoring()

# Get current metrics
metrics = analytics.get_current_metrics()
```

### Custom Analytics
```python
# Track custom metrics
analytics.track_custom_metric("custom_metric", value)

# Generate reports
report = analytics.generate_performance_report()

# Export data
analytics.export_analytics_data("analytics_export.json")
```

## Configuration

### Monitoring Settings
- **Sampling Rate**: Frequency of metric collection
- **Retention Period**: How long to retain historical data
- **Alert Thresholds**: Performance thresholds for alerts
- **Export Format**: Data export format and frequency

### Analytics Parameters
- **Aggregation Windows**: Time windows for metric aggregation
- **Statistical Measures**: Mean, median, percentiles for analysis
- **Trend Detection**: Parameters for trend analysis
- **Anomaly Detection**: Thresholds for anomaly identification

## Data Storage

### Metrics Storage
- **Real-Time Data**: In-memory storage for current metrics
- **Historical Data**: Persistent storage for trend analysis
- **Export Data**: Structured data for external analysis
- **Backup Data**: Regular backups of analytics data

### Data Formats
- **JSON**: Structured data for programmatic access
- **CSV**: Tabular data for spreadsheet analysis
- **Time Series**: Time-stamped data for trend analysis
- **Logs**: Detailed logs for debugging and analysis

## Integration

The monitoring system integrates with:
- **Memory System**: Tracks memory operations and efficiency
- **Truth Verification**: Monitors verification performance and accuracy
- **Multi-Agent System**: Tracks agent coordination and performance
- **Evaluation Framework**: Monitors benchmark performance and results

## Reporting

### Automated Reports
- **Daily Reports**: Daily performance summaries
- **Weekly Reports**: Weekly trend analysis and insights
- **Monthly Reports**: Monthly performance reviews
- **Evaluation Reports**: Post-evaluation performance analysis

### Custom Reports
- **Performance Dashboards**: Real-time performance visualization
- **Trend Analysis**: Historical performance trends
- **Comparative Analysis**: Performance comparison across configurations
- **Anomaly Reports**: Unusual performance patterns and alerts

## Best Practices

### Monitoring Guidelines
- **Comprehensive Coverage**: Monitor all critical system components
- **Real-Time Alerts**: Immediate notification of performance issues
- **Historical Analysis**: Long-term trend analysis and insights
- **Data Quality**: Ensure accurate and reliable metric collection

### Analytics Best Practices
- **Statistical Rigor**: Use appropriate statistical measures
- **Trend Analysis**: Identify and analyze performance trends
- **Anomaly Detection**: Detect and investigate unusual patterns
- **Actionable Insights**: Provide actionable recommendations

## Troubleshooting

### Common Issues
- **Performance Degradation**: Identify and resolve performance issues
- **Memory Leaks**: Detect and fix memory usage problems
- **Verification Errors**: Troubleshoot truth verification issues
- **System Failures**: Diagnose and resolve system failures

### Diagnostic Tools
- **Performance Profiling**: Detailed performance analysis
- **Memory Analysis**: Memory usage and leak detection
- **Error Tracking**: Comprehensive error logging and analysis
- **System Health Checks**: Automated system health validation
