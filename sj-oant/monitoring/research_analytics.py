"""
research_analytics.py - Research-Grade Monitoring and Analytics System

This module provides comprehensive monitoring, analytics, and research-grade insights
for the TMM system, enabling detailed performance analysis and research publication.

Key Features:
- Real-time performance monitoring
- Research-grade metrics collection
- System health monitoring
- Performance trend analysis
- Research publication data export
- A/B testing framework
"""

import logging
import time
import json
import csv
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Different types of metrics."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    USER_EXPERIENCE = "user_experience"

class AlertLevel(Enum):
    """Alert levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricDataPoint:
    """A single data point for a metric."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemAlert:
    """System alert for monitoring."""
    timestamp: float
    level: AlertLevel
    component: str
    message: str
    metric_name: str
    threshold: float
    actual_value: float

@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    memory_usage: Dict[str, Any]
    agent_performance: Dict[str, Any]
    benchmark_scores: Dict[str, Any]

class ResearchAnalytics:
    """
    Research-grade analytics and monitoring system for the TMM system.
    
    This component provides comprehensive monitoring, analytics, and insights
    for research purposes and system optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the research analytics system.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or self._default_config()
        
        # Metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=self.config['max_metric_points']))
        self.performance_snapshots = deque(maxlen=self.config['max_snapshots'])
        self.alerts = deque(maxlen=self.config['max_alerts'])
        
        # System state tracking
        self.system_start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = deque(maxlen=1000)
        
        # Component monitoring
        self.component_metrics = defaultdict(lambda: defaultdict(list))
        self.benchmark_scores = defaultdict(list)
        
        # Research data export
        self.research_data = {
            'experiments': [],
            'benchmark_results': [],
            'performance_trends': [],
            'system_health': []
        }
        
        logger.info("ResearchAnalytics initialized with comprehensive monitoring capabilities")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the analytics system."""
        return {
            'max_metric_points': 10000,
            'max_snapshots': 1000,
            'max_alerts': 500,
            'snapshot_interval': 60,  # seconds
            'alert_thresholds': {
                'response_time': 5.0,  # seconds
                'error_rate': 0.1,  # 10%
                'memory_usage': 0.9,  # 90%
                'cpu_usage': 0.8  # 80%
            },
            'export_formats': ['json', 'csv', 'excel'],
            'real_time_monitoring': True
        }
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """
        Record a metric data point.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        data_point = MetricDataPoint(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {}
        )
        
        self.metrics[metric_name].append(data_point)
        
        # Check for alerts
        self._check_metric_alerts(metric_name, value)
        
        logger.debug(f"Recorded metric {metric_name}: {value}")
    
    def record_request(self, success: bool, response_time: float, metadata: Dict[str, Any] = None):
        """
        Record a request for performance tracking.
        
        Args:
            success: Whether the request was successful
            response_time: Response time in seconds
            metadata: Additional metadata
        """
        self.total_requests += 1
        self.response_times.append(response_time)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Record performance metrics
        self.record_metric('response_time', response_time, metadata)
        self.record_metric('request_success', 1.0 if success else 0.0, metadata)
        
        # Update component metrics
        if metadata:
            component = metadata.get('component', 'unknown')
            self.component_metrics[component]['response_times'].append(response_time)
            self.component_metrics[component]['success_rate'].append(1.0 if success else 0.0)
    
    def record_benchmark_score(self, benchmark: str, metric: str, score: float, metadata: Dict[str, Any] = None):
        """
        Record a benchmark score.
        
        Args:
            benchmark: Name of the benchmark
            metric: Name of the metric
            score: Score value
            metadata: Additional metadata
        """
        score_data = {
            'timestamp': time.time(),
            'benchmark': benchmark,
            'metric': metric,
            'score': score,
            'metadata': metadata or {}
        }
        
        self.benchmark_scores[f"{benchmark}_{metric}"].append(score_data)
        
        # Update research data
        self.research_data['benchmark_results'].append(score_data)
        
        logger.info(f"Recorded benchmark score: {benchmark}.{metric} = {score}")
    
    def record_agent_performance(self, agent_id: str, performance_data: Dict[str, Any]):
        """
        Record agent performance data.
        
        Args:
            agent_id: ID of the agent
            performance_data: Performance metrics
        """
        performance_data['timestamp'] = time.time()
        performance_data['agent_id'] = agent_id
        
        # Store in component metrics
        for metric, value in performance_data.items():
            if metric not in ['timestamp', 'agent_id']:
                self.component_metrics[agent_id][metric].append(value)
        
        logger.debug(f"Recorded agent performance for {agent_id}")
    
    def take_performance_snapshot(self) -> PerformanceSnapshot:
        """
        Take a snapshot of current system performance.
        
        Returns:
            Performance snapshot
        """
        current_time = time.time()
        
        # Calculate current metrics
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
        success_rate = self.successful_requests / max(self.total_requests, 1)
        
        # Get memory usage (simplified)
        memory_usage = {
            'total_requests': self.total_requests,
            'active_metrics': sum(len(metrics) for metrics in self.metrics.values()),
            'alerts_count': len(self.alerts)
        }
        
        # Get agent performance summary
        agent_performance = {}
        for agent_id, metrics in self.component_metrics.items():
            if metrics:
                agent_performance[agent_id] = {
                    'avg_response_time': statistics.mean(metrics.get('response_times', [0])) or 0.0,
                    'success_rate': statistics.mean(metrics.get('success_rate', [0])) or 0.0,
                    'total_operations': len(metrics.get('response_times', []))
                }
        
        # Get latest benchmark scores
        latest_benchmark_scores = {}
        for benchmark_metric, scores in self.benchmark_scores.items():
            if scores:
                latest_benchmark_scores[benchmark_metric] = scores[-1]['score']
        
        snapshot = PerformanceSnapshot(
            timestamp=current_time,
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            avg_response_time=avg_response_time,
            memory_usage=memory_usage,
            agent_performance=agent_performance,
            benchmark_scores=latest_benchmark_scores
        )
        
        self.performance_snapshots.append(snapshot)
        self.research_data['performance_trends'].append(asdict(snapshot))
        
        logger.info(f"Performance snapshot taken: {self.total_requests} requests, {success_rate:.2%} success rate")
        return snapshot
    
    def _check_metric_alerts(self, metric_name: str, value: float):
        """Check if a metric value triggers an alert."""
        thresholds = self.config['alert_thresholds']
        
        if metric_name in thresholds:
            threshold = thresholds[metric_name]
            if value > threshold:
                alert_level = AlertLevel.WARNING if value < threshold * 1.5 else AlertLevel.ERROR
                
                alert = SystemAlert(
                    timestamp=time.time(),
                    level=alert_level,
                    component='analytics',
                    message=f"Metric {metric_name} exceeded threshold",
                    metric_name=metric_name,
                    threshold=threshold,
                    actual_value=value
                )
                
                self.alerts.append(alert)
                self._log_alert(alert)
    
    def _log_alert(self, alert: SystemAlert):
        """Log an alert."""
        log_message = f"ALERT [{alert.level.value.upper()}] {alert.component}: {alert.message}"
        log_message += f" ({alert.metric_name}: {alert.actual_value} > {alert.threshold})"
        
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif alert.level == AlertLevel.ERROR:
            logger.error(log_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health status.
        
        Returns:
            Dictionary containing system health information
        """
        current_time = time.time()
        uptime = current_time - self.system_start_time
        
        # Calculate health metrics
        success_rate = self.successful_requests / max(self.total_requests, 1)
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
        
        # Count recent alerts
        recent_alerts = [
            alert for alert in self.alerts 
            if current_time - alert.timestamp < 3600  # Last hour
        ]
        
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.level.value] += 1
        
        # Determine overall health status
        if alert_counts[AlertLevel.CRITICAL.value] > 0:
            health_status = "critical"
        elif alert_counts[AlertLevel.ERROR.value] > 5:
            health_status = "degraded"
        elif alert_counts[AlertLevel.WARNING.value] > 10:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        health_data = {
            'status': health_status,
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'recent_alerts': dict(alert_counts),
            'active_metrics': len(self.metrics),
            'performance_snapshots': len(self.performance_snapshots)
        }
        
        # Store in research data
        self.research_data['system_health'].append({
            'timestamp': current_time,
            **health_data
        })
        
        return health_data
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends over a specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary containing performance trends
        """
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        # Filter snapshots by time
        recent_snapshots = [
            snapshot for snapshot in self.performance_snapshots
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {"message": "No recent performance data available"}
        
        # Calculate trends
        response_times = [s.avg_response_time for s in recent_snapshots]
        success_rates = [s.successful_requests / max(s.total_requests, 1) for s in recent_snapshots]
        
        trends = {
            'time_period_hours': hours,
            'snapshots_analyzed': len(recent_snapshots),
            'response_time': {
                'current': response_times[-1] if response_times else 0.0,
                'average': statistics.mean(response_times) if response_times else 0.0,
                'trend': self._calculate_trend(response_times),
                'min': min(response_times) if response_times else 0.0,
                'max': max(response_times) if response_times else 0.0
            },
            'success_rate': {
                'current': success_rates[-1] if success_rates else 0.0,
                'average': statistics.mean(success_rates) if success_rates else 0.0,
                'trend': self._calculate_trend(success_rates),
                'min': min(success_rates) if success_rates else 0.0,
                'max': max(success_rates) if success_rates else 0.0
            }
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.05:
            return "increasing"
        elif second_avg < first_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def export_research_data(self, format: str = 'json', filename: str = None) -> str:
        """
        Export research data in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'excel')
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_data_{timestamp}.{format}"
        
        if format == 'json':
            return self._export_json(filename)
        elif format == 'csv':
            return self._export_csv(filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, filename: str) -> str:
        """Export data as JSON."""
        export_data = {
            'export_timestamp': time.time(),
            'system_info': {
                'uptime': time.time() - self.system_start_time,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests
            },
            'research_data': self.research_data,
            'performance_snapshots': [asdict(snapshot) for snapshot in self.performance_snapshots],
            'alerts': [asdict(alert) for alert in self.alerts],
            'metrics_summary': self._get_metrics_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Research data exported to {filename}")
        return filename
    
    def _export_csv(self, filename: str) -> str:
        """Export data as CSV."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write performance snapshots
            writer.writerow(['Type', 'Timestamp', 'Metric', 'Value'])
            
            for snapshot in self.performance_snapshots:
                writer.writerow(['snapshot', snapshot.timestamp, 'total_requests', snapshot.total_requests])
                writer.writerow(['snapshot', snapshot.timestamp, 'success_rate', 
                               snapshot.successful_requests / max(snapshot.total_requests, 1)])
                writer.writerow(['snapshot', snapshot.timestamp, 'avg_response_time', snapshot.avg_response_time])
            
            # Write benchmark scores
            for benchmark_metric, scores in self.benchmark_scores.items():
                for score_data in scores:
                    writer.writerow(['benchmark', score_data['timestamp'], benchmark_metric, score_data['score']])
        
        logger.info(f"Research data exported to {filename}")
        return filename
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for metric_name, data_points in self.metrics.items():
            if data_points:
                values = [dp.value for dp in data_points]
                summary[metric_name] = {
                    'count': len(values),
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else None
                }
        
        return summary
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics for research purposes.
        
        Returns:
            Dictionary containing comprehensive analytics
        """
        return {
            'system_health': self.get_system_health(),
            'performance_trends': self.get_performance_trends(),
            'metrics_summary': self._get_metrics_summary(),
            'component_performance': dict(self.component_metrics),
            'benchmark_scores': dict(self.benchmark_scores),
            'recent_alerts': [asdict(alert) for alert in list(self.alerts)[-10:]],  # Last 10 alerts
            'research_readiness': self._assess_research_readiness()
        }
    
    def _assess_research_readiness(self) -> Dict[str, Any]:
        """Assess research readiness based on collected data."""
        # Check if we have sufficient data
        data_sufficiency = {
            'total_requests': self.total_requests >= 100,
            'benchmark_scores': len(self.benchmark_scores) >= 4,  # All 4 benchmarks
            'performance_snapshots': len(self.performance_snapshots) >= 10,
            'uptime_hours': (time.time() - self.system_start_time) >= 3600  # 1 hour
        }
        
        # Calculate overall readiness score
        readiness_score = sum(data_sufficiency.values()) / len(data_sufficiency)
        
        return {
            'readiness_score': readiness_score,
            'data_sufficiency': data_sufficiency,
            'recommendations': self._get_research_recommendations(data_sufficiency)
        }
    
    def _get_research_recommendations(self, data_sufficiency: Dict[str, bool]) -> List[str]:
        """Get recommendations for improving research readiness."""
        recommendations = []
        
        if not data_sufficiency['total_requests']:
            recommendations.append("Collect more request data (minimum 100 requests)")
        
        if not data_sufficiency['benchmark_scores']:
            recommendations.append("Run comprehensive benchmark evaluations")
        
        if not data_sufficiency['performance_snapshots']:
            recommendations.append("Collect more performance snapshots")
        
        if not data_sufficiency['uptime_hours']:
            recommendations.append("Run system for longer periods to collect stability data")
        
        if not recommendations:
            recommendations.append("System is ready for research publication")
        
        return recommendations
