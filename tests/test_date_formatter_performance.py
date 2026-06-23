"""Performance tests for DateFormatter with large datasets."""

import pytest
import time
import gc
import sys
from datetime import datetime, timedelta
from src.aws_cost_cli.date_formatter import DateFormatter
from src.aws_cost_cli.models import TimePeriod


class TestDateFormatterPerformanceLarge:
    """Comprehensive performance tests for DateFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
    
    def test_performance_10k_single_days(self):
        """Test performance with 10,000 single day periods."""
        # Generate 10,000 consecutive single days
        periods = []
        base_date = datetime(2020, 1, 1)
        
        for i in range(10000):
            start = base_date + timedelta(days=i)
            end = start + timedelta(days=1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Measure formatting time
        start_time = time.time()
        results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Should complete in reasonable time (less than 5 seconds for 10k periods)
        assert elapsed_time < 5.0, f"10k single days took too long: {elapsed_time:.2f} seconds"
        
        # Performance target: at least 2000 formats per second
        formats_per_second = len(periods) / elapsed_time
        assert formats_per_second > 2000, f"Too slow: {formats_per_second:.0f} formats/sec"
        
        # All results should be valid
        assert len(results) == 10000
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
        
        # Spot check some results
        assert results[0] == "January 1, 2020"
        assert "December" in results[-1] and "2047" in results[-1]  # Should be around end of 2047
    
    def test_performance_1k_mixed_periods(self):
        """Test performance with 1,000 mixed period types."""
        periods = []
        base_date = datetime(2020, 1, 1)
        
        # Generate mixed period types
        for i in range(1000):
            if i % 4 == 0:  # Single days
                start = base_date + timedelta(days=i)
                end = start + timedelta(days=1)
            elif i % 4 == 1:  # Single months
                month_offset = i // 30
                year = 2020 + month_offset // 12
                month = (month_offset % 12) + 1
                start = datetime(year, month, 1)
                if month == 12:
                    end = datetime(year + 1, 1, 1)
                else:
                    end = datetime(year, month + 1, 1)
            elif i % 4 == 2:  # Quarters
                quarter_offset = i // 90
                year = 2020 + quarter_offset // 4
                quarter = (quarter_offset % 4) + 1
                start_month = (quarter - 1) * 3 + 1
                start = datetime(year, start_month, 1)
                if start_month == 10:
                    end = datetime(year + 1, 1, 1)
                else:
                    end = datetime(year, start_month + 3, 1)
            else:  # Custom ranges
                start = base_date + timedelta(days=i * 2)
                end = start + timedelta(days=7)
            
            periods.append(TimePeriod(start=start, end=end))
        
        # Measure formatting time
        start_time = time.time()
        results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Should complete in reasonable time (less than 2 seconds for 1k mixed periods)
        assert elapsed_time < 2.0, f"1k mixed periods took too long: {elapsed_time:.2f} seconds"
        
        # Performance target: at least 500 formats per second for mixed types
        formats_per_second = len(periods) / elapsed_time
        assert formats_per_second > 500, f"Too slow: {formats_per_second:.0f} formats/sec"
        
        # All results should be valid
        assert len(results) == 1000
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        # Force garbage collection before test
        gc.collect()
        initial_memory = self._get_memory_usage()
        
        # Process 5000 periods without storing results
        base_date = datetime(2020, 1, 1)
        for i in range(5000):
            start = base_date + timedelta(days=i)
            end = start + timedelta(days=1)
            period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(period)
            # Don't store result to test memory cleanup
        
        # Force garbage collection after test
        gc.collect()
        final_memory = self._get_memory_usage()
        
        # Memory usage should not grow significantly
        memory_growth = final_memory - initial_memory
        # Allow up to 10MB growth for large dataset processing
        assert memory_growth < 10 * 1024 * 1024, f"Too much memory growth: {memory_growth / 1024 / 1024:.1f}MB"
    
    def test_repeated_formatting_performance(self):
        """Test performance of repeated formatting of same periods."""
        # Create a set of periods to format repeatedly
        periods = []
        for i in range(100):
            start = datetime(2024, 1, 1) + timedelta(days=i)
            end = start + timedelta(days=1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Time first pass
        start_time = time.time()
        first_results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            first_results.append(result)
        first_pass_time = time.time() - start_time
        
        # Time repeated passes
        repeat_times = []
        for _ in range(10):
            start_time = time.time()
            for period in periods:
                result = self.formatter.format_time_period(period)
                # Results should be consistent
                assert result in first_results
            repeat_times.append(time.time() - start_time)
        
        avg_repeat_time = sum(repeat_times) / len(repeat_times)
        
        # Repeated formatting should not be significantly slower
        # Allow up to 50% overhead for repeated calls
        assert avg_repeat_time < first_pass_time * 1.5, f"Repeated calls too slow: {avg_repeat_time:.4f}s vs {first_pass_time:.4f}s"
    
    def test_concurrent_formatting_simulation(self):
        """Test performance under simulated concurrent load."""
        import threading
        import queue
        
        # Create work queue
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add work items
        for i in range(1000):
            start = datetime(2024, 1, 1) + timedelta(days=i % 365)
            end = start + timedelta(days=1)
            work_queue.put(TimePeriod(start=start, end=end))
        
        def worker():
            """Worker function for threading test."""
            formatter = DateFormatter()  # Each thread gets its own formatter
            while True:
                try:
                    period = work_queue.get(timeout=1)
                    result = formatter.format_time_period(period)
                    result_queue.put(result)
                    work_queue.task_done()
                except queue.Empty:
                    break
        
        # Start multiple worker threads
        threads = []
        num_threads = 4
        
        start_time = time.time()
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        # Wait for all work to complete
        work_queue.join()
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Should complete in reasonable time with multiple threads
        assert elapsed_time < 5.0, f"Concurrent processing took too long: {elapsed_time:.2f} seconds"
        
        # All results should be collected
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == 1000
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
    
    def test_stress_test_edge_cases(self):
        """Stress test with many edge case scenarios."""
        periods = []
        
        # Generate various edge cases
        base_year = 2020
        
        # Month boundaries
        for year in range(base_year, base_year + 5):
            for month in range(1, 13):
                # Last day of month to first day of next month
                if month == 12:
                    start = datetime(year, month, 31)
                    end = datetime(year + 1, 1, 1)
                else:
                    last_day = 31 if month in [1, 3, 5, 7, 8, 10] else 30
                    if month == 2:
                        last_day = 29 if year % 4 == 0 else 28
                    start = datetime(year, month, last_day)
                    end = datetime(year, month + 1, 1)
                periods.append(TimePeriod(start=start, end=end))
        
        # Year boundaries
        for year in range(base_year, base_year + 5):
            start = datetime(year, 12, 31)
            end = datetime(year + 1, 1, 1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Leap year edge cases
        leap_years = [2020, 2024]
        for year in leap_years:
            # February 29
            start = datetime(year, 2, 29)
            end = datetime(year, 3, 1)
            periods.append(TimePeriod(start=start, end=end))
            
            # February month in leap year
            start = datetime(year, 2, 1)
            end = datetime(year, 3, 1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Measure formatting time for all edge cases
        start_time = time.time()
        results = []
        errors = 0
        
        for period in periods:
            try:
                result = self.formatter.format_time_period(period)
                results.append(result)
            except Exception as e:
                errors += 1
                # Use safe formatting for error cases
                result = self.formatter.safe_format_time_period(period)
                results.append(result)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Should handle all edge cases without too many errors
        error_rate = errors / len(periods)
        assert error_rate < 0.01, f"Too many errors: {error_rate:.2%}"
        
        # Should complete in reasonable time
        assert elapsed_time < 3.0, f"Edge case processing took too long: {elapsed_time:.2f} seconds"
        
        # All results should be valid
        assert len(results) == len(periods)
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
    
    def test_performance_regression_benchmark(self):
        """Benchmark test to detect performance regressions."""
        # Standard benchmark dataset
        periods = []
        
        # 1000 single days
        for i in range(1000):
            start = datetime(2024, 1, 1) + timedelta(days=i % 365)
            end = start + timedelta(days=1)
            periods.append(TimePeriod(start=start, end=end))
        
        # 100 single months
        for i in range(100):
            month = (i % 12) + 1
            year = 2024 + i // 12
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
            periods.append(TimePeriod(start=start, end=end))
        
        # 50 quarters
        for i in range(50):
            quarter = (i % 4) + 1
            year = 2024 + i // 4
            start_month = (quarter - 1) * 3 + 1
            start = datetime(year, start_month, 1)
            if start_month == 10:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, start_month + 3, 1)
            periods.append(TimePeriod(start=start, end=end))
        
        # 100 custom ranges
        for i in range(100):
            start = datetime(2024, 1, 1) + timedelta(days=i * 3)
            end = start + timedelta(days=7)
            periods.append(TimePeriod(start=start, end=end))
        
        # Run benchmark
        start_time = time.time()
        results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_periods = len(periods)
        formats_per_second = total_periods / elapsed_time
        
        # Performance targets (these should be adjusted based on baseline measurements)
        assert elapsed_time < 2.0, f"Benchmark took too long: {elapsed_time:.2f} seconds"
        assert formats_per_second > 600, f"Benchmark too slow: {formats_per_second:.0f} formats/sec"
        
        # All results should be valid
        assert len(results) == total_periods
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
        
        # Log performance metrics for monitoring
        print(f"\nPerformance Benchmark Results:")
        print(f"Total periods: {total_periods}")
        print(f"Elapsed time: {elapsed_time:.3f} seconds")
        print(f"Formats per second: {formats_per_second:.0f}")
        print(f"Average time per format: {elapsed_time / total_periods * 1000:.3f} ms")
    
    def _get_memory_usage(self):
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback to basic object count if psutil not available
            return len(gc.get_objects()) * 100  # Rough estimate


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])