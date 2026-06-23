# Design Document

## Overview

This design addresses the inconsistent date formatting issue in the AWS Cost Explorer CLI by creating a centralized, intelligent date formatting system. The solution will replace the current inconsistent date display with human-readable formats that automatically adapt to the time period type (monthly, yearly, quarterly, daily, or custom ranges).

## Architecture

The date formatting improvement will be implemented through a centralized utility that integrates with the existing response formatting system:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Response          │────│  DateFormatter       │────│   TimePeriod        │
│   Formatters        │    │  (New Utility)       │    │   (Existing Model)  │
│   (Existing)        │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
         │                           │                           │
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│ SimpleFormatter     │    │ PeriodTypeDetector   │    │ FormatRuleEngine    │
│ RichFormatter       │    │ (New Component)      │    │ (New Component)     │
│ LLMFormatter        │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## Components and Interfaces

### 1. DateFormatter Utility
**Location:** `src/aws_cost_cli/date_formatter.py`
**Responsibilities:**
- Centralized date formatting logic
- Period type detection and classification
- Format rule application
- Fallback handling for edge cases

**Key Classes:**
```python
class DateFormatter:
    def format_time_period(self, time_period: TimePeriod) -> str
    def format_time_period_with_context(self, time_period: TimePeriod, context: str) -> str
    def get_period_type(self, time_period: TimePeriod) -> PeriodType

class PeriodType(Enum):
    SINGLE_DAY = "single_day"
    SINGLE_MONTH = "single_month"
    SINGLE_QUARTER = "single_quarter"
    SINGLE_YEAR = "single_year"
    MULTI_MONTH = "multi_month"
    CUSTOM_RANGE = "custom_range"
```

### 2. Period Type Detection
**Responsibilities:**
- Analyze TimePeriod objects to determine their type
- Handle edge cases and boundary conditions
- Support fiscal year and quarter detection

**Key Methods:**
```python
class PeriodTypeDetector:
    def detect_period_type(self, time_period: TimePeriod) -> PeriodType
    def is_single_month(self, time_period: TimePeriod) -> bool
    def is_single_quarter(self, time_period: TimePeriod) -> bool
    def is_single_year(self, time_period: TimePeriod) -> bool
    def is_fiscal_period(self, time_period: TimePeriod) -> bool
```

### 3. Format Rule Engine
**Responsibilities:**
- Apply formatting rules based on period type
- Handle localization and customization
- Provide fallback formatting options

**Format Rules:**
```python
class FormatRules:
    SINGLE_DAY = "{month} {day}, {year}"  # "January 15, 2025"
    SINGLE_MONTH = "{month} {year}"       # "January 2025"
    SINGLE_QUARTER = "Q{quarter} {year}"  # "Q1 2025"
    SINGLE_YEAR = "{year}"                # "2025"
    MULTI_MONTH = "{start_month} {start_year} - {end_month} {end_year}"
    CUSTOM_RANGE = "{start_month} {start_day}, {start_year} - {end_month} {end_day}, {end_year}"
```

## Data Models

### Enhanced TimePeriod Handling
The existing `TimePeriod` model will be extended with utility methods:

```python
# Extension to existing TimePeriod class
class TimePeriod:
    # Existing fields: start, end
    
    @property
    def duration_days(self) -> int:
        return (self.end - self.start).days
    
    @property
    def is_single_day(self) -> bool:
        return self.duration_days == 1
    
    @property
    def spans_month_boundary(self) -> bool:
        return self.start.month != self.end.month or self.start.year != self.end.year
```

### Formatting Context
```python
@dataclass
class FormattingContext:
    output_format: str  # "simple", "rich", "llm"
    locale: str = "en_US"
    fiscal_year_start: int = 1  # January
    timezone: str = "UTC"
```

## Error Handling

### Graceful Degradation
- **Invalid Date Ranges:** Fall back to ISO date format with warning
- **Timezone Issues:** Use UTC as default with clear indication
- **Locale Problems:** Fall back to English month names
- **Formatting Failures:** Return original date range format

### Error Recovery Strategy
```python
def safe_format_time_period(time_period: TimePeriod) -> str:
    try:
        return DateFormatter().format_time_period(time_period)
    except Exception as e:
        logger.warning(f"Date formatting failed: {e}")
        return f"{time_period.start.strftime('%Y-%m-%d')} to {time_period.end.strftime('%Y-%m-%d')}"
```

## Testing Strategy

### Unit Testing
- **Period Type Detection:** Test all period types with various date ranges
- **Format Rule Application:** Verify correct formatting for each period type
- **Edge Cases:** Test month boundaries, year boundaries, leap years
- **Error Handling:** Test graceful degradation scenarios

### Integration Testing
- **Response Formatter Integration:** Test with existing SimpleResponseFormatter, RichResponseFormatter, and LLMResponseFormatter
- **Real Data Testing:** Test with actual AWS Cost Explorer API responses
- **Cross-Format Consistency:** Verify consistent formatting across all output formats

### Test Cases
```python
class TestDateFormatter:
    def test_single_month_formatting(self):
        # January 1-31, 2025 -> "January 2025"
        
    def test_single_quarter_formatting(self):
        # Q1 2025 (Jan-Mar) -> "Q1 2025"
        
    def test_year_boundary_spanning(self):
        # Dec 2024 - Jan 2025 -> "December 2024 - January 2025"
        
    def test_partial_month_formatting(self):
        # Jan 15-25, 2025 -> "January 15, 2025 - January 25, 2025"
```

## Implementation Details

### Integration Points
1. **SimpleResponseFormatter:** Replace `_format_time_period` method calls
2. **RichResponseFormatter:** Update table and panel formatting
3. **LLMResponseFormatter:** Provide formatted dates in cost summary data
4. **Export Functions:** Include formatted dates in CSV/JSON exports

### Backward Compatibility
- Maintain existing `_format_time_period` methods as deprecated
- Provide migration path for cached data
- Support configuration option to use legacy formatting

### Performance Considerations
- Cache period type detection results
- Minimize datetime operations
- Use efficient string formatting methods
- Lazy evaluation for complex formatting rules

## Configuration Options

### User Preferences
```yaml
date_formatting:
  enabled: true
  format_style: "smart"  # "smart", "verbose", "compact"
  fiscal_year_start: 1   # January
  locale: "en_US"
  fallback_to_iso: true
```

### Format Styles
- **Smart:** Automatically choose the most appropriate format
- **Verbose:** Always include full context (e.g., "January 1-31, 2025")
- **Compact:** Use shortest reasonable format (e.g., "Jan 2025")

## Migration Strategy

### Phase 1: Core Implementation
- Implement DateFormatter utility
- Add unit tests
- Create integration with SimpleResponseFormatter

### Phase 2: Full Integration
- Integrate with RichResponseFormatter and LLMResponseFormatter
- Add configuration options
- Update export functions

### Phase 3: Enhancement
- Add localization support
- Implement fiscal year handling
- Add advanced formatting options

## Security Considerations

### Data Privacy
- No sensitive data is processed in date formatting
- Formatting operations are purely computational
- No external API calls required

### Input Validation
- Validate TimePeriod objects before formatting
- Handle malformed date ranges gracefully
- Prevent injection attacks through date strings (though unlikely in this context)