import time
import functools
import pandas as pd
import dash
from dash import html

# Input Validation
def validate_inputs(df, required_columns, numeric_columns=None, datetime_columns=None):
    """
    Validate DataFrame inputs with comprehensive checks.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        required_columns (list): Columns that must be present
        numeric_columns (list): Columns that must be numeric
        datetime_columns (list): Columns that must be convertible to datetime
    
    Returns:
        list: List of error messages, empty if valid
    """
    errors = []
    
    # Check required columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    
    # Check data types
    if numeric_columns:
        non_numeric = [col for col in numeric_columns 
                      if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            errors.append(f"Non-numeric data in columns: {', '.join(non_numeric)}")
            
    if datetime_columns:
        for col in datetime_columns:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col])
                except Exception:
                    errors.append(f"Could not convert {col} to datetime")
    
    # Check minimum data points
    if len(df) < 2:
        errors.append("At least two data points required")
    
    return errors


def parse_weights(weight_str):
    """
    Parse weights string with comprehensive error handling
    
    Args:
        weight_str (str): Comma-separated weights
    
    Returns:
        tuple: (weights list, error message) 
    """
    if not weight_str:
        return [], "Weight string is empty"
    
    try:
        weights = [float(w.strip()) for w in weight_str.split(',')]
        return weights, ""
    except ValueError:
        return [], f"Invalid weight format: '{weight_str}'. Use comma-separated numbers"


# Timing Decorator
def timed_computation(func):
    """Decorator to time computations and store result"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


# Alert System
def create_alert(message, alert_type='success'):
    """Create a styled alert component"""
    styles = {
        'success': {'background': '#e8f5e9', 'color': '#2e7d32'},
        'error': {'background': '#ffebee', 'color': '#c62828'},
        'warning': {'background': '#fff8e1', 'color': '#f9a825'},
        'info': {'background': '#e3f2fd', 'color': '#1565c0'}
    }
    style = {
        'padding': '10px',
        'borderRadius': '5px',
        'margin': '10px 0'
    }
    style.update(styles.get(alert_type, styles['info']))
    return html.Div(message, style=style)


# Global Exception Handler
def global_catch_exception(func):
    """Decorator to catch exceptions in callbacks and return user-friendly message"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the full exception
            print(f"Error in {func.__name__}: {str(e)}")
            # Return user-friendly error message
            return create_alert(f"Error: {str(e)}", 'error')
    return wrapper
