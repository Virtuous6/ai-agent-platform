"""
Terminal Colors and Styling

Retro terminal color schemes and styling utilities for the dashboard.
Supports both ANSI escape codes and curses color pairs.
"""

class TerminalColors:
    """ANSI color codes for terminal styling."""
    
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Reset
    RESET = '\033[0m'
    
    # Retro color schemes
    RETRO_SCHEMES = {
        "green_terminal": {
            "primary": GREEN,
            "secondary": BRIGHT_GREEN,
            "accent": YELLOW,
            "warning": BRIGHT_YELLOW,
            "error": BRIGHT_RED,
            "info": CYAN,
            "background": BLACK
        },
        
        "amber_terminal": {
            "primary": YELLOW,
            "secondary": BRIGHT_YELLOW,
            "accent": BRIGHT_WHITE,
            "warning": RED,
            "error": BRIGHT_RED,
            "info": BRIGHT_CYAN,
            "background": BLACK
        },
        
        "blue_terminal": {
            "primary": CYAN,
            "secondary": BRIGHT_CYAN,
            "accent": WHITE,
            "warning": YELLOW,
            "error": RED,
            "info": BLUE,
            "background": BLACK
        },
        
        "matrix": {
            "primary": BRIGHT_GREEN,
            "secondary": GREEN,
            "accent": WHITE,
            "warning": BRIGHT_YELLOW,
            "error": BRIGHT_RED,
            "info": CYAN,
            "background": BLACK
        }
    }
    
    def __init__(self, scheme="green_terminal"):
        """Initialize with a color scheme."""
        self.scheme = self.RETRO_SCHEMES.get(scheme, self.RETRO_SCHEMES["green_terminal"])
    
    def colorize(self, text: str, color: str) -> str:
        """Colorize text with ANSI codes."""
        return f"{color}{text}{self.RESET}"
    
    def status_color(self, value: float) -> str:
        """Get status color based on value (0.0-1.0)."""
        if value >= 0.8:
            return self.scheme["primary"]  # Good
        elif value >= 0.6:
            return self.scheme["warning"]  # Warning
        else:
            return self.scheme["error"]    # Critical
    
    def gradient_bar(self, value: float, width: int = 20) -> str:
        """Create a gradient progress bar."""
        filled = int(value * width)
        empty = width - filled
        
        bar = "█" * filled + "░" * empty
        color = self.status_color(value)
        
        return f"{color}[{bar}]{self.RESET} {value:.1%}"
    
    def sparkline(self, values: list, width: int = 20) -> str:
        """Create a sparkline chart with colors."""
        if not values or len(values) < 2:
            return "No data"
        
        max_val = max(values)
        min_val = min(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Map values to spark characters
        spark_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        
        sparkline = ""
        for value in values[-width:]:  # Take last 'width' values
            normalized = (value - min_val) / range_val
            char_index = min(int(normalized * (len(spark_chars) - 1)), len(spark_chars) - 1)
            
            # Color based on trend
            if value > (max_val * 0.8):
                color = self.scheme["primary"]
            elif value > (max_val * 0.6):
                color = self.scheme["warning"]
            else:
                color = self.scheme["error"]
            
            sparkline += f"{color}{spark_chars[char_index]}{self.RESET}"
        
        return sparkline
    
    def format_metric(self, label: str, value: any, unit: str = "", 
                     status_value: float = None) -> str:
        """Format a metric with color coding."""
        if status_value is not None:
            color = self.status_color(status_value)
        else:
            color = self.scheme["primary"]
        
        return f"{self.scheme['secondary']}{label}:{self.RESET} {color}{value}{unit}{self.RESET}"
    
    def format_table_row(self, *columns, status: str = "normal") -> str:
        """Format a table row with alternating colors."""
        color_map = {
            "normal": self.WHITE,
            "good": self.scheme["primary"],
            "warning": self.scheme["warning"],
            "error": self.scheme["error"],
            "header": self.scheme["accent"]
        }
        
        color = color_map.get(status, self.WHITE)
        formatted_columns = [f"{color}{col}{self.RESET}" for col in columns]
        
        return " │ ".join(formatted_columns)

# Curses color pair definitions
CURSES_COLOR_PAIRS = {
    1: "GREEN_ON_BLACK",      # Success/Good
    2: "RED_ON_BLACK",        # Error/Critical  
    3: "YELLOW_ON_BLACK",     # Warning
    4: "CYAN_ON_BLACK",       # Info/Headers
    5: "MAGENTA_ON_BLACK",    # Highlights
    6: "WHITE_ON_BLACK",      # Normal text
    7: "BRIGHT_GREEN_ON_BLACK", # Active elements
    8: "BRIGHT_RED_ON_BLACK",   # Critical alerts
}

# Status indicators with colors
STATUS_INDICATORS = {
    "excellent": ("●●●●●", "green"),
    "good": ("●●●●○", "green"),
    "warning": ("●●●○○", "yellow"),
    "critical": ("●●○○○", "red"),
    "offline": ("○○○○○", "red")
}

# Loading animations
LOADING_ANIMATIONS = {
    "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "bars": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"],
    "pulse": ["●", "◐", "○", "◑"],
    "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]
}

def get_status_indicator(value: float) -> tuple:
    """Get status indicator and color for a value."""
    if value >= 0.9:
        return STATUS_INDICATORS["excellent"]
    elif value >= 0.7:
        return STATUS_INDICATORS["good"]
    elif value >= 0.5:
        return STATUS_INDICATORS["warning"]
    elif value >= 0.1:
        return STATUS_INDICATORS["critical"]
    else:
        return STATUS_INDICATORS["offline"]

def format_bytes(bytes_value: int) -> str:
    """Format byte values with colors."""
    colors = TerminalColors()
    
    if bytes_value < 1024:
        return colors.colorize(f"{bytes_value}B", colors.GREEN)
    elif bytes_value < 1024 * 1024:
        return colors.colorize(f"{bytes_value/1024:.1f}KB", colors.YELLOW)
    elif bytes_value < 1024 * 1024 * 1024:
        return colors.colorize(f"{bytes_value/(1024*1024):.1f}MB", colors.RED)
    else:
        return colors.colorize(f"{bytes_value/(1024*1024*1024):.1f}GB", colors.BRIGHT_RED)

def format_duration(seconds: float) -> str:
    """Format duration with colors."""
    colors = TerminalColors()
    
    if seconds < 1:
        return colors.colorize(f"{seconds*1000:.0f}ms", colors.GREEN)
    elif seconds < 60:
        return colors.colorize(f"{seconds:.1f}s", colors.YELLOW)
    elif seconds < 3600:
        return colors.colorize(f"{seconds/60:.1f}m", colors.RED)
    else:
        return colors.colorize(f"{seconds/3600:.1f}h", colors.BRIGHT_RED)

def format_currency(amount: float) -> str:
    """Format currency with colors."""
    colors = TerminalColors()
    
    if amount < 0.01:
        return colors.colorize(f"${amount:.4f}", colors.GREEN)
    elif amount < 1.0:
        return colors.colorize(f"${amount:.2f}", colors.YELLOW)
    elif amount < 10.0:
        return colors.colorize(f"${amount:.2f}", colors.RED)
    else:
        return colors.colorize(f"${amount:.2f}", colors.BRIGHT_RED) 