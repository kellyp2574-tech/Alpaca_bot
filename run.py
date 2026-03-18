"""
Entry point for the 0DTE Options Bot.
Run daily at 9:00 AM ET.

Usage:
    python run.py              # Live/paper trading
    python run.py --dry-run    # Log signals without submitting orders
"""
from bot.integrated_main import main

if __name__ == "__main__":
    main()
