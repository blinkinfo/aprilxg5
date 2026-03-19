"""Entry point for BTC 5m Signal Bot."""
import asyncio
import sys

from src.bot import run_bot


def main():
    """Run the bot."""
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
