#!/usr/bin/env python3
"""Command line interface for the tool."""

import argparse
import sys
from typing import List, Optional

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Description of your command line tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments here
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {__import__("cli_tool").__version__}'
    )
    
    parser.add_argument(
        '--example',
        help='Example argument',
        type=str,
        default='default_value'
    )
    
    return parser.parse_args(args)

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the command line tool."""
    parsed_args = parse_args(args)
    
    try:
        # Your main logic here
        print(f"Example argument value: {parsed_args.example}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
