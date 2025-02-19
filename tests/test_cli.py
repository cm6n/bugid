"""Tests for the command line interface."""

import pytest
from cli_tool.cli import parse_args, main

def test_parse_args_default():
    """Test argument parsing with default values."""
    args = parse_args([])
    assert args.example == 'default_value'

def test_parse_args_custom():
    """Test argument parsing with custom values."""
    args = parse_args(['--example', 'custom_value'])
    assert args.example == 'custom_value'

def test_main_success():
    """Test main function with successful execution."""
    result = main(['--example', 'test_value'])
    assert result == 0

def test_main_no_args():
    """Test main function with no arguments."""
    result = main([])
    assert result == 0
