"""Drive application."""

from commands   import run_task
from utils      import ARGS, BANNER, LOGGER

if __name__ == "__main__":
    """Execute command."""
    
    LOGGER.info(BANNER)
    
    # Match command
    match ARGS.cmd:
        
        # Run task
        case "run-task":    run_task(**vars(ARGS))
        
        # Invalid command
        case _:             LOGGER.error(f"Invalid command provided: {ARGS.cmd}")