import os
from tensorboard import program
import sys

def start_tensorboard(event_file_path, port):
    # Ensure the event file path exists
    if not os.path.exists(event_file_path):
        print(f"Error: The file {event_file_path} does not exist.")
        return

    # Get the directory containing the events file
    log_dir = os.path.dirname(event_file_path)

    # Start TensorBoard program
    tb = program.TensorBoard()

    # Configure TensorBoard with the log directory and port
    tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port)])

    # Launch TensorBoard and print the URL
    url = tb.launch()
    print(f"TensorBoard is running at {url}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python start_tensorboard.py <event_file_path> <port_number>")
    else:
        event_file_path = sys.argv[1]
        port = int(sys.argv[2])
        start_tensorboard(event_file_path, port)
