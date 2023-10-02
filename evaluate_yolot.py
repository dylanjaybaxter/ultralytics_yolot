from tqdm.auto import tqdm
import time

total_items = 100
epoch = 1
for i in tqdm(range(total_items), desc="Processing", bar_format=f"{epoch} {{bar}}|{{percentage}} | {{desc}}", ncols=100,
              position=0, leave=True):
    # Your processing logic here
    time.sleep(0.1)  # Simulate some work

    # Print additional information inside the loop

# Add a description after the progress bar
tqdm.write("Finishing up...")

# Add a description after the progress bar is done
tqdm.write("Process completed.")
