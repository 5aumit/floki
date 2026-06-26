# Where should we write the output files?
Output demo.gif
Output demo.mp4

# Set up a crisp, standard-sized terminal window
Set FontSize 14
Set Width 1200
Set Height 650
Set Padding 20

# --- Step 1: Manual Conda Initialization ---
# This bypasses the VHS shell argument bug by manually sourcing your profile
Type "source ~/.bashrc"
Enter
Sleep 2s

# Now activate your local environment path
Type "conda activate ./mlflow_agentic_env"
Enter
Sleep 1s

# Clear the screen so the initialization commands don't show up in your demo
# Ctrl+L
Sleep 500ms

# --- Step 2: Launch the Agent ---
Type "./run_agent.sh"
Enter

# Wait for agent to load completely (75 seconds)
Sleep 75s

# Clear the screen buffer so the user query starts on a pristine canvas
Ctrl+L
Sleep 500ms

# --- Step 3: Run the User Query ---
Type "What are the architectures or algorithms tried in the customer churn experiment? How did the hyperparameters affect the performance? Make a brief report."
Enter

# Wait for the agent to stream/generate the output
Sleep 10000ms

# Final pause to let the viewer read the output on screen
Sleep 4s

# --- Step 4: Clean Exit ---
Type "exit"
Enter
Sleep 1s