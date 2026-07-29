# VHS tape: One Dark TUI demo for README assets
Output assets/floki-demo.gif

Set Shell bash
Set FontSize 14
Set Width 1200
Set Height 700
Set Padding 12
Set Theme "OneDark"
Set TypingSpeed 40ms

# Hide launch + import from the GIF; force truecolor (agent env may set NO_COLOR=1)
Hide
Type "unset NO_COLOR FORCE_COLOR; export COLORTERM=truecolor TERM=xterm-256color FORCE_COLOR=1"
Enter
Type "./mlflow_agentic_env/bin/python docs/artifacts/TUI-mockups/demo_tui_launcher.py"
Enter
Sleep 45s
Show

# Hold on welcome / banner
Sleep 2.5s

Type "What experiments are in the tracking server?"
Enter

# thinking → listing → scanning → counting → formatting → answer
Sleep 6s

# Let viewer read the answer
Sleep 4s

Type "exit"
Enter
Sleep 800ms
