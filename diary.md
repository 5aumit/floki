# Diary

This is a personal log/thought dump I'm trying to maintain throughout the development of this project. I admittedly did not start this when the project began, but I think its a good way to be transparent about my thoughts as it progresses.

# Entires

## 4/12/2026

There's already quite a few issues that I've set up for me to work on. But one that I think I should prioritize is tracing and observability. Often, the LLM makes small mistakes like passing an incorrect data type as an argument for a tool. Today, I asked it:
`tell me about the various runs and their differences in the neural network experiment`

There are 2 experiments in my `mlruns`, one which is named `neural_network_digit_classification` and one named `digit_classification`. What I expected the agent to do was to identify which one I'm referring to, grab the name and query its data using the experiment ID. There's tools defined for it, which I will have to revisit and check. I believe it could be happening because the model (`llama-3.1-8b-instant` right now) is stupid, but this is too simple for it to make that mistake. 

Hence, tracing is important.

Challenge now is how to integrate tracing while keeping a budget of $0. The API I'm using is Groq, and LangFuse is free. Need to figure out how these 2 can be integrated.

## 4/17/2026

Just started using Copilot CLI. Its pretty cool.

LangFuse initialization isn't failing anymore, but the traces don't show up on the tool. Need to fix that.